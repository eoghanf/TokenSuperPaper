"""
Token Superposition Training experiment — arXiv 2605.06546
GPT-2-small (124M) trained from scratch on FineWeb-Edu.
Two runs at equal FLOPs, 20000 steps each:
  - Baseline: standard next-token CE
  - TST:      s=6, r=0.3 (6000 superposition steps + 14000 recovery steps)

Data: pre-tokenized FineWeb-Edu shards from fineweb-volume (nanoGPT uint16 format).
"""

import math
import time

import modal
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache     = modal.Volume.from_name("hf-cache",       create_if_missing=True)
fineweb_vol  = modal.Volume.from_name("fineweb-volume",  create_if_missing=False)

FINEWEB_DIR = Path("/fineweb")
BYTES_PER_TOKEN = 4.0  # GPT-2 BPE on English, approximate

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "wandb==0.18.6",
        "huggingface_hub[hf_transfer]==0.26.2",
        "numpy==1.26.4",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_python_source("src")
)

app = modal.App("tst-experiment", image=image)

# ── Training helpers ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    run_name: str
    model_size: str  = "small"   # "small" (124M) or "medium" (345M)
    bag_size: int    = 1
    tst_ratio: float = 0.3
    total_steps: int = 20000
    latent_seq: int  = 1024
    batch_size: int  = 32
    lr: float        = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    grad_clip: float  = 1.0
    log_every: int    = 25
    eval_every: int   = 250
    gradient_checkpointing: bool = False   # enable for larger models to free activation memory


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-Stable-Decay: linear warmup, stable plateau, linear decay to 10% LR."""
    min_lr = cfg.lr * 0.1
    if step < cfg.warmup_steps:
        return min_lr + (cfg.lr - min_lr) * (step + 1) / cfg.warmup_steps
    decay_start = int(0.9 * cfg.total_steps)
    if step < decay_start:
        return cfg.lr
    t = (step - decay_start) / max(1, cfg.total_steps - decay_start)
    return cfg.lr - (cfg.lr - min_lr) * t


class TokenDataset:
    """Reads consecutive non-overlapping chunks from a pre-tokenized uint16 array."""

    def __init__(self, tokens, seq_len: int, batch_size: int, start: int = 0):
        self.tokens = tokens
        self.seq_len = seq_len
        self.bs = batch_size
        self.stride = batch_size * (seq_len + 1)
        self.pos = start

    def next_batch(self):
        import numpy as np
        if self.pos + self.stride > len(self.tokens):
            self.pos = 0
        chunk = self.tokens[self.pos: self.pos + self.stride].astype(np.int64)
        self.pos += self.stride
        chunk = chunk.reshape(self.bs, self.seq_len + 1)
        x = torch.tensor(chunk[:, :-1], dtype=torch.long)
        y = torch.tensor(chunk[:, 1:],  dtype=torch.long)
        return x, y

    @property
    def tokens_consumed(self) -> int:
        return self.pos


def evaluate(model, val_tokens, batch_size: int, device: torch.device) -> tuple:
    """Returns (ce_nats, bpb). Always evaluates in standard AR mode (bag_size=1)."""
    from src.model import compute_loss
    import numpy as np
    seq_len = 1024
    n = seq_len + 1
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_tokens) - n * batch_size, n * batch_size):
            chunk = val_tokens[i: i + n * batch_size].astype(np.int64)
            chunk = chunk.reshape(batch_size, n)
            x = torch.tensor(chunk[:, :-1], dtype=torch.long, device=device)
            y = torch.tensor(chunk[:, 1:],  dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, bag_size=1)
                loss = compute_loss(logits, y, bag_size=1)
            losses.append(loss.item())
            if len(losses) >= 50:
                break
    model.train()
    ce = sum(losses) / max(len(losses), 1)
    bpb = ce / math.log(2) / BYTES_PER_TOKEN
    return ce, bpb


def train_run(
    cfg: TrainConfig,
    device: torch.device,
    train_tokens,
    val_tokens,
    target_val_ce: float = None,   # if set, log the step where val CE first crosses below this
) -> float:
    import wandb
    from src.model import TSTModel, compute_loss

    model = TSTModel(size=cfg.model_size).to(device)
    if cfg.gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable()
        print(f"[{cfg.run_name}] Gradient checkpointing enabled")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.run_name}] {n_params / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    wandb.init(
        project="token-superposition",
        name=cfg.run_name,
        config=cfg.__dict__,
        group="tst-vs-baseline-v2",
        reinit=True,
    )

    # Custom x-axis metrics — enables 8 wandb panels
    wandb.define_metric("tokens_processed")
    wandb.define_metric("total_flops")
    wandb.define_metric("wall_clock_s")
    wandb.define_metric("train/loss_vs_tokens", step_metric="tokens_processed")
    wandb.define_metric("train/loss_vs_flops",  step_metric="total_flops")
    wandb.define_metric("train/loss_vs_time",   step_metric="wall_clock_s")
    wandb.define_metric("val/ce_vs_tokens",     step_metric="tokens_processed")
    wandb.define_metric("val/bpb_vs_tokens",    step_metric="tokens_processed")
    wandb.define_metric("val/ce_vs_flops",      step_metric="total_flops")
    wandb.define_metric("val/bpb_vs_flops",     step_metric="total_flops")
    wandb.define_metric("val/ce_vs_time",       step_metric="wall_clock_s")
    wandb.define_metric("val/bpb_vs_time",      step_metric="wall_clock_s")

    tst_steps      = int(cfg.total_steps * cfg.tst_ratio) if cfg.bag_size > 1 else 0
    recovery_steps = cfg.total_steps - tst_steps
    global_step    = 0
    total_tokens   = 0   # data tokens consumed (6× per step during TST phase)
    total_flops    = 0   # forward+backward FLOPs (constant per step, equal across runs)
    flops_per_step = 6 * n_params * cfg.batch_size * cfg.latent_seq
    start_time     = time.time()
    iso_loss_step  = None  # first step where val CE <= target_val_ce

    def run_phase(data_seq: int, n_steps: int, bag: int, phase: str, data_start: int = 0):
        nonlocal global_step, total_tokens, total_flops, iso_loss_step
        tokens_per_step = cfg.batch_size * data_seq
        ds = TokenDataset(train_tokens, data_seq, cfg.batch_size, start=data_start)
        model.train()
        for _ in range(n_steps):
            lr = get_lr(global_step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = ds.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, bag_size=bag)
                loss = compute_loss(logits, y, bag_size=bag)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_tokens += tokens_per_step
            total_flops  += flops_per_step
            global_step  += 1
            elapsed = time.time() - start_time

            if global_step % cfg.log_every == 0:
                wandb.log({
                    "train/loss":           loss.item(),
                    "train/loss_vs_tokens": loss.item(),
                    "train/loss_vs_flops":  loss.item(),
                    "train/loss_vs_time":   loss.item(),
                    "tokens_processed": total_tokens,
                    "total_flops":      total_flops,
                    "wall_clock_s":     elapsed,
                    "lr": lr,
                }, step=global_step)

            if global_step % cfg.eval_every == 0:
                ce, bpb = evaluate(model, val_tokens, cfg.batch_size, device)
                elapsed = time.time() - start_time
                log_dict = {
                    "val/ce":            ce,
                    "val/bpb":           bpb,
                    "val/ce_vs_tokens":  ce,
                    "val/bpb_vs_tokens": bpb,
                    "val/ce_vs_flops":   ce,
                    "val/bpb_vs_flops":  bpb,
                    "val/ce_vs_time":    ce,
                    "val/bpb_vs_time":   bpb,
                    "tokens_processed": total_tokens,
                    "total_flops":      total_flops,
                    "wall_clock_s":     elapsed,
                }
                if target_val_ce is not None and iso_loss_step is None and ce <= target_val_ce:
                    iso_loss_step = global_step
                    log_dict["iso_loss/step"]    = iso_loss_step
                    log_dict["iso_loss/wall_clock_s"] = elapsed
                    log_dict["iso_loss/val_ce"]  = ce
                    print(f"  [{cfg.run_name}] *** ISO-LOSS at step={iso_loss_step} "
                          f"val_ce={ce:.4f} (target={target_val_ce:.4f}) t={elapsed:.0f}s ***")
                wandb.log(log_dict, step=global_step)
                print(f"  [{cfg.run_name}] step={global_step} phase={phase} "
                      f"train={loss.item():.4f} val_ce={ce:.4f} val_bpb={bpb:.4f}")

        return ds.tokens_consumed

    # Phase 1: TST superposition
    offset_after_tst = 0
    if tst_steps > 0:
        data_seq_tst = cfg.latent_seq * cfg.bag_size
        print(f"[{cfg.run_name}] Phase 1 TST: s={cfg.bag_size}, "
              f"{tst_steps} steps, data_seq={data_seq_tst}")
        offset_after_tst = run_phase(data_seq_tst, tst_steps, cfg.bag_size, "tst", data_start=0)

    # Phase 2: standard AR recovery (or plain baseline)
    print(f"[{cfg.run_name}] Phase 2: {recovery_steps} steps, data_seq={cfg.latent_seq}")
    run_phase(cfg.latent_seq, recovery_steps, 1, "recovery", data_start=offset_after_tst)

    ce, bpb = evaluate(model, val_tokens, cfg.batch_size, device)
    elapsed = time.time() - start_time
    wandb.log({
        "val/ce_final":      ce,
        "val/bpb_final":     bpb,
        "val/ce":            ce,
        "val/bpb":           bpb,
        "val/ce_vs_tokens":  ce,
        "val/bpb_vs_tokens": bpb,
        "val/ce_vs_flops":   ce,
        "val/bpb_vs_flops":  bpb,
        "val/ce_vs_time":    ce,
        "val/bpb_vs_time":   bpb,
        "tokens_processed": total_tokens,
        "total_flops":      total_flops,
        "wall_clock_s":     elapsed,
    }, step=global_step)
    wandb.finish()
    print(f"[{cfg.run_name}] Final val CE={ce:.4f} BPB={bpb:.4f}")
    return ce


# ── GPU training functions ────────────────────────────────────────────────────

_gpu_kwargs = dict(
    gpu="H100",
    timeout=14400,
    scaledown_window=2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/fineweb": fineweb_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)


HEADER_BYTES = 1024  # build-nanogpt format: 256×int32 header before token data

def _load_shard(path) -> "np.ndarray":
    import numpy as np
    return np.fromfile(path, dtype=np.uint16, offset=HEADER_BYTES)

def _load_data():
    """Load all finewebedu train shards + val shard from fineweb-volume."""
    import numpy as np
    shards = sorted(FINEWEB_DIR.glob("finewebedu_train_*.bin"))
    print(f"Found {len(shards)} train shards: {[s.name for s in shards]}")
    train_tokens = np.concatenate([_load_shard(s) for s in shards])
    val_tokens = _load_shard(FINEWEB_DIR / "finewebedu_val_000000.bin")
    print(f"Train tokens: {len(train_tokens):,}  Val tokens: {len(val_tokens):,}")
    print(f"Max train token ID: {train_tokens.max()}  Max val token ID: {val_tokens.max()}")
    return train_tokens, val_tokens


_gpu_kwargs_long = dict(
    gpu="H100",
    timeout=86400,   # 24h — baseline medium may run ~12h
    scaledown_window=2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/fineweb": fineweb_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)


@app.function(**_gpu_kwargs)
def run_baseline():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_tokens, val_tokens = _load_data()
    return train_run(
        TrainConfig(run_name="baseline-gpt2s-20k-v2"),
        device=device, train_tokens=train_tokens, val_tokens=val_tokens,
    )


@app.function(**_gpu_kwargs)
def run_tst():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_tokens, val_tokens = _load_data()
    return train_run(
        TrainConfig(run_name="tst-s6-r03-gpt2s-20k-v2", bag_size=6, tst_ratio=0.3),
        device=device, train_tokens=train_tokens, val_tokens=val_tokens,
    )


@app.function(**_gpu_kwargs_long)
def run_tst_medium():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_tokens, val_tokens = _load_data()
    return train_run(
        TrainConfig(
            run_name="tst-s6-r03-gpt2m-20k",
            model_size="medium",
            bag_size=6,
            tst_ratio=0.3,
            total_steps=20000,
            gradient_checkpointing=True,
        ),
        device=device, train_tokens=train_tokens, val_tokens=val_tokens,
    )


@app.function(**_gpu_kwargs_long)
def run_baseline_medium(target_val_ce: float):
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target val CE (from TST medium): {target_val_ce:.4f}")
    train_tokens, val_tokens = _load_data()
    return train_run(
        TrainConfig(
            run_name="baseline-gpt2m-40k",
            model_size="medium",
            bag_size=1,
            total_steps=40000,
            warmup_steps=400,
            gradient_checkpointing=True,
        ),
        device=device, train_tokens=train_tokens, val_tokens=val_tokens,
        target_val_ce=target_val_ce,
    )


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Step 1: TST medium (GPT-2 medium, 20k steps) ...")
    tst_loss = run_tst_medium.remote()
    print(f"\nTST medium final val CE: {tst_loss:.4f}")
    print(f"\nStep 2: Baseline medium (GPT-2 medium, 60k steps) — iso-loss target: {tst_loss:.4f} ...")
    baseline_loss = run_baseline_medium.remote(target_val_ce=tst_loss)
    print(f"\n{'=' * 50}")
    print(f"TST (20k steps):       {tst_loss:.4f} nats")
    print(f"Baseline (60k steps):  {baseline_loss:.4f} nats")
    print(f"{'=' * 50}")
    print("Check wandb for iso_loss/step to read the speedup ratio.")
