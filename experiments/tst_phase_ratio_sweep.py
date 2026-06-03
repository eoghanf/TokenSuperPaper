"""
TST Phase Ratio Sweep — Experiment 6.

Five variants of TST, all with total_steps=20000, s=6, GPT-2 medium.
Only the superposition/recovery split varies: 20%, 40%, 60%, 80%, 120%
of the reference 6000-step superposition phase (tst_ratio=0.30).

All five runs launch in parallel, each on 1×H100.
"""

import dataclasses
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache    = modal.Volume.from_name("hf-cache",       create_if_missing=True)
fineweb_vol = modal.Volume.from_name("fineweb-volume",  create_if_missing=False)
ckpt_vol    = modal.Volume.from_name("tst-checkpoints", create_if_missing=True)

FINEWEB_DIR   = Path("/fineweb")
CKPT_DIR_ROOT = Path("/checkpoints")
BYTES_PER_TOKEN = 4.0
HEADER_BYTES    = 1024   # build-nanogpt .bin format

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

app = modal.App("tst-phase-ratio-sweep", image=image)

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    run_name:     str
    tst_ratio:    float          # fraction of total_steps in superposition
    total_steps:  int   = 20000
    bag_size:     int   = 6
    latent_seq:   int   = 1024
    batch_size:   int   = 32
    lr:           float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int   = 200
    grad_clip:    float = 1.0
    log_every:    int   = 25
    eval_every:   int   = 250


VARIANTS = [
    TrainConfig(run_name="tst-r006", tst_ratio=0.06),  # 1200 sup steps — 20% of 6k
    TrainConfig(run_name="tst-r012", tst_ratio=0.12),  # 2400 sup steps — 40% of 6k
    TrainConfig(run_name="tst-r018", tst_ratio=0.18),  # 3600 sup steps — 60% of 6k
    TrainConfig(run_name="tst-r024", tst_ratio=0.24),  # 4800 sup steps — 80% of 6k
    TrainConfig(run_name="tst-r036", tst_ratio=0.36),  # 7200 sup steps — 120% of 6k
]

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-Stable-Decay: linear warmup, stable, linear decay to 10% LR."""
    min_lr = cfg.lr * 0.1
    if step < cfg.warmup_steps:
        return min_lr + (cfg.lr - min_lr) * (step + 1) / cfg.warmup_steps
    decay_start = int(0.9 * cfg.total_steps)
    if step < decay_start:
        return cfg.lr
    t = (step - decay_start) / max(1, cfg.total_steps - decay_start)
    return cfg.lr - (cfg.lr - min_lr) * t

# ── Data ──────────────────────────────────────────────────────────────────────

def _load_shard(path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16, offset=HEADER_BYTES)

def _load_data():
    shards = sorted(FINEWEB_DIR.glob("finewebedu_train_*.bin"))
    print(f"Found {len(shards)} train shards")
    train_tokens = np.concatenate([_load_shard(s) for s in shards])
    val_tokens   = _load_shard(FINEWEB_DIR / "finewebedu_val_000000.bin")
    print(f"Train: {len(train_tokens):,}  Val: {len(val_tokens):,}")
    return train_tokens, val_tokens

# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, val_tokens, batch_size: int, device) -> tuple[float, float]:
    """Val CE and BPB; always runs in standard AR mode (bag_size=1)."""
    from src.model import compute_loss
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
                loss   = compute_loss(logits, y, bag_size=1)
            losses.append(loss.item())
            if len(losses) >= 50:
                break
    model.train()
    ce  = sum(losses) / max(len(losses), 1)
    bpb = ce / math.log(2) / BYTES_PER_TOKEN
    return ce, bpb

# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(step, model, optimizer, data_pos, total_tokens,
                    total_flops, elapsed_s, wandb_run_id, ckpt_dir):
    state = {
        "step":           step,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "data_pos":       data_pos,
        "total_tokens":   total_tokens,
        "total_flops":    total_flops,
        "elapsed_s":      elapsed_s,
        "wandb_run_id":   wandb_run_id,
        "torch_rng":      torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng":      np.random.get_state(),
        "python_rng":     random.getstate(),
    }
    tmp   = ckpt_dir / f"step_{step}.pt.tmp"
    final = ckpt_dir / f"step_{step}.pt"
    torch.save(state, tmp)
    tmp.rename(final)
    # Rolling 3-deep retention
    existing = sorted(ckpt_dir.glob("step_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    for old in existing[:-3]:
        old.unlink()
    ckpt_vol.commit()
    print(f"  checkpoint saved: {final.name}")


def maybe_resume(model, optimizer, ckpt_dir):
    """Load latest checkpoint if present. Returns (start_step, elapsed_offset,
    wandb_run_id, data_pos, total_tokens, total_flops)."""
    existing = sorted(ckpt_dir.glob("step_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if not existing:
        return 0, 0.0, None, 0, 0, 0

    ckpt = existing[-1]
    print(f"  resuming from {ckpt.name}")
    state = torch.load(ckpt, map_location="cuda")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return (
        state["step"],
        state["elapsed_s"],
        state["wandb_run_id"],
        state["data_pos"],
        state["total_tokens"],
        state["total_flops"],
    )

# ── Training ──────────────────────────────────────────────────────────────────

def train_run(cfg_dict: dict) -> float:
    import wandb
    from src.model import TSTModel, compute_loss

    cfg = TrainConfig(**cfg_dict)
    tst_steps = int(cfg.total_steps * cfg.tst_ratio)
    print(f"[{cfg.run_name}] tst_steps={tst_steps}  recovery_steps={cfg.total_steps - tst_steps}")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_tokens, val_tokens = _load_data()

    model = TSTModel(size="medium").to(device)
    model.transformer.gradient_checkpointing_enable()
    n_params       = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * n_params * cfg.batch_size * cfg.latent_seq
    print(f"[{cfg.run_name}] {n_params / 1e6:.1f}M params  "
          f"{flops_per_step:.3e} FLOPs/step")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    # Checkpoint dir
    ckpt_dir = CKPT_DIR_ROOT / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step, elapsed_offset, wandb_run_id, data_pos, total_tokens, total_flops = \
        maybe_resume(model, optimizer, ckpt_dir)

    # Init wandb (resume if we have a run ID)
    wandb.init(
        project="token-superposition",
        name=cfg.run_name,
        id=wandb_run_id,
        resume="allow",
        config=cfg_dict,
        group="tst-phase-ratio-sweep",
        reinit=True,
    )
    wandb_run_id = wandb.run.id

    wandb.define_metric("tokens_processed")
    wandb.define_metric("total_flops")
    wandb.define_metric("wall_clock_s")
    for metric in [
        "train/loss_vs_tokens", "train/loss_vs_flops", "train/loss_vs_time",
        "val/ce_vs_tokens",  "val/bpb_vs_tokens",
        "val/ce_vs_flops",   "val/bpb_vs_flops",
        "val/ce_vs_time",    "val/bpb_vs_time",
    ]:
        xm = {"tokens": "tokens_processed", "flops": "total_flops", "time": "wall_clock_s"}
        ax = xm[metric.split("_vs_")[1]]
        wandb.define_metric(metric, step_metric=ax)

    container_start = time.time()
    model.train()

    for global_step in range(start_step, cfg.total_steps):
        # Phase
        in_tst   = global_step < tst_steps
        bag      = cfg.bag_size if in_tst else 1
        data_seq = (cfg.latent_seq * cfg.bag_size) if in_tst else cfg.latent_seq
        stride   = cfg.batch_size * (data_seq + 1)

        # LR
        lr = get_lr(global_step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Batch
        if data_pos + stride > len(train_tokens):
            data_pos = 0
        chunk    = train_tokens[data_pos: data_pos + stride].astype(np.int64)
        data_pos += stride
        chunk    = chunk.reshape(cfg.batch_size, data_seq + 1)
        x = torch.tensor(chunk[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(chunk[:, 1:],  dtype=torch.long, device=device)

        # Forward / backward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, bag_size=bag)
            loss   = compute_loss(logits, y, bag_size=bag)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_tokens += cfg.batch_size * data_seq
        total_flops  += flops_per_step
        logged_step   = global_step + 1   # 1-indexed to match original runs
        elapsed = elapsed_offset + (time.time() - container_start)

        if logged_step % cfg.log_every == 0:
            wandb.log({
                "train/loss":           loss.item(),
                "train/loss_vs_tokens": loss.item(),
                "train/loss_vs_flops":  loss.item(),
                "train/loss_vs_time":   loss.item(),
                "tokens_processed":     total_tokens,
                "total_flops":          total_flops,
                "wall_clock_s":         elapsed,
                "lr":                   lr,
            }, step=logged_step)

        if logged_step % cfg.eval_every == 0:
            ce, bpb = evaluate(model, val_tokens, cfg.batch_size, device)
            elapsed = elapsed_offset + (time.time() - container_start)
            wandb.log({
                "val/ce":            ce,
                "val/bpb":           bpb,
                "val/ce_vs_tokens":  ce,  "val/bpb_vs_tokens": bpb,
                "val/ce_vs_flops":   ce,  "val/bpb_vs_flops":  bpb,
                "val/ce_vs_time":    ce,  "val/bpb_vs_time":   bpb,
                "tokens_processed":  total_tokens,
                "total_flops":       total_flops,
                "wall_clock_s":      elapsed,
            }, step=logged_step)
            phase_tag = "tst" if in_tst else "rec"
            print(f"  [{cfg.run_name}] step={logged_step} ({phase_tag}) "
                  f"loss={loss.item():.4f} val_ce={ce:.4f} bpb={bpb:.4f}")

        if logged_step % 1000 == 0:
            save_checkpoint(
                step=logged_step, model=model, optimizer=optimizer,
                data_pos=data_pos, total_tokens=total_tokens,
                total_flops=total_flops,
                elapsed_s=elapsed_offset + (time.time() - container_start),
                wandb_run_id=wandb_run_id, ckpt_dir=ckpt_dir,
            )

    # Final eval
    ce, bpb = evaluate(model, val_tokens, cfg.batch_size, device)
    elapsed = elapsed_offset + (time.time() - container_start)
    wandb.log({
        "val/ce_final":      ce,   "val/bpb_final":     bpb,
        "val/ce":            ce,   "val/bpb":           bpb,
        "val/ce_vs_tokens":  ce,   "val/bpb_vs_tokens": bpb,
        "val/ce_vs_flops":   ce,   "val/bpb_vs_flops":  bpb,
        "val/ce_vs_time":    ce,   "val/bpb_vs_time":   bpb,
        "tokens_processed":  total_tokens,
        "total_flops":       total_flops,
        "wall_clock_s":      elapsed,
    }, step=cfg.total_steps)
    wandb.finish()
    print(f"[{cfg.run_name}] Final val CE={ce:.4f} BPB={bpb:.4f}")
    return ce


# ── Modal GPU function ────────────────────────────────────────────────────────

_gpu_kwargs = dict(
    gpu="H100",
    timeout=18000,       # 5h — ~4.1h expected, 20% buffer
    scaledown_window=2,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/fineweb":                 fineweb_vol,
        "/checkpoints":             ckpt_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
)


@app.function(**_gpu_kwargs)
def run_variant(cfg_dict: dict) -> float:
    return train_run(cfg_dict)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print(f"Launching {len(VARIANTS)} variants in parallel …")
    handles = [run_variant.spawn(dataclasses.asdict(cfg)) for cfg in VARIANTS]
    results = {}
    for cfg, handle in zip(VARIANTS, handles):
        ce = handle.get()
        results[cfg.run_name] = ce
        print(f"  {cfg.run_name}  tst_ratio={cfg.tst_ratio}  "
              f"sup_steps={int(cfg.total_steps * cfg.tst_ratio)}  final_val_ce={ce:.4f}")
    print("\nSweep complete.")
    for name, ce in sorted(results.items()):
        print(f"  {name}: {ce:.4f}")
