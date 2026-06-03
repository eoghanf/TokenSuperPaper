"""
Schedule-matched baseline — Experiment 7.

Standard AR training for 20,000 steps on GPT-2 medium, with WSD LR decay
starting at step 18,000 (90% of total, identical timing to TST20k).

Total FLOPs ≈ TST20k total FLOPs (verified: Baseline40k reaches TST20k's
FLOPs at its step 20,000).  No bag training.  Tests whether TST's advantage
over an equal-FLOPs baseline is real, or is purely an artefact of the WSD
decay horizon arriving earlier in FLOPs-space.

Wandb run name: baseline-20k-matched
Wandb group:    schedule-ablation
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

hf_cache    = modal.Volume.from_name("hf-cache",        create_if_missing=True)
fineweb_vol = modal.Volume.from_name("fineweb-volume",   create_if_missing=False)
ckpt_vol    = modal.Volume.from_name("tst-checkpoints",  create_if_missing=True)

FINEWEB_DIR   = Path("/fineweb")
CKPT_DIR_ROOT = Path("/checkpoints")
HEADER_BYTES  = 1024
BYTES_PER_TOKEN = 4.0

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

app = modal.App("schedule-matched-baseline", image=image)

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    run_name:     str   = "baseline-20k-matched"
    total_steps:  int   = 20000
    batch_size:   int   = 32
    seq_len:      int   = 1024
    lr:           float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int   = 200
    grad_clip:    float = 1.0
    log_every:    int   = 25
    eval_every:   int   = 250

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """WSD: linear warmup → stable → linear decay to 10% LR.
    Decay starts at step 18,000 (= 0.9 × 20,000), matching TST20k exactly."""
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
    from src.model import compute_loss
    n = cfg_global.seq_len + 1
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
    existing = sorted(ckpt_dir.glob("step_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    for old in existing[:-3]:
        old.unlink()
    ckpt_vol.commit()
    print(f"  checkpoint saved: {final.name}")


def maybe_resume(model, optimizer, ckpt_dir):
    existing = sorted(ckpt_dir.glob("step_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
    if not existing:
        return 0, 0.0, None, 0, 0, 0
    ckpt  = existing[-1]
    print(f"  resuming from {ckpt.name}")
    state = torch.load(ckpt, map_location="cuda")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return (
        state["step"], state["elapsed_s"], state["wandb_run_id"],
        state["data_pos"], state["total_tokens"], state["total_flops"],
    )

# ── Training ──────────────────────────────────────────────────────────────────

cfg_global = TrainConfig()   # used by evaluate() for seq_len

def train_run() -> float:
    import wandb
    from src.model import TSTModel, compute_loss

    cfg = TrainConfig()
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_tokens, val_tokens = _load_data()

    model = TSTModel(size="medium").to(device)
    model.transformer.gradient_checkpointing_enable()
    n_params       = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * n_params * cfg.batch_size * cfg.seq_len
    print(f"{n_params / 1e6:.1f}M params  {flops_per_step:.3e} FLOPs/step")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    ckpt_dir = CKPT_DIR_ROOT / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step, elapsed_offset, wandb_run_id, data_pos, total_tokens, total_flops = \
        maybe_resume(model, optimizer, ckpt_dir)

    wandb.init(
        project="token-superposition",
        name=cfg.run_name,
        id=wandb_run_id,
        resume="allow",
        config=dataclasses.asdict(cfg),
        group="schedule-ablation",
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
        ax = {"tokens": "tokens_processed", "flops": "total_flops",
              "time": "wall_clock_s"}[metric.split("_vs_")[1]]
        wandb.define_metric(metric, step_metric=ax)

    container_start = time.time()
    model.train()

    stride = cfg.batch_size * (cfg.seq_len + 1)

    for global_step in range(start_step, cfg.total_steps):
        lr = get_lr(global_step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if data_pos + stride > len(train_tokens):
            data_pos = 0
        chunk    = train_tokens[data_pos: data_pos + stride].astype(np.int64)
        data_pos += stride
        chunk    = chunk.reshape(cfg.batch_size, cfg.seq_len + 1)
        x = torch.tensor(chunk[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(chunk[:, 1:],  dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, bag_size=1)
            loss   = compute_loss(logits, y, bag_size=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_tokens += cfg.batch_size * cfg.seq_len
        total_flops  += flops_per_step
        logged_step   = global_step + 1
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
                "val/ce":            ce,   "val/bpb":           bpb,
                "val/ce_vs_tokens":  ce,   "val/bpb_vs_tokens": bpb,
                "val/ce_vs_flops":   ce,   "val/bpb_vs_flops":  bpb,
                "val/ce_vs_time":    ce,   "val/bpb_vs_time":   bpb,
                "tokens_processed":  total_tokens,
                "total_flops":       total_flops,
                "wall_clock_s":      elapsed,
            }, step=logged_step)
            print(f"  step={logged_step}  loss={loss.item():.4f} "
                  f"val_ce={ce:.4f}  bpb={bpb:.4f}")

        if logged_step % 1000 == 0:
            save_checkpoint(
                step=logged_step, model=model, optimizer=optimizer,
                data_pos=data_pos, total_tokens=total_tokens,
                total_flops=total_flops,
                elapsed_s=elapsed_offset + (time.time() - container_start),
                wandb_run_id=wandb_run_id, ckpt_dir=ckpt_dir,
            )

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
    print(f"Final val CE={ce:.4f}  BPB={bpb:.4f}")
    return ce

# ── Modal function ────────────────────────────────────────────────────────────

@app.function(
    gpu="H100",
    timeout=18000,
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
def run_baseline() -> float:
    return train_run()

# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Launching schedule-matched baseline (20k steps, AR only, decay@18k) …")
    ce = run_baseline.remote()
    print(f"Done. Final val CE={ce:.4f}")
