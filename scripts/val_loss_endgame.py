"""
Val loss (BPB) from step 18000–20000 for all sweep variants + reference TST20k.
Saved to assets/val_loss_endgame.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import wandb

PROJECT = "token-superposition"

RUNS = {
    "Baseline40k":           "u519arkx",
    "TokenSuperposition20k": "mk3hrzt5",
    "tst-r006":              "hed9ojvl",
    "tst-r012":              "sdsyx0p2",
    "tst-r018":              "7nq2xtyb",
    "tst-r024":              "r5ki6lsm",
    "tst-r036":              "1spskdlo",
}

_plasma = matplotlib.colormaps["plasma"]
COLORS = {
    "Baseline40k":           "#2166ac",
    "TokenSuperposition20k": "#d6604d",
    "tst-r006":              _plasma(0.15),
    "tst-r012":              _plasma(0.32),
    "tst-r018":              _plasma(0.50),
    "tst-r024":              _plasma(0.68),
    "tst-r036":              _plasma(0.84),
}

STEP_LO, STEP_HI = 18000, 20000


def fetch_val(run_id: str) -> tuple:
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    df = run.history(samples=50000, keys=["_step", "val/bpb"], pandas=True)
    df = (df.dropna(subset=["val/bpb"])
            .sort_values("_step")
            .reset_index(drop=True))
    mask = (df["_step"] >= STEP_LO) & (df["_step"] <= STEP_HI)
    sub = df[mask]
    return sub["_step"].to_numpy(), sub["val/bpb"].to_numpy()


def main():
    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, run_id in RUNS.items():
        print(f"  {name} …")
        steps, bpb = fetch_val(run_id)
        if len(steps) == 0:
            print(f"    no data in range")
            continue
        lw = 2.2 if name in ("Baseline40k", "TokenSuperposition20k") else 1.6
        ls = "--" if name == "Baseline40k" else "-"
        ax.plot(steps, bpb, label=name, color=COLORS[name], linewidth=lw, linestyle=ls)
        # annotate final value
        ax.annotate(f"{bpb[-1]:.4f}",
                    xy=(steps[-1], bpb[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=COLORS[name])

    ax.set_xlim(STEP_LO, STEP_HI)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Val loss (BPB)", fontsize=11)
    ax.set_title("Validation loss — steps 18 000–20 000", fontsize=12)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("assets/val_loss_endgame.png", dpi=150, bbox_inches="tight")
    print("→ assets/val_loss_endgame.png")


if __name__ == "__main__":
    main()
