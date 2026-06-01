"""
Isoloss plots for Baseline40k vs TokenSuperposition20k.

For each run and each loss type (train / val), we compute the running
minimum of loss and then invert: for each loss level, report the earliest
step / tokens / wall-clock time / FLOPs at which that level was first reached.

Produces two figures (train and val), each with 4 subplots in a 2×2 grid.
Val loss is reported in bits-per-byte (BPB).
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wandb

# ── Config ────────────────────────────────────────────────────────────────────

ENTITY  = None          # uses ~/.netrc default entity
PROJECT = "token-superposition"

RUNS = {
    "Baseline40k":          "u519arkx",
    "TokenSuperposition20k": "mk3hrzt5",
}

COLORS = {
    "Baseline40k":          "#2166ac",
    "TokenSuperposition20k": "#d6604d",
}

OUTPUT_DIR = "assets"   # relative to project root; run from project root

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_history(run_id: str) -> dict:
    """Return dict with DataFrames for train and val history."""
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")

    x_cols = ["_step", "tokens_processed", "total_flops", "wall_clock_s"]

    # Fetch train and val separately so NaN-dropping doesn't discard rows.
    # samples=50000 >> actual rows so we get the full history.
    train_hist = run.history(samples=50000, keys=x_cols + ["train/loss"], pandas=True)
    train_df   = (
        train_hist.dropna(subset=["train/loss"])
                  .sort_values("_step")
                  .reset_index(drop=True)
    )

    val_hist = run.history(samples=50000, keys=x_cols + ["val/bpb"], pandas=True)
    val_df   = (
        val_hist.dropna(subset=["val/bpb"])
                .sort_values("_step")
                .reset_index(drop=True)
    )

    return {"train": train_df, "val": val_df, "name": run.name}


# ── Isoloss computation ───────────────────────────────────────────────────────

def running_min(loss_arr: np.ndarray) -> np.ndarray:
    """Cumulative minimum of a 1-D array."""
    return np.minimum.accumulate(loss_arr)


def isoloss_curve(df, loss_col: str, x_col: str):
    """
    Given a DataFrame sorted by step, compute:
      - x_vals: the x values at each logged step
      - loss_running_min: the running minimum loss up to each step

    This gives the isoloss curve: (x_val, best_loss_so_far).
    Plotting x on the horizontal and loss on the vertical axis shows
    the "envelope" of best loss achieved by each resource level.
    """
    x    = df[x_col].to_numpy(dtype=float)
    loss = df[loss_col].to_numpy(dtype=float)
    rmin = running_min(loss)
    return x, rmin


# ── Plotting ──────────────────────────────────────────────────────────────────

X_CONFIGS = [
    ("_step",            "Steps",          "Steps",   None),
    ("tokens_processed", "Tokens",         "Tokens",  1e9),
    ("wall_clock_s",     "Wall-clock time","Time (s)", None),
    ("total_flops",      "FLOPs",          "FLOPs",   1e18),
]


def _add_iso_arrow(ax, data: dict, loss_type: str, loss_col: str,
                   x_col: str, scale):
    """Draw a double-headed arrow at TST's final loss level between the
    TST endpoint and the point where the baseline first matches that loss."""
    tst_df  = data["TokenSuperposition20k"][loss_type]
    base_df = data["Baseline40k"][loss_type]
    if tst_df.empty or base_df.empty:
        return
    if x_col not in tst_df.columns or x_col not in base_df.columns:
        return

    tst_x,  tst_rmin  = isoloss_curve(tst_df,  loss_col, x_col)
    base_x, base_rmin = isoloss_curve(base_df, loss_col, x_col)

    tst_final_loss = tst_rmin[-1]
    tst_end_x      = tst_x[-1] / scale if scale else tst_x[-1]

    # First baseline point that reaches TST's final loss
    hits = np.where(base_rmin <= tst_final_loss)[0]
    if len(hits) == 0:
        return
    base_iso_x = base_x[hits[0]] / scale if scale else base_x[hits[0]]

    # Arrow y sits at TST's final loss; label goes a little above
    y_arrow = tst_final_loss
    y_lo, y_hi = ax.get_ylim()
    y_offset = (y_hi - y_lo) * 0.04

    ARROW_COLOR = (1.0, 0.647059, 0.0)   # matches original paper annotation colour

    ax.annotate(
        "",
        xy=(tst_end_x, y_arrow),
        xytext=(base_iso_x, y_arrow),
        arrowprops=dict(
            arrowstyle="<->",
            color=ARROW_COLOR,
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
        ),
        annotation_clip=False,
    )

    # Dots at both endpoints
    ax.plot([tst_end_x, base_iso_x], [y_arrow, y_arrow],
            "o", color=ARROW_COLOR, markersize=5, zorder=5)

    # Label: ratio and direction word
    ratio = base_iso_x / tst_end_x if base_iso_x > tst_end_x else tst_end_x / base_iso_x
    word  = "fewer" if base_iso_x > tst_end_x else "more"
    mid_x = (tst_end_x + base_iso_x) / 2
    ax.text(mid_x, y_arrow + y_offset, f"{ratio:.2f}× {word}",
            ha="center", va="bottom", fontsize=9, color=ARROW_COLOR,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def make_figure(loss_type: str, loss_col: str, y_label: str, data: dict,
                show_title: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    if show_title:
        fig.suptitle(
            f"Isoloss: {y_label} — Baseline40k vs TokenSuperposition20k",
            fontsize=13,
        )
    axes_flat = axes.flatten()

    for ax, (x_col, x_title, x_short, scale) in zip(axes_flat, X_CONFIGS):
        for run_name, run_data in data.items():
            df = run_data[loss_type]
            if x_col not in df.columns or df.empty:
                continue
            x, rmin = isoloss_curve(df, loss_col, x_col)
            if scale:
                x = x / scale
            ax.plot(x, rmin, label=run_name, color=COLORS[run_name], linewidth=1.8)

        x_label = x_short + (f" (×{scale:.0e})" if scale else "")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(x_title, fontsize=11)
        ax.grid(True, alpha=0.35, linewidth=0.6)
        ax.legend(fontsize=8)

        # Add arrow after axes limits are set by the plotted data
        _add_iso_arrow(ax, data, loss_type, loss_col, x_col, scale)

    fig.tight_layout()
    return fig


def main():
    print("Fetching wandb history …")
    data = {}
    for run_name, run_id in RUNS.items():
        print(f"  {run_name} ({run_id}) …")
        data[run_name] = fetch_history(run_id)
        print(f"    train rows: {len(data[run_name]['train'])}, "
              f"val rows: {len(data[run_name]['val'])}")

    print("Building train loss isoloss figure …")
    fig_train = make_figure("train", "train/loss", "Train loss (nats)", data)
    fig_train.savefig(f"{OUTPUT_DIR}/isoloss_train.png", dpi=150, bbox_inches="tight")
    print("  → isoloss_train.png")

    print("Building val BPB isoloss figure …")
    fig_val = make_figure("val", "val/bpb", "Val loss (BPB)", data)
    fig_val.savefig(f"{OUTPUT_DIR}/isoloss_val.png", dpi=150, bbox_inches="tight")
    print("  → isoloss_val.png")

    print("Building val BPB isoloss figure (no title, for paper) …")
    fig_val_notitle = make_figure("val", "val/bpb", "Val loss (BPB)", data,
                                  show_title=False)
    fig_val_notitle.savefig(f"{OUTPUT_DIR}/isoloss_val_notitle.png", dpi=150, bbox_inches="tight")
    print("  → isoloss_val_notitle.png")

    print("Done.")


if __name__ == "__main__":
    main()
