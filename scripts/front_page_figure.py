"""
Front-page figure: two panels side by side.
Left:  val BPB vs training steps for all three key runs.
Right: bar chart of final val BPB for each run.
Saved to assets/front_page_figure.png.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import wandb

PROJECT = "token-superposition"

RUNS = {
    "Baseline (40k steps)":           ("u519arkx", "#2166ac"),
    "Token Superposition (20k steps)": ("mk3hrzt5", "#d6604d"),
    "Schedule-matched (20k steps)":    ("4a2n0fas", "#5aae61"),
}

BAR_ORDER = [
    "Schedule-matched (20k steps)",
    "Token Superposition (20k steps)",
    "Baseline (40k steps)",
]


def fetch_val_curve(run_id: str):
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    df = (run.history(samples=50000, keys=["_step", "val/bpb"], pandas=True)
             .dropna(subset=["val/bpb"])
             .sort_values("_step")
             .reset_index(drop=True))
    return df["_step"].to_numpy(), df["val/bpb"].to_numpy()


def main():
    print("Fetching val curves …")
    curves = {}
    for label, (run_id, color) in RUNS.items():
        print(f"  {label} …")
        steps, bpb = fetch_val_curve(run_id)
        curves[label] = (steps, bpb, color)
        print(f"    {len(steps)} rows, final BPB={bpb[-1]:.4f}")

    fig, (ax_line, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── Left: val BPB vs steps ─────────────────────────────────────────────────
    for label, (steps, bpb, color) in curves.items():
        lw = 2.0 if "Baseline" in label and "Schedule" not in label else 1.8
        ls = "--" if label == "Baseline (40k steps)" else "-"
        ax_line.plot(steps, bpb, label=label, color=color, linewidth=lw, linestyle=ls)

    # Mark the WSD decay horizon for the 20k runs
    ax_line.axvline(18000, color="gray", linewidth=0.9, linestyle=":", alpha=0.8)
    ax_line.text(18200, ax_line.get_ylim()[1] if ax_line.get_ylim()[1] < 2 else 1.55,
                 "LR decay\n(step 18k)", fontsize=7.5, color="gray", va="top")

    ax_line.set_xlabel("Training step", fontsize=10)
    ax_line.set_ylabel("Validation loss (BPB)", fontsize=10)
    ax_line.set_title("Validation loss during training", fontsize=11)
    ax_line.legend(fontsize=8, loc="upper right")
    ax_line.grid(True, alpha=0.3, linewidth=0.6)
    ax_line.set_xlim(left=0)

    # Re-draw the decay line after ylim is set
    ylo, yhi = ax_line.get_ylim()
    ax_line.axvline(18000, color="gray", linewidth=0.9, linestyle=":", alpha=0.8)
    ax_line.text(18300, ylo + (yhi - ylo) * 0.97,
                 "LR decay\n(step 18k)", fontsize=7.5, color="gray", va="top")

    # ── Right: bar chart of final BPB ─────────────────────────────────────────
    bar_labels  = [l.replace(" (", "\n(") for l in BAR_ORDER]
    bar_colors  = [RUNS[l][1] for l in BAR_ORDER]
    bar_values  = [curves[l][1][-1] for l in BAR_ORDER]

    bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors,
                      width=0.5, edgecolor="white", linewidth=0.8)

    # Annotate bar tops
    for bar, val in zip(bars, bar_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.001, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Bracket showing the small TST-vs-schedule gap
    sched_val = bar_values[0]
    tst_val   = bar_values[1]
    x0, x1    = bars[0].get_x() + bars[0].get_width() / 2, \
                 bars[1].get_x() + bars[1].get_width() / 2
    mid_x     = (x0 + x1) / 2
    bracket_y = max(sched_val, tst_val) + 0.007
    ax_bar.annotate("", xy=(x0, bracket_y), xytext=(x1, bracket_y),
                    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
    ax_bar.text(mid_x, bracket_y + 0.001,
                f"Δ = {abs(sched_val - tst_val):.3f} BPB\n(15% of TST gain)",
                ha="center", va="bottom", fontsize=7.5, color="#555555")

    ymin = min(bar_values) - 0.025
    ymax = max(bar_values) + 0.04
    ax_bar.set_ylim(ymin, ymax)
    ax_bar.set_ylabel("Final validation loss (BPB)", fontsize=10)
    ax_bar.set_title("Final validation loss", fontsize=11)
    ax_bar.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax_bar.tick_params(axis="x", labelsize=8)

    fig.tight_layout(pad=1.5)
    fig.savefig("assets/front_page_figure.png", dpi=150, bbox_inches="tight")
    print("→ assets/front_page_figure.png")


if __name__ == "__main__":
    main()
