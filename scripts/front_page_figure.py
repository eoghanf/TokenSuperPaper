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

ARROW_COLOR_TST   = (1.0, 0.647059, 0.0)   # yellow — matches isoloss plots
ARROW_COLOR_SCHED = (0.878, 0.235, 0.518)   # pink


def fetch_val_curve(run_id: str):
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    df = (run.history(samples=50000, keys=["_step", "val/bpb"], pandas=True)
             .dropna(subset=["val/bpb"])
             .sort_values("_step")
             .reset_index(drop=True))
    return df["_step"].to_numpy(), df["val/bpb"].to_numpy()


def _add_speedup_arrow(ax, base_steps, base_bpb, query_end_step,
                       query_end_bpb, color, label_above=True):
    """Horizontal <-> arrow from query endpoint to baseline iso-loss step."""
    # First baseline step where running-min bpb ≤ query_end_bpb
    base_rmin = np.minimum.accumulate(base_bpb)
    hits = np.where(base_rmin <= query_end_bpb)[0]
    if len(hits) == 0:
        return
    base_iso_step = base_steps[hits[0]]

    y     = query_end_bpb
    ylo, yhi = ax.get_ylim()
    dy    = (yhi - ylo) * 0.035
    y_label = y + dy if label_above else y - dy * 1.6
    mid_x = (query_end_step + base_iso_step) / 2
    ratio = base_iso_step / query_end_step

    ax.annotate("", xy=(query_end_step, y), xytext=(base_iso_step, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=2.0,
                                shrinkA=0, shrinkB=0))
    ax.plot([query_end_step, base_iso_step], [y, y],
            "o", color=color, markersize=5, zorder=5)
    ax.text(mid_x, y_label, f"{ratio:.2f}× fewer steps",
            ha="center", va="bottom" if label_above else "top",
            fontsize=8.5, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


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

    ax_line.set_xlabel("Training step", fontsize=10)
    ax_line.set_ylabel("Validation loss (BPB)", fontsize=10)
    ax_line.set_title("Validation loss during training", fontsize=11)
    ax_line.legend(fontsize=8, loc="upper right")
    ax_line.grid(True, alpha=0.3, linewidth=0.6)
    ax_line.set_xlim(left=0)

    # Speedup arrows (drawn after ylim is set by plotted data)
    base_steps, base_bpb, _ = curves["Baseline (40k steps)"]
    tst_steps,  tst_bpb,  _ = curves["Token Superposition (20k steps)"]
    sched_steps, sched_bpb, _ = curves["Schedule-matched (20k steps)"]

    _add_speedup_arrow(ax_line, base_steps, base_bpb,
                       tst_steps[-1], tst_bpb[-1],
                       ARROW_COLOR_TST, label_above=True)
    _add_speedup_arrow(ax_line, base_steps, base_bpb,
                       sched_steps[-1], sched_bpb[-1],
                       ARROW_COLOR_SCHED, label_above=False)

    # Decay marker (re-drawn after ylim settled)
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
