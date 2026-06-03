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

TARGET_BPB = 1.25   # threshold for the RH bar chart


def fetch_val_curve(run_id: str):
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    df = (run.history(samples=50000, keys=["_step", "val/bpb"], pandas=True)
             .dropna(subset=["val/bpb"])
             .sort_values("_step")
             .reset_index(drop=True))
    return df["_step"].to_numpy(), df["val/bpb"].to_numpy()


def steps_to_target(steps, bpb, target):
    """First step at which running-min BPB ≤ target, or None."""
    rmin = np.minimum.accumulate(bpb)
    hits = np.where(rmin <= target)[0]
    return int(steps[hits[0]]) if len(hits) else None


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

    # Decay marker
    ylo, yhi = ax_line.get_ylim()
    ax_line.axvline(18000, color="gray", linewidth=0.9, linestyle=":", alpha=0.8)
    ax_line.text(18300, ylo + (yhi - ylo) * 0.97,
                 "LR decay\n(step 18k)", fontsize=7.5, color="gray", va="top")

    # ── Right: steps to reach TARGET_BPB ─────────────────────────────────────
    bar_labels = [l.replace(" (", "\n(") for l in BAR_ORDER]
    bar_colors = [RUNS[l][1] for l in BAR_ORDER]
    bar_values = []
    for l in BAR_ORDER:
        s, b, _ = curves[l]
        n = steps_to_target(s, b, TARGET_BPB)
        bar_values.append(n)
        print(f"  {l}: steps to BPB≤{TARGET_BPB} = {n}")

    bars = ax_bar.bar(bar_labels, bar_values, color=bar_colors,
                      width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, bar_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 200, f"{val:,}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Horizontal blue reference line at the baseline bar height
    baseline_steps = bar_values[2]   # Baseline (40k steps) is last in BAR_ORDER
    ax_bar.axhline(baseline_steps, color="#2166ac", linewidth=1.5,
                   linestyle="--", alpha=0.7, zorder=3)

    # Vertical arrows from the reference line down to the 20k bars, with speedup labels
    for i in range(2):   # schedule-matched (0) and TST (1)
        bar   = bars[i]
        val   = bar_values[i]
        color = bar_colors[i]
        x     = bar.get_x() + bar.get_width() / 2
        speedup = baseline_steps / val

        ax_bar.annotate("",
            xy=(x, val),
            xytext=(x, baseline_steps),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                            mutation_scale=12))

        mid_y = (val + baseline_steps) / 2
        ax_bar.text(x + 0.18, mid_y, f"{speedup:.2f}×",
                    ha="left", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax_bar.set_ylim(0, max(bar_values) * 1.18)
    ax_bar.set_ylabel("Training steps", fontsize=10)
    ax_bar.set_title(f"Steps to {TARGET_BPB} validation loss", fontsize=11)
    ax_bar.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax_bar.tick_params(axis="x", labelsize=8)

    fig.tight_layout(pad=1.5)
    fig.savefig("assets/front_page_figure.png", dpi=150, bbox_inches="tight")
    print("→ assets/front_page_figure.png")


if __name__ == "__main__":
    main()
