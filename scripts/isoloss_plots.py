"""
Isoloss plots for all runs: Baseline40k, TokenSuperposition20k, and the
TST phase ratio sweep (tst-r006 … tst-r036).

For each run and each loss type (train / val), we compute the running
minimum of loss. Val loss is reported in bits-per-byte (BPB).

Wall-clock subplot has a green secondary x-axis showing estimated H100 cost ($).
FLOPs subplot has a green secondary x-axis showing wall-clock hours.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import numpy as np
import matplotlib.pyplot as plt
import wandb

# ── Config ────────────────────────────────────────────────────────────────────

ENTITY  = None          # uses ~/.netrc default entity
PROJECT = "token-superposition"

REFERENCE_RUNS = {
    "Baseline40k":           "u519arkx",
    "TokenSuperposition20k": "mk3hrzt5",
}

SWEEP_GROUP = "tst-phase-ratio-sweep"

COLORS = {
    "Baseline40k":           "#2166ac",
    "TokenSuperposition20k": "#d6604d",
}

# Sweep variants: plasma colormap, ordered light→dark by increasing ratio
_plasma = matplotlib.colormaps["plasma"]
SWEEP_COLORS = {
    "tst-r006": _plasma(0.15),
    "tst-r012": _plasma(0.32),
    "tst-r018": _plasma(0.50),
    "tst-r024": _plasma(0.68),
    "tst-r036": _plasma(0.84),
}

H100_COST_PER_SECOND = 3.95 / 3600   # $/s on H100

OUTPUT_DIR = "assets"   # relative to project root; run from project root

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_run_history(run_id: str, label: str) -> dict:
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}/{run_id}" if ENTITY else f"{PROJECT}/{run_id}"
    run  = api.run(path)
    x_cols = ["_step", "tokens_processed", "total_flops", "wall_clock_s"]

    train_hist = run.history(samples=50000, keys=x_cols + ["train/loss"], pandas=True)
    train_df   = (train_hist.dropna(subset=["train/loss"])
                            .sort_values("_step")
                            .reset_index(drop=True))

    val_hist = run.history(samples=50000, keys=x_cols + ["val/bpb"], pandas=True)
    val_df   = (val_hist.dropna(subset=["val/bpb"])
                        .sort_values("_step")
                        .reset_index(drop=True))

    return {"train": train_df, "val": val_df}


def fetch_all_data() -> dict:
    data = {}

    print("Fetching reference runs …")
    for name, run_id in REFERENCE_RUNS.items():
        print(f"  {name} ({run_id}) …")
        hist = _fetch_run_history(run_id, name)
        print(f"    train rows: {len(hist['train'])}, val rows: {len(hist['val'])}")
        data[name] = hist

    print("Fetching phase ratio sweep runs …")
    api  = wandb.Api()
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    runs = api.runs(path, filters={"group": SWEEP_GROUP})
    sweep = {}
    for run in runs:
        name = run.name
        print(f"  {name} ({run.id}) …")
        try:
            hist = _fetch_run_history(run.id, name)
            print(f"    train rows: {len(hist['train'])}, val rows: {len(hist['val'])}")
            sweep[name] = hist
        except Exception as exc:
            print(f"    skipped — {exc}")

    # Insert sweep variants in ratio order (alphabetical = ratio order here)
    for name in sorted(sweep):
        data[name] = sweep[name]

    print(f"Total runs loaded: {len(data)}")
    return data


# ── Isoloss computation ───────────────────────────────────────────────────────

def running_min(arr: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(arr)


def isoloss_curve(df, loss_col: str, x_col: str):
    x    = df[x_col].to_numpy(dtype=float)
    loss = df[loss_col].to_numpy(dtype=float)
    return x, running_min(loss)


def compute_flops_per_second(data: dict) -> float:
    """OLS-through-origin estimate of H100 FLOP/s from logged run data."""
    fv, sv = [], []
    for run_data in data.values():
        df = run_data.get("train")
        if df is None or df.empty:
            continue
        if "total_flops" not in df.columns or "wall_clock_s" not in df.columns:
            continue
        f = df["total_flops"].dropna().to_numpy(dtype=float)
        s = df["wall_clock_s"].dropna().to_numpy(dtype=float)
        mask = (f > 1e10) & (s > 10)
        fv.extend(f[mask].tolist())
        sv.extend(s[mask].tolist())
    if len(fv) < 2:
        return 9.45e13  # fallback: typical H100 GPT-2 medium throughput
    f_arr, s_arr = np.array(fv), np.array(sv)
    return float(np.dot(f_arr, s_arr) / np.dot(s_arr, s_arr))


# ── Secondary axis helpers ────────────────────────────────────────────────────

_SEC_COLOR = "#2ca25f"   # green for both secondary axes


def _add_cost_axis(ax):
    """Green secondary (top) x-axis: wall-clock seconds → estimated H100 cost ($)."""
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda s: s * H100_COST_PER_SECOND,
                   lambda c: c / H100_COST_PER_SECOND),
    )
    sec.set_xlabel("Cost ($)", fontsize=9, color=_SEC_COLOR)
    sec.tick_params(axis="x", colors=_SEC_COLOR, labelsize=8)


def _add_hours_axis(ax, flops_per_second: float, scale: float):
    """Green secondary (top) x-axis: scaled FLOPs → wall-clock hours."""
    # Primary axis values are total_flops / scale, so actual FLOPs = val * scale.
    def to_hours(f_scaled):
        return f_scaled * scale / flops_per_second / 3600.0
    def from_hours(h):
        return h * 3600.0 * flops_per_second / scale
    sec = ax.secondary_xaxis("top", functions=(to_hours, from_hours))
    sec.set_xlabel("Time (hours)", fontsize=9, color=_SEC_COLOR)
    sec.tick_params(axis="x", colors=_SEC_COLOR, labelsize=8)


# ── Plotting ──────────────────────────────────────────────────────────────────

# (x_col, subplot_title, x_axis_label, scale_divisor, secondary_type)
X_CONFIGS = [
    ("_step",            "Steps",                   "Steps",    None,  None),
    ("tokens_processed", "Tokens",                  "Tokens",   1e9,   None),
    ("wall_clock_s",     "Wall-clock time / cost",  "Time (s)", None,  "cost"),
    ("total_flops",      "FLOPs / time",             "FLOPs",   1e18,  "hours"),
]


def _run_color(name: str):
    if name in COLORS:
        return COLORS[name]
    return SWEEP_COLORS.get(name, "#888888")


def _add_iso_arrow(ax, data: dict, loss_type: str, loss_col: str,
                   x_col: str, scale):
    """Arrow between TokenSuperposition20k endpoint and Baseline40k iso-loss match."""
    if "TokenSuperposition20k" not in data or "Baseline40k" not in data:
        return
    tst_df  = data["TokenSuperposition20k"][loss_type]
    base_df = data["Baseline40k"][loss_type]
    if tst_df.empty or base_df.empty:
        return
    if x_col not in tst_df.columns or x_col not in base_df.columns:
        return

    tst_x,  tst_rmin  = isoloss_curve(tst_df,  loss_col, x_col)
    base_x, base_rmin = isoloss_curve(base_df, loss_col, x_col)

    tst_final_loss = tst_rmin[-1]
    tst_end_x      = tst_x[-1]  / scale if scale else tst_x[-1]

    hits = np.where(base_rmin <= tst_final_loss)[0]
    if len(hits) == 0:
        return
    base_iso_x = base_x[hits[0]] / scale if scale else base_x[hits[0]]

    y_arrow = tst_final_loss
    y_lo, y_hi = ax.get_ylim()
    y_offset   = (y_hi - y_lo) * 0.04

    ARROW_COLOR = (1.0, 0.647059, 0.0)   # matches original paper annotation colour

    ax.annotate(
        "",
        xy=(tst_end_x, y_arrow),
        xytext=(base_iso_x, y_arrow),
        arrowprops=dict(arrowstyle="<->", color=ARROW_COLOR, lw=2.0,
                        shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.plot([tst_end_x, base_iso_x], [y_arrow, y_arrow],
            "o", color=ARROW_COLOR, markersize=5, zorder=5)

    ratio = base_iso_x / tst_end_x if base_iso_x > tst_end_x else tst_end_x / base_iso_x
    word  = "fewer" if base_iso_x > tst_end_x else "more"
    mid_x = (tst_end_x + base_iso_x) / 2
    ax.text(mid_x, y_arrow + y_offset, f"{ratio:.2f}× {word}",
            ha="center", va="bottom", fontsize=9, color=ARROW_COLOR,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def make_figure(loss_type: str, loss_col: str, y_label: str, data: dict,
                show_title: bool = True,
                flops_per_second: float = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if show_title:
        fig.suptitle(f"Isoloss: {y_label} — all runs", fontsize=13)
    axes_flat = axes.flatten()

    for ax, (x_col, x_title, x_short, scale, secondary) in zip(axes_flat, X_CONFIGS):
        for run_name, run_data in data.items():
            df = run_data[loss_type]
            if x_col not in df.columns or df.empty:
                continue
            x, rmin = isoloss_curve(df, loss_col, x_col)
            if scale:
                x = x / scale
            ax.plot(x, rmin, label=run_name, color=_run_color(run_name), linewidth=1.8)

        x_label = x_short + (f" (×{scale:.0e})" if scale else "")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(x_title, fontsize=11)
        ax.grid(True, alpha=0.35, linewidth=0.6)
        ax.legend(fontsize=7)

        # Arrow drawn before secondary axis (uses ylim set by plotted data)
        _add_iso_arrow(ax, data, loss_type, loss_col, x_col, scale)

        if secondary == "cost":
            _add_cost_axis(ax)
        elif secondary == "hours" and flops_per_second is not None:
            _add_hours_axis(ax, flops_per_second, scale)

    fig.tight_layout()
    return fig


def main():
    data = fetch_all_data()

    fps = compute_flops_per_second(data)
    print(f"Estimated FLOP/s: {fps:.3e}")

    print("Building train loss figure …")
    fig = make_figure("train", "train/loss", "Train loss (nats)", data,
                      flops_per_second=fps)
    fig.savefig(f"{OUTPUT_DIR}/isoloss_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → isoloss_train.png")

    print("Building val BPB figure …")
    fig = make_figure("val", "val/bpb", "Val loss (BPB)", data,
                      flops_per_second=fps)
    fig.savefig(f"{OUTPUT_DIR}/isoloss_val.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → isoloss_val.png")

    print("Building val BPB figure (no title, for paper) …")
    fig = make_figure("val", "val/bpb", "Val loss (BPB)", data,
                      show_title=False, flops_per_second=fps)
    fig.savefig(f"{OUTPUT_DIR}/isoloss_val_notitle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → isoloss_val_notitle.png")

    print("Done.")


if __name__ == "__main__":
    main()
