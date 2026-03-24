"""
Plot val_bpb learning curves for all benchmarked submissions on 1x RTX 4090.
Reads curve.csv from each models/<submission>/ directory.

Usage:
    python dsa/plot_curves.py
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODELS_DIR = Path("models")

# Short display names (order: competition BPB, best last so it draws on top)
DISPLAY_NAMES = {
    "NaiveBaseline": "Baseline",
    "LongContextSeq2048": "Seq2048",
    "TrainingOptSeq4096": "Seq4096",
    "SlidingWindowEval": "SW Eval",
    "SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit": "10L Overtone",
    "MixedQuant_Int6Int8_SlidingWindow": "MixedQuant",
    "Seq2048_FP16Emb_TunedLR": "Seq2048 Int6",
    "smeargate_orthoinit_muonwd": "SmearGate",
    "MLP3x_QAT_Int6_SlidingWindow": "MLP3x QAT",
    "Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA": "SmearGate+SWA",
    "10L_Int5MLP_MuonWD04_SWA50": "Int5 MLP SWA",
}

# Plot order: weakest first so stronger lines draw on top
PLOT_ORDER = [
    "NaiveBaseline",
    "MLP3x_QAT_Int6_SlidingWindow",
    "SlidingWindowEval",
    "SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit",
    "MixedQuant_Int6Int8_SlidingWindow",
    "smeargate_orthoinit_muonwd",
    "LongContextSeq2048",
    "Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA",
    "TrainingOptSeq4096",
    "Seq2048_FP16Emb_TunedLR",
    "10L_Int5MLP_MuonWD04_SWA50",
]


def load_curve(submission: str) -> tuple[list[float], list[float]]:
    """Return (time_minutes, val_bpb) lists from curve.csv."""
    path = MODELS_DIR / submission / "curve.csv"
    times, bpbs = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_min = int(row["train_time_ms"]) / 60_000.0
            bpb = float(row["val_bpb"])
            times.append(t_min)
            bpbs.append(bpb)
    return times, bpbs


def main():
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 7))

    # Color palette: tab10 + a few extra
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(PLOT_ORDER))]

    for i, sub in enumerate(PLOT_ORDER):
        if not (MODELS_DIR / sub / "curve.csv").exists():
            continue
        times, bpbs = load_curve(sub)
        label = DISPLAY_NAMES.get(sub, sub)
        color = colors[i]
        lw = 2.5 if i >= len(PLOT_ORDER) - 3 else 1.5  # bold top 3
        alpha = 1.0 if i >= len(PLOT_ORDER) - 3 else 0.7

        ax_full.plot(times, bpbs, label=label, color=color, lw=lw, alpha=alpha)
        ax_zoom.plot(times, bpbs, label=label, color=color, lw=lw, alpha=alpha)

    # --- Full view ---
    ax_full.set_xlabel("Train Time (minutes)", fontsize=12)
    ax_full.set_ylabel("val_bpb (bits per byte)", fontsize=12)
    ax_full.set_title("Full Training Curve (0-60 min)", fontsize=13)
    ax_full.set_xlim(0, 62)
    ax_full.set_ylim(1.20, 2.50)
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(fontsize=8, loc="upper right")

    # Vertical lines for competition-equivalent checkpoints
    for t, lbl in [
        (10, "10m"),
        (20, "20m\n~comp"),
        (30, "30m"),
        (45, "45m"),
        (60, "60m"),
    ]:
        ax_full.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.5)
        ax_full.text(t + 0.3, 2.45, lbl, fontsize=7, color="gray", va="top")

    # --- Zoomed view (last 30 min, where differentiation happens) ---
    ax_zoom.set_xlabel("Train Time (minutes)", fontsize=12)
    ax_zoom.set_ylabel("val_bpb (bits per byte)", fontsize=12)
    ax_zoom.set_title("Zoomed: 30-60 min (differentiation region)", fontsize=13)
    ax_zoom.set_xlim(28, 62)
    ax_zoom.set_ylim(1.22, 1.32)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(fontsize=8, loc="upper right")
    ax_zoom.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax_zoom.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
    ax_zoom.grid(True, which="minor", alpha=0.15)

    for t in [30, 45, 60]:
        ax_zoom.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.5)

    fig.suptitle(
        "Benchmark: Competition Submissions on 1x RTX 4090 (60 min)",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path("dsa/benchmark_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Also make a final BPB bar chart ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    subs_sorted = []
    for sub in PLOT_ORDER:
        if not (MODELS_DIR / sub / "curve.csv").exists():
            continue
        _, bpbs = load_curve(sub)
        final_bpb = bpbs[-1]
        subs_sorted.append((sub, final_bpb))

    subs_sorted.sort(
        key=lambda x: x[1], reverse=True
    )  # worst first (top of horizontal bar)

    names = [DISPLAY_NAMES.get(s, s) for s, _ in subs_sorted]
    vals = [v for _, v in subs_sorted]
    bar_colors = [
        "#2196F3" if v > 1.25 else "#4CAF50" if v > 1.235 else "#FF9800" for v in vals
    ]

    bars = ax2.barh(names, vals, color=bar_colors, edgecolor="white", height=0.6)
    ax2.set_xlabel("val_bpb at 60 min (lower is better)", fontsize=12)
    ax2.set_title("Final val_bpb at 60 min — 1x RTX 4090 Benchmark", fontsize=13)
    ax2.set_xlim(min(vals) - 0.005, max(vals) + 0.005)
    ax2.grid(True, axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, vals):
        ax2.text(
            val + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    out2 = Path("dsa/benchmark_final_bpb.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()


if __name__ == "__main__":
    main()
