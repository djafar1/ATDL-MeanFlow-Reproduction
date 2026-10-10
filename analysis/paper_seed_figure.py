#!/usr/bin/env python3
"""Create paper-style combined seed figures from verified data or a labeled demo."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, StrMethodFormatter

from preview_seed_visualisations import REPORT, MARKERS, curves, save

# Darker gold, green and magenta stay distinct at the report's printed size.
PAPER_COLORS = ["#0072B2", "#B57E00", "#007F5F", "#A83F85", "#D55E00"]


def styling():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
                         "axes.edgecolor": "#222222", "axes.labelcolor": "#111111",
                         "xtick.color": "#222222", "ytick.color": "#222222",
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def legends(first, mode):
    powers = [Line2D([0], [0], color=color, marker=marker, markersize=4,
                     linewidth=1.6, label=f"$p={run['p']:g}$")
              for run, color, marker in zip(first, PAPER_COLORS, MARKERS)]
    seeds = [Line2D([0], [0], color="#334155", linewidth=1.6, marker="o",
                    markerfacecolor="#334155", markersize=4, label="Seed 42"),
             Line2D([0], [0], color="#334155", linewidth=1.6, marker="o",
                    linestyle=(0, (4, 2)), markerfacecolor="white", markersize=4, label="Seed 43 (mock)" if mode == "synthetic" else "Seed 43")]
    return powers, seeds


def plot_axes(ax, first, second, mode, yscale):
    for seed, runs in zip([42, 43], [first, second]):
        for run, color, marker in zip(runs, PAPER_COLORS, MARKERS):
            xs = [int(row["epoch"]) for row in run["curve"]]
            fids = [float(row["FID_ema"]) for row in run["curve"]]
            ax.plot(xs, fids, color=color, linewidth=1.4 if seed == 42 else 1.1,
                    linestyle="-" if seed == 42 else (0, (4, 2)),
                    marker=marker, markersize=4.8 if seed == 42 else 3.5,
                    markerfacecolor=color if seed == 42 else "white",
                    markeredgewidth=0.9, markeredgecolor=color)
    ax.set_yscale(yscale)
    ax.set(xlim=(8, 83), xticks=list(range(10, 81, 10)),
           ylim=(55, 160), yticks=[60, 80, 100, 120, 140, 160])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Training epoch", labelpad=5)
    ax.set_ylabel("EMA FID-50K (log scale)" if yscale == "log" else "EMA FID-50K", labelpad=5)
    ax.grid(color="#e5e7eb", linewidth=0.6, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, length=3)
    _, seeds = legends(first, mode)
    seed_key = ax.legend(handles=seeds, loc="upper right", bbox_to_anchor=(0.99, 1.0),
                         ncol=1, frameon=False, handlelength=1.5, handletextpad=0.5,
                         borderaxespad=0.2, labelspacing=0.3, fontsize=10)
    if mode == "synthetic":
        seed_key.get_texts()[-1].set_color("#b91c1c")


def draw(first, second, mode, yscale="linear"):
    # Match one ISBI column at native 10pt, with the power key below the plot.
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.95)
    plot_axes(ax, first, second, mode, yscale)
    powers, _ = legends(first, mode)
    fig.legend(handles=powers, loc="lower center", bbox_to_anchor=(0.54, 0.005),
               ncol=5, frameon=False, handlelength=1.0, handletextpad=0.35,
               columnspacing=0.65, borderaxespad=0.25, fontsize=10)
    return fig


def draw_scale_comparison(first, second, mode):
    # A separate review asset, not a two-panel figure in the report.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.75, wspace=0.30)
    for ax, scale in zip(axes, ["linear", "log"]):
        plot_axes(ax, first, second, mode, scale)
        ax.set_title("Linear y-axis" if scale == "linear" else "Logarithmic y-axis",
                     fontsize=10, loc="left", pad=5)
    powers, _ = legends(first, mode)
    fig.legend(handles=powers, loc="upper center", bbox_to_anchor=(0.535, 0.99),
               ncol=5, frameon=False, handlelength=1.2, columnspacing=1.2, fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "recorded"], default="recorded")
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--compare-scales", action="store_true",
                        help="Also save matched linear/log versions and a separate comparison figure.")
    parser.add_argument("--summary", type=Path, default=REPORT / "analysis/generated/analysis_summary.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    first, second = curves(summary, args.mode)
    folder = REPORT / "analysis/visualisation-previews" / (
        "SYNTHETIC_SEED43" if args.mode == "synthetic" else "RECORDED_SEEDS") / "paper_figures"
    folder.mkdir(parents=True, exist_ok=True)
    styling()
    save(draw(first, second, args.mode, args.yscale), folder / "combined_seeds_2d_paper")
    if args.compare_scales:
        for scale in ["linear", "log"]:
            save(draw(first, second, args.mode, scale), folder / f"combined_seeds_2d_{scale}")
        save(draw_scale_comparison(first, second, args.mode), folder / "linear_vs_log_comparison")
    (folder / "figure_provenance.json").write_text(json.dumps({
        "mode": args.mode, "synthetic_seed43": args.mode == "synthetic",
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "seed42": [{"p": x["p"], "curve": x["curve"]} for x in first],
        "seed43": [{"p": x["p"], "curve": x["curve"]} for x in second],
        "encoding": "color and marker shape: norm_p; solid/filled: seed 42; dashed/open: seed 43",
        "layout": "One full-range plot in a single ISBI column; p legend in one row below the plot and compact seed key inside the upper-right corner.",
        "y_axis_scale": args.yscale,
        "scale_comparison_generated": args.compare_scales,
        "scope": "No synthetic mean, SD or variability estimate is calculated."
    }, indent=2) + "\n")
    print(f"Combined seed paper figures written to {folder}")


if __name__ == "__main__":
    main()
