#!/usr/bin/env python3
"""Render the recorded FID curves for both training seeds."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, StrMethodFormatter

# Darker gold, green and magenta stay distinct at the report's printed size.
PAPER_COLORS = ["#0072B2", "#B57E00", "#007F5F", "#A83F85", "#D55E00"]
MARKERS = ["o", "s", "^", "D", "v"]


def styling():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42,
                         "axes.edgecolor": "#222222", "axes.labelcolor": "#111111",
                         "xtick.color": "#222222", "ytick.color": "#222222",
                         "axes.spines.top": True, "axes.spines.right": True,
                         "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def legends(first):
    powers = [Line2D([0], [0], color=color, marker=marker, markersize=4,
                     linewidth=1.6, label=f"$p={run['p']:g}$")
              for run, color, marker in zip(first, PAPER_COLORS, MARKERS)]
    seeds = [Line2D([0], [0], color="#334155", linewidth=1.6, marker="o",
                    markerfacecolor="#334155", markersize=4, label="Seed 42"),
             Line2D([0], [0], color="#334155", linewidth=1.6, marker="o",
                    linestyle=(0, (4, 2)), markerfacecolor="white", markersize=4, label="Seed 43")]
    return powers, seeds


def plot_axes(ax, first, second):
    for seed, runs in zip([42, 43], [first, second]):
        for run, color, marker in zip(runs, PAPER_COLORS, MARKERS):
            xs = [int(row["epoch"]) for row in run["curve"]]
            fids = [float(row["FID_ema"]) for row in run["curve"]]
            ax.plot(xs, fids, color=color, linewidth=1.4 if seed == 42 else 1.1,
                    linestyle="-" if seed == 42 else (0, (4, 2)),
                    marker=marker, markersize=4.8 if seed == 42 else 3.5,
                    markerfacecolor=color if seed == 42 else "white",
                    markeredgewidth=0.9, markeredgecolor=color)
    ax.set(xlim=(8, 83), xticks=list(range(10, 81, 10)),
           ylim=(55, 160), yticks=[60, 80, 100, 120, 140, 160])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Training epoch", labelpad=5)
    ax.set_ylabel("EMA FID-50K", labelpad=5)
    ax.grid(color="#e5e7eb", linewidth=0.6, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, length=3)
    _, seeds = legends(first)
    ax.legend(handles=seeds, loc="upper right", bbox_to_anchor=(0.99, 1.0),
                         ncol=1, frameon=False, handlelength=1.5, handletextpad=0.5,
                         borderaxespad=0.2, labelspacing=0.3, fontsize=10)


def draw(first, second):
    # Match one ISBI column at native 10pt, with the power key below the plot.
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.95)
    plot_axes(ax, first, second)
    powers, _ = legends(first)
    fig.legend(handles=powers, loc="lower center", bbox_to_anchor=(0.54, 0.005),
               ncol=5, frameon=False, handlelength=1.0, handletextpad=0.35,
               columnspacing=0.65, borderaxespad=0.25, fontsize=10)
    return fig
