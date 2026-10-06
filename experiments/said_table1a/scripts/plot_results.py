import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/said_table1a/tables/table1a_results.csv")

x = df["ratio_r_not_equal_t"]
paper = df["paper_fid"]
repro = df["reproduced_fid"]

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 6), height_ratios=[2, 3]
)

ax1.plot(x, paper, marker="o", label="Paper")
ax1.plot(x, repro, marker="o", label="Reproduced")

ax2.plot(x, paper, marker="o", label="Paper")
ax2.plot(x, repro, marker="o", label="Reproduced")

# Top axis shows the big 0% value
ax1.set_ylim(280, 340)

# Bottom axis zooms into 25/50/100
ax2.set_ylim(55, 72)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

d = 0.015
kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

for xi, yi in zip(x, paper):
    if yi > 280:
        ax1.text(xi, yi + 2, f"{yi:.2f}", ha="center", fontsize=9)
    else:
        ax2.text(xi, yi + 0.6, f"{yi:.2f}", ha="center", fontsize=9)

for xi, yi in zip(x, repro):
    if yi > 280:
        ax1.text(xi, yi - 8, f"{yi:.2f}", ha="center", fontsize=9)
    else:
        ax2.text(xi, yi - 1.6, f"{yi:.2f}", ha="center", fontsize=9)

ax1.legend()
ax1.set_title("Table 1a: Paper vs Reproduced FID")
ax2.set_xlabel("r != t (%)")
ax1.set_ylabel("FID")
ax2.set_ylabel("FID")

plt.tight_layout()

out = Path("results/said_table1a/figures/table1a_trend_broken.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(out)
