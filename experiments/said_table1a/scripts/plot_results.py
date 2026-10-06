import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    "results/said_table1a/tables/table1a_results.csv",
    delimiter=",", skiprows=1, usecols=(0, 1, 2)
)

x, paper, reproduced = data.T
diff = reproduced - paper

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, diff, width=12)

ax.axhline(0, linewidth=1)
ax.set_xticks(x)
ax.set_xlabel("r != t (%)")
ax.set_ylabel("FID difference")
ax.set_title("Table 1a reproduction error\nReproduced FID - paper FID")
ax.grid(axis="y", alpha=0.2)

for bar, value in zip(bars, diff):
    y = value + 0.05 if value >= 0 else value - 0.05
    va = "bottom" if value >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width()/2, y, f"{value:+.2f}",
            ha="center", va=va)

plt.tight_layout()
plt.savefig("results/said_table1a/figures/table1a_fid_difference.png", dpi=250)
plt.savefig("results/said_table1a/figures/table1a_fid_difference.pdf")
