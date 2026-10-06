from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("results/said_table1a/tables/table1a_results.csv")
out_dir = Path("results/said_table1a/figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

x = df["ratio_r_not_equal_t"]
y = df["reproduced_fid"] - df["paper_fid"]

colors = ["green", "orange", "blue", "purple"]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    x,
    y,
    color=colors,
    width=12,
    edgecolor="black",
    linewidth=1.2
)

plt.axhline(0, linewidth=1.2)
plt.grid(axis="y", linestyle="--", alpha=0.25)

plt.title("Table 1a reproduction error\nReproduced FID - paper FID", fontsize=16)
plt.xlabel("r != t (%)", fontsize=13)
plt.ylabel("FID difference", fontsize=13)
plt.xticks(x, [str(v) for v in x], fontsize=12)
plt.yticks(fontsize=12)

ymin = min(y)
ymax = max(y)
pad = 0.12
plt.ylim(ymin - pad, ymax + pad)

for bar, yi in zip(bars, y):
    x_text = bar.get_x() + bar.get_width() / 2
    if yi >= 0:
        plt.text(
            x_text, yi + 0.025, f"{yi:+.2f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )
    else:
        plt.text(
            x_text, yi - 0.035, f"{yi:+.2f}",
            ha="center", va="top", fontsize=12, fontweight="bold"
        )

plt.tight_layout()
plt.savefig(out_dir / "table1a_fid_difference.png", dpi=220, bbox_inches="tight")
plt.close()

print("Saved:", out_dir / "table1a_fid_difference.png")
