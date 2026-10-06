import numpy as np
import matplotlib.pyplot as plt

x, paper, ours = np.loadtxt(
    "results/said_table1a/tables/table1a_results.csv",
    delimiter=",", skiprows=1, usecols=(0, 1, 2), unpack=True
)

plt.plot(x, paper, "o-", label="Paper")
plt.plot(x, ours, "o-", label="Reproduced")
plt.xlabel("r != t (%)")
plt.ylabel("FID")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.savefig("results/said_table1a/figures/table1a_trend.png", dpi=200)
