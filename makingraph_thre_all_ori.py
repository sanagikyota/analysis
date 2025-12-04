import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")   

def load_values(paths):
    arr = []
    for p in paths:
        if not os.path.exists(p):
            print("Missing:", p)
            continue
        df = pd.read_csv(p, header=None, skiprows=1, names=["median"])
        arr.append(pd.to_numeric(df["median"], errors="coerce").dropna().values)
    return np.concatenate(arr) if arr else np.array([])


csv_paths_exp = [
    "398_bta_exp_1111-completed.db_median_fluo1_fluo_intensities.csv",
    "p7_bta_exp_1111-completed.db_median_fluo1_fluo_intensities.csv",
    "398_bta_exp_1023-completed.db_median_fluo1_fluo_intensities (1).csv",
    "p7_bta_exp_1023-completed.db_median_fluo1_fluo_intensities.csv",
    "398_bta_exp_1015-completed.db_median_fluo1_fluo_intensities.csv",
    "p7_bta_exp_1015-completed.db_median_fluo1_fluo_intensities.csv",
    "398_bta_exp_1007-completed.db_median_fluo_intensities.csv",
    "p7_bta_exp_1007-completed.db_median_fluo_intensities.csv",
    "398_bta_0exp001-completed.db_median_fluo_intensities.csv",
    "p7_bta_0exp-completed.db_median_fluo_intensities.csv",
]

exp_vals = load_values(csv_paths_exp)
exp_bins = 15
exp_counts, exp_edges = np.histogram(exp_vals, bins=exp_bins)
exp_centers = (exp_edges[:-1] + exp_edges[1:]) / 2
exp_smooth = np.convolve(exp_counts, np.ones(3)/3, mode="same")


csv_paths_un = [
    "398_bta_13_1016-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_15_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1015-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_15_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_18_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1008-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_15_1008-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_18_1009-completed.db_median_fluo1_fluo_intensities_2.csv",
]

un_vals = load_values(csv_paths_un)
un_bins = 15
un_counts, un_edges = np.histogram(un_vals, bins=un_bins)
un_centers = (un_edges[:-1] + un_edges[1:]) / 2
un_smooth = np.convolve(un_counts, np.ones(3)/3, mode="same")

x_min = min(exp_centers.min(), un_centers.min())
x_max = max(exp_centers.max(), un_centers.max())

common_x = np.linspace(x_min, x_max, 1000)
exp_interp = np.interp(common_x, exp_centers, exp_smooth)
un_interp  = np.interp(common_x, un_centers, un_smooth)

diff = exp_interp - un_interp
roots = []
for i in range(len(diff)-1):
    if diff[i] * diff[i+1] < 0:
        x1, x2 = common_x[i], common_x[i+1]
        y1, y2 = diff[i], diff[i+1]
        xr = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(xr)

median_root = np.median(roots) if len(roots) else None

plt.figure(figsize=(9, 6))

plt.bar(exp_centers, exp_counts,
        width=(exp_edges[1]-exp_edges[0]),
        color="skyblue", alpha=0.5, edgecolor="k",
        label="exp histogram")

plt.bar(un_centers, un_counts,
        width=(un_edges[1]-un_edges[0]),
        color="lightgreen", alpha=0.5, edgecolor="k",
        label="un histogram")

plt.plot(exp_centers, exp_smooth, color="blue", lw=2, label="exp smoothed")
plt.plot(un_centers, un_smooth, color="green", lw=2, label="un smoothed")

for r in roots:
    y = np.interp(r, exp_centers, exp_smooth)
    plt.scatter(r, y, color="red",zorder=5)

if median_root is not None:
    plt.axvline(median_root, color="red", ls="--", lw=1.2)

info = (
    f"Intersections: {', '.join(f'{v:.4f}' for v in roots)}\n"
    f"exp total: {len(exp_vals)}\n"
    f"un total: {len(un_vals)}"
)

plt.text(0.98, 0.77, info,
         transform=plt.gca().transAxes, ha="right", va="top",
         fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3",
                   fc="white", ec="gray", alpha=0.9))

plt.xlabel("Median")
plt.ylabel("Cell count")
plt.title("exp vs un (original bin range & bin width)")

plt.xlim(x_min, x_max)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("exp_un_c.count_original.png", dpi=300)
plt.show()
