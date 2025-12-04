import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


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


def read_values(path):
    df = pd.read_csv(path, header=None, skiprows=1, usecols=[0])
    v = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
    return v


def load_group(paths):
    arr = []
    for p in paths:
        if not os.path.exists(p):
            print("Missing:", p)
            continue
        vals = read_values(p)
        if vals.size > 0:
            arr.append(vals)
    return np.concatenate(arr) if arr else np.array([])



data_exp = load_group(csv_paths_exp)
data_un = load_group(csv_paths_un)

if data_exp.size == 0 or data_un.size == 0:
    print("データが読み込めていません")
    exit()

total_exp = len(data_exp)
total_un = len(data_un)
print(f"Total cells - exp: {total_exp}, un: {total_un}")


bins = 15
global_min = min(data_exp.min(), data_un.min())
global_max = max(data_exp.max(), data_un.max())
bin_edges = np.linspace(global_min, global_max, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


exp_hist, _ = np.histogram(data_exp, bins=bin_edges)
un_hist,  _ = np.histogram(data_un,  bins=bin_edges)

def smooth(x, w=3):
    if w <= 1:
        return x.astype(float)
    pad = w * 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    sm = np.convolve(xp, np.ones(w)/w, mode="same")
    return sm[pad:pad+len(x)]

exp_smooth = smooth(exp_hist, 3)
un_smooth = smooth(un_hist, 3)


diff = exp_smooth - un_smooth
roots = []
for i in range(len(diff)-1):
    y1, y2 = diff[i], diff[i+1]
    if y1 * y2 < 0:
        x1, x2 = bin_centers[i], bin_centers[i+1]
        xr = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(xr)

median_root = float(np.median(roots)) if roots else None

print("Intersections:", roots)
print("Median:", median_root)


plt.figure(figsize=(9, 6))

plt.plot(bin_centers, exp_smooth, label="exp smoothed", color="blue", lw=2)
plt.plot(bin_centers, un_smooth, label="un smoothed", color="green", lw=2)


plt.bar(bin_centers, exp_hist,
        width=(bin_edges[1]-bin_edges[0]), alpha=0.5,
        color="skyblue", edgecolor="k", label="exp histogram")

plt.bar(bin_centers, un_hist,
        width=(bin_edges[1]-bin_edges[0]), alpha=0.5,
        color="lightgreen", edgecolor="k", label="un histogram")

for r in roots:
    y = np.interp(r, bin_centers, exp_smooth)
    plt.scatter(r, y, color="red", zorder=5)

if median_root is not None:
    plt.axvline(median_root, color="red", ls="--", lw=1.2)


ax = plt.gca()

if roots:
    root_str = ", ".join([f"{x:.4f}" for x in roots])
    info_text = (
        f"Intersections: {root_str}\n"
        f"exp total cells: {total_exp}\n"
        f"un total cells: {total_un}"
    )
else:
    info_text = (
        f"No intersections\n"
        f"exp total cells: {total_exp}\n"
        f"un total cells: {total_un}"
    )

ax.text(0.98, 0.77, info_text,
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="white", ec="gray", alpha=0.9))

plt.xlabel("Median")
plt.ylabel("Cell count")
plt.title("exp vs un (common bin range & bin width)")

plt.xlim(global_min, global_max)

plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("exp_un_c.count_common.png", dpi=300)
plt.show()
