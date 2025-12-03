import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# --- 編集可: 各グループのCSVリスト（既存ファイル名に合わせてください） ---
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
    "398_bta_15_1008-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_15_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1008-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1015-completed.db_median_fluo1_fluo_intensities_2.csv",
    "398_bta_18_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_15_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_18_1009-completed.db_median_fluo1_fluo_intensities_2.csv",
    "p7_bta_18_1023-completed.db_median_fluo1_fluo_intensities_2.csv",
]

# parameters
bins = 15
smooth_window = 3  # odd preferred

def load_concat(csv_list):
    arrs = []
    for p in csv_list:
        if not os.path.exists(p):
            print(f"Skipping missing: {p}")
            continue
        df = pd.read_csv(p, header=None, skiprows=1, names=["median"])
        vals = pd.to_numeric(df["median"].dropna(), errors="coerce").dropna().values
        if vals.size:
            arrs.append(vals)
    return np.concatenate(arrs) if arrs else np.array([])

# load
data_exp = load_concat(csv_paths_exp)
data_un = load_concat(csv_paths_un)

if data_exp.size == 0 or data_un.size == 0:
    print("One or both groups have no data. Check file paths.")
    raise SystemExit(1)

# common histogram bins
counts_exp, bin_edges = np.histogram(data_exp, bins=bins)
counts_un, _ = np.histogram(data_un, bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# smoothing (moving average)
def moving_avg(arr, w):
    if w <= 1:
        return arr.astype(float)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')

smooth_exp = moving_avg(counts_exp.astype(float), smooth_window)
smooth_un = moving_avg(counts_un.astype(float), smooth_window)

# find intersections by linear interpolation between consecutive centers where sign changes
diff = smooth_exp - smooth_un
roots = []
for i in range(len(diff)-1):
    y0, y1 = diff[i], diff[i+1]
    if y0 == 0:
        roots.append(bin_centers[i])
    if y0 * y1 < 0:
        x0, x1 = bin_centers[i], bin_centers[i+1]
        # linear interp for root
        xr = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(xr)

if len(roots) == 0:
    print("No intersections found between smoothed curves.")
else:
    median_root = float(np.median(roots))
    print(f"Intersections (x): {roots}")
    print(f"Median of intersection x-values: {median_root}")

# plot for visual check
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(bin_centers, smooth_exp, color="blue", lw=1.8, label="exp (smoothed)")
ax.plot(bin_centers, smooth_un, color="green", lw=1.8, label="un (smoothed)")
ax.plot(bin_centers, counts_exp, color="blue", alpha=0.25, ls=':', label="exp raw counts")
ax.plot(bin_centers, counts_un, color="green", alpha=0.25, ls=':', label="un raw counts")
if roots:
    ax.scatter(roots, [np.interp(r, bin_centers, smooth_exp) for r in roots], color="red", zorder=5, label="intersections")
    ax.text(0.98, 0.02, f"Median intersection: {median_root:.4f}" if roots else "No intersections",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("Median fluorescence intensity (bin centers)")
ax.set_ylabel("Cell count (smoothed)")
ax.set_title("Smoothed hist counts and intersections")
ax.legend()
ax.set_xlim(0.65, 0.9)
plt.tight_layout()
out_png = "intersection_smoothed_curves.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved plot: {out_png}")