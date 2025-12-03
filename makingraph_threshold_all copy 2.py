# ...existing code...
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- edit lists if needed ---
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

def load_concat(csv_list):
    arrs = []
    total = 0
    for p in csv_list:
        if not os.path.exists(p):
            print(f"Skipping missing: {p}")
            continue
        df = pd.read_csv(p, header=None, skiprows=1, names=["median"])
        vals = pd.to_numeric(df["median"].dropna(), errors="coerce").dropna().values
        if vals.size:
            arrs.append(vals)
            total += vals.size
    return (np.concatenate(arrs) if arrs else np.array([]), total)

# load data and counts
data_exp, total_exp = load_concat(csv_paths_exp)
data_un, total_un = load_concat(csv_paths_un)

if data_exp.size == 0 or data_un.size == 0:
    print("One or both groups have no data. Check file paths.")
    raise SystemExit(1)

# histogram bins and smoothing parameters
bins = 40
# choose smoothing method: "moving" or "gaussian"
smoothing_method = "gaussian"
# parameters:
moving_window = 5   # odd integer for moving average
gauss_sigma = 1.0   # sigma in bins for gaussian kernel

if moving_window % 2 == 0:
    moving_window += 1

# common bin edges from combined data
all_data = np.concatenate([data_exp, data_un])
bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

counts_exp, _ = np.histogram(data_exp, bins=bin_edges)
counts_un, _ = np.histogram(data_un, bins=bin_edges)

def moving_avg(arr, w):
    if w <= 1:
        return arr.astype(float)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')

def gaussian_smooth(arr, sigma):
    if sigma <= 0:
        return arr.astype(float)
    # kernel radius ~ 3 sigma
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return np.convolve(arr, kernel, mode='same')

# apply smoothing
if smoothing_method == "moving":
    smooth_exp = moving_avg(counts_exp.astype(float), moving_window)
    smooth_un = moving_avg(counts_un.astype(float), moving_window)
else:
    smooth_exp = gaussian_smooth(counts_exp.astype(float), gauss_sigma)
    smooth_un = gaussian_smooth(counts_un.astype(float), gauss_sigma)

# find intersections (linear interpolation between bin_centers)
diff = smooth_exp - smooth_un
roots = []
for i in range(len(diff)-1):
    y0, y1 = diff[i], diff[i+1]
    if y0 == 0:
        roots.append(bin_centers[i])
    if y0 * y1 < 0:
        x0, x1 = bin_centers[i], bin_centers[i+1]
        xr = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(xr)

median_root = float(np.median(roots)) if roots else None

# print results
print(f"Total cells (exp): {total_exp}")
print(f"Total cells (un) : {total_un}")
if roots:
    print("Intersection x-values:", [round(r,6) for r in roots])
    print("Median intersection:", round(median_root,6))
else:
    print("No intersections found between smoothed histograms.")

# plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8,5))

# plot smoothed curves
ax.plot(bin_centers, smooth_exp, color="blue", lw=2, label=f"exp (smoothed, {smoothing_method})")
ax.plot(bin_centers, smooth_un, color="green", lw=2, label=f"un  (smoothed, {smoothing_method})")

# optional: show raw counts faintly
ax.plot(bin_centers, counts_exp, color="blue", alpha=0.25, ls=':', label="exp raw counts")
ax.plot(bin_centers, counts_un, color="green", alpha=0.25, ls=':', label="un raw counts")

# mark intersections and median
if roots:
    y_at_roots = [np.interp(r, bin_centers, smooth_exp) for r in roots]
    ax.scatter(roots, y_at_roots, color="red", zorder=5, label="intersections")
    ax.axvline(median_root, color="red", ls="--", lw=1)
    ax.text(0.98, 0.02, f"Median intersection: {median_root:.4f}\nTotal exp: {total_exp}\nTotal un: {total_un}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
else:
    ax.text(0.98, 0.02, f"No intersections\nTotal exp: {total_exp}\nTotal un: {total_un}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

ax.set_xlabel("Median fluorescence intensity")
ax.set_ylabel("Cell count (smoothed)")
ax.set_title("Smoothed histograms (exp vs un)")
ax.legend()
ax.set_xlim(0.625, 0.875)
plt.tight_layout()

out_png = "smoothed_intersections_combined.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out_png}")
