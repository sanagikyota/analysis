import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


csv_paths_agg = [
    "398_bta_15_1008-completed.db_median_fluo1_fluo_intensities_1.csv",
    "398_bta_18_1008-completed.db_median_fluo1_fluo_intensities_1.csv",
    "398_bta_18_1015-completed.db_median_fluo1_fluo_intensities_1.csv",
    "p7_bta_15_1023-completed.db_median_fluo1_fluo_intensities_1.csv",
    "p7_bta_18_1009-completed.db_median_fluo1_fluo_intensities_1.csv",
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
    return np.concatenate(arr) if len(arr) else np.array([])



data_agg = load_group(csv_paths_agg)
data_un  = load_group(csv_paths_un)

if data_agg.size == 0 or data_un.size == 0:
    print("データ読み込みエラー")
    exit()

total_agg = len(data_agg)
total_un = len(data_un)
print(f"Total cells - agg: {total_agg}, un: {total_un}")


combined_min = min(data_agg.min(), data_un.min())
combined_max = max(data_agg.max(), data_un.max())

bins = 15
bin_edges = np.linspace(combined_min, combined_max, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]



hist_agg, _ = np.histogram(data_agg, bins=bin_edges, density=True)
hist_un,  _ = np.histogram(data_un,  bins=bin_edges, density=True)



def smooth(x, w=3):
    return np.convolve(x, np.ones(w)/w, mode="same")

smooth_agg = smooth(hist_agg, 3)
smooth_un  = smooth(hist_un, 3)



diff = smooth_agg - smooth_un
idx = np.where(np.diff(np.sign(diff)) != 0)[0]

roots = []
for i in idx:
    x1, x2 = bin_centers[i], bin_centers[i+1]
    y1, y2 = diff[i], diff[i+1]
    xr = x1 - y1 * (x2 - x1) / (y2 - y1)
    roots.append(xr)

median_root = np.median(roots) if len(roots) else None



plt.figure(figsize=(9,6))


plt.bar(bin_centers - bin_width/2, hist_agg, width=bin_width,
        color="orange", alpha=0.3, label="agg histogram")
plt.bar(bin_centers - bin_width/2, hist_un, width=bin_width,
        color="green", alpha=0.3, label="un histogram")


plt.plot(bin_centers, smooth_agg, color="orange", lw=2, label="agg smoothed")
plt.plot(bin_centers, smooth_un,  color="green", lw=2, label="un smoothed")


for r in roots:
    y = np.interp(r, bin_centers, smooth_agg)
    plt.scatter(r, y, color="red", zorder=5)

if median_root:
    plt.axvline(median_root, color="red", ls="--", lw=1.2)

ax = plt.gca()
plt.legend(loc="upper right")

if roots:
    root_str = ", ".join([f"{x:.4f}" for x in roots])
    info_text = f"Intersections: {root_str}\nagg total cells: {total_agg}\nun total cells: {total_un}"
else:
    info_text = f"No intersections\nagg total cells: {total_agg}\nun total cells: {total_un}"

ax.text(0.98, 0.77, info_text,
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.xlabel("Median")
plt.ylabel("Probability density (PDF)")
plt.title("agg vs un (common bin range & bin width)")
plt.savefig("agg_un_pdf_common.png", dpi=300)
plt.tight_layout()
plt.show()
