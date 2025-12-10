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

csv_paths_agg = [
    "398_bta_15_1008-completed.db_median_fluo1_fluo_intensities_1.csv",
    "398_bta_18_1008-completed.db_median_fluo1_fluo_intensities_1.csv",
    "398_bta_18_1015-completed.db_median_fluo1_fluo_intensities_1.csv",
    "p7_bta_15_1023-completed.db_median_fluo1_fluo_intensities_1.csv",
    "p7_bta_18_1009-completed.db_median_fluo1_fluo_intensities_1.csv",
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


data_exp = load_group(csv_paths_exp)
data_un  = load_group(csv_paths_un)
data_agg = load_group(csv_paths_agg)

if data_exp.size == 0 or data_un.size == 0 or data_agg.size == 0:
    print("データ読み込みエラー")
    exit()

total_exp = len(data_exp)
total_un  = len(data_un)
total_agg = len(data_agg)
print(f"Total cells - exp: {total_exp}, un: {total_un}, agg: {total_agg}")

combined_min = min(data_exp.min(), data_un.min(), data_agg.min())
combined_max = max(data_exp.max(), data_un.max(), data_agg.max())

bins = 15
bin_edges   = np.linspace(combined_min, combined_max, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width   = bin_edges[1] - bin_edges[0]

hist_exp, _  = np.histogram(data_exp, bins=bin_edges, density=True)
hist_un, _   = np.histogram(data_un,  bins=bin_edges, density=True)
hist_agg, _  = np.histogram(data_agg, bins=bin_edges, density=True)

def smooth(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="same")

smooth_exp = smooth(hist_exp, 3)
smooth_un  = smooth(hist_un, 3)
smooth_agg = smooth(hist_agg, 3)


def find_intersections(x, y1, y2):
    diff = y1 - y2
    roots = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            x1, x2 = x[i], x[i + 1]
            yA, yB = diff[i], diff[i + 1]
            xr = x1 - yA * (x2 - x1) / (yB - yA)
            roots.append(xr)
    return roots


roots_agg_un = find_intersections(bin_centers, smooth_agg, smooth_un)
roots_un_exp = find_intersections(bin_centers, smooth_un,  smooth_exp)

all_roots = sorted(roots_agg_un + roots_un_exp)

print("\nagg–un intersections:", roots_agg_un)
print("un–exp intersections:", roots_un_exp)
print("all intersections (sorted):", all_roots)

plt.figure(figsize=(9, 6))

plt.bar(bin_centers - bin_width/2, hist_agg, width=bin_width,
        color="orange", alpha=0.3, label="agg histogram")
plt.bar(bin_centers - bin_width/2, hist_un,  width=bin_width,
        color="green",  alpha=0.3, label="un histogram")
plt.bar(bin_centers - bin_width/2, hist_exp, width=bin_width,
        color="blue",   alpha=0.3, label="exp histogram")

plt.plot(bin_centers, smooth_agg, color="orange", lw=2, label="agg smoothed")
plt.plot(bin_centers, smooth_un,  color="green",  lw=2, label="un smoothed")
plt.plot(bin_centers, smooth_exp, color="blue",   lw=2, label="exp smoothed")

for r in roots_agg_un:
    y = np.interp(r, bin_centers, smooth_agg)
    plt.scatter(r, y, color="red", s=40, zorder=5)
    plt.axvline(r, color="red", ls="--", lw=1.2)

for r in roots_un_exp:
    y = np.interp(r, bin_centers, smooth_un)
    plt.scatter(r, y, color="red", s=40, zorder=5)
    plt.axvline(r, color="red", ls="--", lw=1.2)

ax = plt.gca()
plt.legend(loc="upper left")

info_lines = [
    "agg–un: " + (", ".join(f"{x:.4f}" for x in roots_agg_un) if roots_agg_un else "None"),
    "un–exp: " + (", ".join(f"{x:.4f}" for x in roots_un_exp) if roots_un_exp else "None"),
    # "all (sorted): " + (", ".join(f"{x:.4f}" for x in all_roots) if all_roots else "None"),
    f"agg total cells: {total_agg}",
    f"un total cells:  {total_un}",
    f"exp total cells: {total_exp}",
]
info_text = "\n".join(info_lines)

ax.text(0.02, 0.69, info_text,
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.xlabel("Median")
plt.ylabel("Probability density (PDF)")
plt.title("agg vs un vs exp (common bin range & bin width)")
plt.xlim(combined_min, combined_max)
plt.tight_layout()
plt.savefig("all_pdf_common.png", dpi=300)
plt.show()
