import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


csv_paths_non = [
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

csv_paths_par = [
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

csv_paths_com = [
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


data_non = load_group(csv_paths_non)
data_par  = load_group(csv_paths_par)
data_com = load_group(csv_paths_com)

if data_non.size == 0 or data_par.size == 0 or data_com.size == 0:
    print("データ読み込みエラー")
    exit()

total_non = len(data_non)
total_par  = len(data_par)
total_com = len(data_com)
print(f"Total cells - non: {total_non}, par: {total_par}, com: {total_com}")

combined_min = min(data_non.min(), data_par.min(), data_com.min())
combined_max = max(data_non.max(), data_par.max(), data_com.max())
# Use a denser binning based on Freedman-Diaconis with a floor for stability.
combined_data = np.concatenate([data_non, data_par, data_com])
iqr = np.subtract(*np.percentile(combined_data, [75, 25]))
fd_width = 2 * iqr / (len(combined_data) ** (1 / 3)) if len(combined_data) > 0 else 0
fd_bins = int(np.ceil((combined_max - combined_min) / fd_width)) if fd_width > 0 else 0
# Slightly wider bins: use 80% of FD-estimated bin count with a modest floor.
bins = max(int(fd_bins * 0.8), 44)
bin_edges   = np.linspace(combined_min, combined_max, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width   = bin_edges[1] - bin_edges[0]

hist_non, _  = np.histogram(data_non, bins=bin_edges, density=True)
hist_par, _   = np.histogram(data_par,  bins=bin_edges, density=True)
hist_com, _  = np.histogram(data_com, bins=bin_edges, density=True)

def smooth(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="same")

smooth_non = smooth(hist_non, 3)
smooth_par  = smooth(hist_par, 3)
smooth_com = smooth(hist_com, 3)


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


roots_com_par = find_intersections(bin_centers, smooth_com, smooth_par)
roots_par_non = find_intersections(bin_centers, smooth_par,  smooth_non)

all_roots = sorted(roots_com_par + roots_par_non)

print("\ncom-par intersections:", roots_com_par)
print("par-non intersections:", roots_par_non)
print("all intersections (sorted):", all_roots)

plt.figure(figsize=(9, 6))

plt.bar(bin_centers - bin_width/2, hist_com, width=bin_width,
        color="orange", alpha=0.3, label="com histogram")
plt.bar(bin_centers - bin_width/2, hist_par,  width=bin_width,
        color="green",  alpha=0.3, label="par histogram")
plt.bar(bin_centers - bin_width/2, hist_non, width=bin_width,
        color="blue",   alpha=0.3, label="non histogram")

plt.plot(bin_centers, smooth_com, color="orange", lw=2, label="com smoothed")
plt.plot(bin_centers, smooth_par,  color="green",  lw=2, label="par smoothed")
plt.plot(bin_centers, smooth_non, color="blue",   lw=2, label="non smoothed")

for r in roots_com_par:
    y = np.interp(r, bin_centers, smooth_com)
    plt.scatter(r, y, color="red", s=40, zorder=5)
    plt.axvline(r, color="red", ls="--", lw=1.2)

for r in roots_par_non:
    y = np.interp(r, bin_centers, smooth_par)
    plt.scatter(r, y, color="red", s=40, zorder=5)
    plt.axvline(r, color="red", ls="--", lw=1.2)

ax = plt.gca()
plt.legend(loc="upper left")

def fmt_roots(vals):
    return ", ".join(fr"$\mathbf{{{x:.4f}}}$" for x in vals) if vals else "None"

info_lines = [
    f"com-par: {fmt_roots(roots_com_par)}",
    f"par-non: {fmt_roots(roots_par_non)}",
    f"com total cells: {total_com}",
    f"par total cells:  {total_par}",
    f"non total cells: {total_non}",
]
info_text = "\n".join(info_lines)

ax.text(0.02, 0.69, info_text,
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))


plt.xlabel("Median")
plt.ylabel("Probability density (PDF)")
plt.title("com vs par vs non")
plt.xlim(combined_min, combined_max)
plt.tight_layout()
plt.savefig("all_pdf_common.png", dpi=300)
plt.show()
