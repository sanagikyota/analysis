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
    if len(arr) == 0:
        return np.array([])
    return np.concatenate(arr)


data_exp = load_group(csv_paths_exp)
data_un  = load_group(csv_paths_un)

if data_exp.size == 0 or data_un.size == 0:
    print("データが読み込めていません")
    exit()

# totals
total_exp = len(data_exp)
total_un = len(data_un)
print(f"Total cells - exp: {total_exp}, un: {total_un}")

bins = 15

exp_min, exp_max = data_exp.min(), data_exp.max()
exp_edges = np.linspace(exp_min, exp_max, bins + 1)
exp_centers = (exp_edges[:-1] + exp_edges[1:]) / 2
exp_hist, _ = np.histogram(data_exp, bins=exp_edges, density=True)

un_min, un_max = data_un.min(), data_un.max()
un_edges = np.linspace(un_min, un_max, bins + 1)
un_centers = (un_edges[:-1] + un_edges[1:]) / 2
un_hist, _ = np.histogram(data_un, bins=un_edges, density=True)

def smooth(x, w=3):
    # reflect padding to reduce edge artefacts
    if w <= 1:
        return x.astype(float)
    pad = w * 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    sm = np.convolve(xp, np.ones(w)/w, mode="same")
    return sm[pad:pad+len(x)]

exp_smooth = smooth(exp_hist, 3)
un_smooth  = smooth(un_hist, 3)

# interpolate un smoothed onto exp centers for intersection finding
un_interp = np.interp(exp_centers, un_centers, un_smooth)

diff = exp_smooth - un_interp
sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]

roots = []
for i in sign_change:
    x1, x2 = exp_centers[i], exp_centers[i+1]
    y1, y2 = diff[i], diff[i+1]
    # linear interpolation for root
    if (y2 - y1) != 0:
        xr = x1 - y1 * (x2 - x1) / (y2 - y1)
        roots.append(xr)

median_root = float(np.median(roots)) if len(roots) > 0 else None

# print intersection x-coordinates
if roots:
    print("Intersection x-values:", [round(r,6) for r in roots])
    print("Median intersection:", round(median_root,6))
else:
    print("No intersections found")

plt.figure(figsize=(9,6))

plt.plot(exp_centers, exp_smooth, label=f"exp smoothed", color="blue", lw=2)
plt.bar(exp_centers, exp_hist,
        width=(exp_edges[1]-exp_edges[0]), alpha=0.3, 
        color="blue", label="exp histogram")

plt.plot(un_centers, un_smooth, label=f"un smoothed", color="green", lw=2)
plt.bar(un_centers, un_hist,
        width=(un_edges[1]-un_edges[0]), alpha=0.3, 
        color="green", label="un histogram")
for r in roots:
    y = np.interp(r, exp_centers, exp_smooth)
    plt.scatter(r, y, color="red", zorder=5)

if len(roots) > 0:
     plt.axvline(median_root, color="red", ls="--", lw=1.2)

ax = plt.gca()
plt.legend(loc="upper right")

if roots:
    root_str = ", ".join([f"{x:.4f}" for x in roots])
    info_text = f"Intersections: {root_str}\nexp total cells: {total_exp}\nun total cells: {total_un}"
else:
    info_text = f"No intersections\nexp total cells: {total_exp}\nun total cells: {total_un}"

ax.text(0.98, 0.77, info_text,
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.xlabel("Median")
plt.ylabel("Probability density (PDF)")
plt.title("exp vs un (original bin range & bin width)")    
plt.xlim(0.625, 0.875)
plt.savefig("exp_un_pdf_original.png", dpi=300)
plt.tight_layout()
plt.show()

print("\n=== Intersections ===")
print("roots =", roots)
if len(roots) > 0:
    print("median =", np.median(roots))
else:
    print("No intersection found")
