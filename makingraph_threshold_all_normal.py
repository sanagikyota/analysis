# ...existing code...
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

# compute mean/std
mu_exp, sigma_exp = data_exp.mean(), data_exp.std(ddof=0)
mu_un, sigma_un = data_un.mean(), data_un.std(ddof=0)
print(f"exp: n={len(data_exp)} mean={mu_exp:.4f} std={sigma_exp:.4f}")
print(f"un:  n={len(data_un)} mean={mu_un:.4f} std={sigma_un:.4f}")

# x range (slightly extended)
x_min = min(data_exp.min(), data_un.min())
x_max = max(data_exp.max(), data_un.max())
pad = (x_max - x_min) * 0.05
x = np.linspace(x_min - pad, x_max + pad, 4000)

# safe normal pdf
def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

pdf_exp = normal_pdf(x, mu_exp, sigma_exp)
pdf_un  = normal_pdf(x, mu_un,  sigma_un)

# find intersections: roots of pdf_exp - pdf_un (linear interp)
diff = pdf_exp - pdf_un
roots = []
for i in range(len(diff)-1):
    y0, y1 = diff[i], diff[i+1]
    if y0 == 0:
        roots.append(x[i])
    if y0 * y1 < 0:
        xr = x[i] - y0 * (x[i+1] - x[i]) / (y1 - y0)
        roots.append(xr)

if roots:
    median_root = float(np.median(roots))
    print("Intersections (x):", [round(r,6) for r in roots])
    print("Median intersection:", round(median_root,6))
else:
    median_root = None
    print("No intersections found between PDFs.")

# plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x, pdf_exp, color="blue", lw=2, label=f"exp (μ={mu_exp:.3f}, σ={sigma_exp:.3f})")
ax.plot(x, pdf_un,  color="green", lw=2, label=f"un  (μ={mu_un:.3f}, σ={sigma_un:.3f})")

# mark intersections
if roots:
    y_roots = np.interp(roots, x, pdf_exp)
    ax.scatter(roots, y_roots, color="red", zorder=6, label="intersections")
    ax.axvline(median_root, color="red", ls="--", lw=1)
    ax.text(0.98, 0.02, f"Median intersection: {median_root:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.set_xlabel("Median fluorescence intensity")
ax.set_ylabel("Probability density (PDF)")
ax.set_title("Normal distributions (exp vs un) and intersections")
ax.legend()
plt.tight_layout()
out_png = "normal_overlap_intersections.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out_png}")
# ...existing code...