import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# CSVファイルを複数指定（ここに増やしてください）
csv_paths = [
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

sns.set(style="whitegrid")

all_values = []

for csv_path in csv_paths:
    if not os.path.exists(csv_path):
        print(f"ファイルが見つかりません: {csv_path} -- スキップ")
        continue
    df = pd.read_csv(csv_path, header=None, skiprows=1, names=["median"])
    values = pd.to_numeric(df["median"].dropna(), errors="coerce")
    values = values.dropna().values
    if values.size > 0:
        all_values.append(values)

# データが無ければ終了
if len(all_values) == 0:
    print("有効なデータが見つかりません。csv_paths とファイルの中身を確認してください。")
else:
    data = np.concatenate(all_values)

    mu = data.mean()
    sigma = data.std(ddof=0)
    total_cells = len(data)  

    bins = 15
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, _ = ax.hist(data, bins=bins, color="skyblue", edgecolor="k")

    x = np.linspace(data.min(), data.max(), 200)
    if sigma > 0:
        pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        bin_width = bin_edges[1] - bin_edges[0]
        pdf_counts = pdf * total_cells * bin_width
        ax.plot(x, pdf_counts, color="red", lw=1.5, label="Normal fit (scaled)")

    ax.set_xlabel("Median")
    ax.set_ylabel("Cell count")
    ax.set_title("Histogram of median_exp")

    # 総細胞数のみをプロット上に表示
    info_text = f"Total cells: {total_cells}"
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.legend(loc="upper left")
    plt.tight_layout()

    out_png = "testgram.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")