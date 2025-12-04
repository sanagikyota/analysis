# ...existing code...
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

   # ...existing code...
    # 基本統計（プロット用）
    mu = data.mean()
    sigma = data.std(ddof=0)
    total_cells = len(data)  # 総細胞数

    # ヒストグラム（PDF表示に変更）
    bins = 15
    # ヒストグラムを密度 (density=True) で計算して PDF 表示にする
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    # 棒グラフは密度値で描画（cell count ではなく PDF）
    ax.bar(bin_centers - bin_width/2, counts, width=bin_width, color="skyblue", alpha=0.7,
           edgecolor="k", linewidth=0.8, align="edge", label="Histogram")

    # ビン中心を結ぶ線（密度）
    ax.plot(bin_centers, counts, color="black", marker="o", linestyle="-", lw=1)

    # 平滑化（移動平均）で曲線化（密度単位）
    window = 3
    smooth_counts = np.convolve(counts, np.ones(window)/window, mode="same")
    ax.plot(bin_centers, smooth_counts, color="black", linestyle="--", lw=1.5, label="Smoothed histogram")

    # 正規分布（PDF、ヒストグラムと比較するためスケール不要）
    x = np.linspace(data.min(), data.max(), 200)
    if sigma > 0:
        pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, pdf, color="red", lw=1.5, label="Normal PDF (μ,σ)")

    ax.set_xlabel("Median")
    ax.set_ylabel("Probability density (PDF)")
    ax.set_title("Histogram of median_exp (PDF)")

    # 総細胞数は注記として表示（PDF表示でも有用）
    info_text = f"Total cells: {total_cells}"
    ax.text(0.98, 0.80, info_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out_png = "exp_histogram_pdf.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"Saved: {out_png}")