import pandas as pd
import matplotlib.pyplot as plt

# ファイルパスをリストで指定（最初のファイル＋3ファイル）
file_paths = [
    "p7_bta_0exp-completed.db_median_fluo_intensities.csv",  # 最初のファイル
    "p7_bta_1001-completed.db_median_fluo_intensities.csv",
    "p7_bta_13001-completed.db_median_fluo_intensities.csv",
    "p7_bta_15001-completed.db_median_fluo_intensities.csv"
]

# 最初のファイルを読み込み
file1_data = pd.read_csv(file_paths[0])

# 最初のファイル中のデータの5パーセンタイル値を計算
threshold = file1_data.select_dtypes(include=["number"]).quantile(0.05).min()

print(f"最初のファイルにおける5パーセンタイル閾値: {threshold}")

ratios = []
labels = []

# 他のファイルで閾値以下の値を持つ割合を計算
for i, file_path in enumerate(file_paths[1:], start=2):
    file_data = pd.read_csv(file_path)
    numeric_data = file_data.select_dtypes(include=["number"])
    count_below_threshold = (numeric_data < threshold).sum().sum()
    total_values = numeric_data.size
    ratio = count_below_threshold / total_values if total_values > 0 else 0
    ratios.append(float(ratio))
    labels.append(file_path)

print(f"割合:{ratios}")