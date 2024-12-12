import pandas as pd

# ファイルパスをリストで指定
file_paths = [
    "sk402_005_bta_0-completed.db_median_fluo_intensities.csv",  # 最初のファイル
    "sk402_005_bta_07-completed.db_median_fluo_intensities.csv",  # 残りの5つのファイル
    "sk402_005_bta_1-completed.db_median_fluo_intensities.csv",
    "sk402_005_bta_13001-completed.db_median_fluo_intensities.csv",
    "sk402_005_bta_15001-completed.db_median_fluo_intensities.csv",
    "sk402_005_bta_18001-completed.db_median_fluo_intensities.csv"
]

# 最初のファイルを読み込み
file1_data = pd.read_csv(file_paths[0])

# 最初のファイル中のデータの5パーセンタイル値を計算
# (数値データのみを対象とする)
threshold = file1_data.select_dtypes(include=["number"]).quantile(0.05).min()

print(f"最初のファイルにおける5パーセンタイル閾値: {threshold}")

ratios = []

# 他のファイルで閾値以下の値を持つ割合を計算
for i, file_path in enumerate(file_paths[1:], start=2):
    # ファイルの読み込み
    file_data = pd.read_csv(file_path)
    
    # 数値データのみを対象とする
    numeric_data = file_data.select_dtypes(include=["number"])
    
    # 閾値以下の値の割合を計算
    count_below_threshold = (numeric_data < threshold).sum().sum()  # 閾値以下の総数
    total_values = numeric_data.size  # 全要素数
    ratio = count_below_threshold / total_values if total_values > 0 else 0  # 安全のため条件付き
    
    ratios.append(float(ratio))

print(f"割合:{ratios}")
