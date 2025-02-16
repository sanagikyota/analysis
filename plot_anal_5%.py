import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_list = [
    "sk398_01_bta_0001-completed.db_median_fluo_intensities.csv",
    "SK398_01_bta_0001-completed.db_median_fluo_intensities(2).csv",
    "sk402_01_bta_0001-completed.db_median_fluo_intensities (1).csv",
    "SK402_01_bta_0001-completed.db_median_fluo_intensities (2).csv",
]

# 各ファイルに対応する名前を設定
file_names = [
    "SK398_canny40",
    "SK398",
    "SK402_canny40",
    "SK402",
]

percentile_values = []

for file in file_list:
    # ファイルの存在を確認
    if os.path.exists(file):
        try:
            # CSVデータを読み込み
            data = pd.read_csv(file, header=None)  # ヘッダーなしで読み込み
            # 数値データに変換
            data = data.apply(pd.to_numeric, errors='coerce')
            # 1列目（または全データ）の下位5パーセンタイルを計算
            percentile_5 = data[0].quantile(0.05)
            percentile_values.append(percentile_5)
            print(f"{file} の下位5パーセンタイル: {percentile_5}")
        except Exception as e:
            print(f"ファイル {file} の処理中にエラーが発生しました: {e}")
    else:
        print(f"ファイル {file} が見つかりませんでした。")

# データを整形（棒グラフ用に準備）
df = pd.DataFrame({"File": file_names, "5th Percentile": percentile_values})

# # SK402 01 BTAの5th Percentile Valueを取得
# sk402_01_bta_value = df[df["File"] == "SK402 01 BTA 1128"]["5th Percentile"].values[0]

# # カスタムカラーリストを作成
# colors = ['blue' if value > sk402_01_bta_value else 'red' for value in df["5th Percentile"]]

# 棒グラフを作成
plt.figure(figsize=(10, 6))
sns.barplot(x="File", y="5th Percentile", data=df, width=0.5)  # カスタムカラーリストを使用
plt.xlabel("Sample", fontsize=12)
plt.ylabel("5th Percentile Value", fontsize=12)
plt.title("Comparison of 5th Percentile Values", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig_5th_percentile_comparison_250217.png")

# グラフを表示
plt.show()