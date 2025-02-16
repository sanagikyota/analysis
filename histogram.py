import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
csv_file = "sk402_01_bta_0001-completed.db_median_fluo_intensities (1).csv"  # CSVファイル名を指定してください
data = pd.read_csv(csv_file, header=None)  # header=Noneで列名なしとして読み込む

# ヒストグラムを作成
plt.hist(data[0], bins=30, color='b', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")

# グラフを表示
plt.show()
