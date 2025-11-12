import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルパスをリストで指定（最初のファイル＋4ファイル＝合計5ファイル）
file_paths = [
    "p7ara_bta_exp_1111-completed.db_median_fluo1_fluo_intensities.csv",  # 最初のファイル
    "p7ara_bta_1_1111-completed.db_median_fluo1_fluo_intensities.csv",
    "p7ara_bta_13_1111-completed.db_median_fluo1_fluo_intensities.csv",
    "p7ara_bta_15_1111-completed.db_median_fluo1_fluo_intensities.csv",
    "p7ara_bta_18_1111-completed.db_median_fluo1_fluo_intensities.csv"
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

# makingraph の処理をここに追加 -- 先頭に 0.05 を付けた値列でグラフ作成
values = [0.05] + ratios
plot_labels = ['0', '1.0', '1.3', '1.5', '1.8']  # 必要なら変更可（values と長さを合わせること）

# DataFrame 作成して描画
df_plot = pd.DataFrame({'Label': plot_labels, 'Value': values})
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.barplot(x='Label', y='Value', data=df_plot, color='blue')

plt.title('pKJE7_Ara')
plt.ylim(0, 1)
plt.ylabel('Formation Rate')
plt.xlabel('Butanol Concentration (v/v%)')

# 保存と表示
outfile = 'pKJE7_Ara_barplot_251111.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.show()