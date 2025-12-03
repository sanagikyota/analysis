import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. データファイルリスト
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. データ読み込み関数
# ---------------------------------------------------------
def load_and_combine_data(file_list):
    combined_data = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            # 数値データが含まれる最初の列を取得
            data_col = df.select_dtypes(include=[np.number]).iloc[:, 0]
            combined_data.extend(data_col.dropna().tolist())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return np.array(combined_data)

data_exp = load_and_combine_data(csv_paths_exp)
data_un = load_and_combine_data(csv_paths_un)

# ---------------------------------------------------------
# 3. 度数分布（Raw Count）の作成と平滑化
# ---------------------------------------------------------
# パラメータ調整
BIN_NUM = 200        # 解像度
WINDOW_SIZE = 15     # 移動平均の窓幅（少し強めにかけてノイズを消します）

# 共通のビンを作成
x_min = min(data_exp.min(), data_un.min())
x_max = max(data_exp.max(), data_un.max())
bins = np.linspace(x_min, x_max, BIN_NUM)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

def get_smoothed_count(data, bins, window):
    # density=False に変更。これで純粋な「個数」になります。
    counts, _ = np.histogram(data, bins=bins, density=False)
    
    # 移動平均を適用
    smoothed = pd.Series(counts).rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.values

y_exp = get_smoothed_count(data_exp, bins, WINDOW_SIZE)
y_un = get_smoothed_count(data_un, bins, WINDOW_SIZE)

# ---------------------------------------------------------
# 4. 交点の算出
# ---------------------------------------------------------
diff = y_exp - y_un
idx = np.argwhere(np.diff(np.sign(diff))).flatten()

intersections = []
for i in idx:
    x1, x2 = bin_centers[i], bin_centers[i+1]
    y_diff1, y_diff2 = diff[i], diff[i+1]
    
    # 線形補間で正確なXを求める
    x_cross = x1 + (x2 - x1) * abs(y_diff1) / (abs(y_diff1) + abs(y_diff2))
    
    # 交点のY座標（Exp側の曲線上の値を取得して補間）
    y1, y2 = y_exp[i], y_exp[i+1]
    y_cross = y1 + (y2 - y1) * ((x_cross - x1) / (x2 - x1))
    
    # ノイズ除去: ある程度の度数（最大度数の0.5%以上）がある場所のみ交点とする
    # 度数基準なので、裾野の小さなノイズ交差を拾わないようにします
    max_count = max(y_exp.max(), y_un.max())
    if y_cross > max_count * 0.005: 
        intersections.append((x_cross, y_cross))

# 交点のX座標を文字列化
cross_points_str = ", ".join([f"{p[0]:.2f}" for p in intersections])

# ---------------------------------------------------------
# 5. グラフ描画
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# データセット1 (Exp)
sns.lineplot(x=bin_centers, y=y_exp, label=f'Exp (N={len(data_exp)})', color='blue', linewidth=2)

# データセット2 (Un)
sns.lineplot(x=bin_centers, y=y_un, label=f'Un (N={len(data_un)})', color='orange', linewidth=2)

# 交点プロット
for (cx, cy) in intersections:
    plt.plot(cx, cy, 'ro', markersize=8, zorder=10)
    # 点の上に座標を表示
    plt.text(cx, cy + (max(y_exp)*0.02), f'{cx:.2f}', 
             color='red', ha='center', fontsize=10, fontweight='bold')

# 凡例用のダミー
plt.plot([], [], 'ro', label=f'Intersection X: {cross_points_str}')

plt.title("Smoothed Histogram (Raw Counts)", fontsize=15)
plt.xlabel("Fluorescence Intensity", fontsize=12)
plt.ylabel("Count (Frequency)", fontsize=12) # Y軸ラベルをCountに変更
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()