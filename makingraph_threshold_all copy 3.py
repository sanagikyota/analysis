import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# ---------------------------------------------------------
# 1. ファイルリストの定義
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
# 2. データの読み込みと結合関数
# ---------------------------------------------------------
def load_and_combine_data(file_list):
    combined_data = []
    for file_path in file_list:
        try:
            # CSVを読み込む（ヘッダーがある前提。ない場合はheader=Noneを指定）
            df = pd.read_csv(file_path)
            
            # 数値データが含まれる最初の列を取得すると仮定
            # ※特定の列名が決まっている場合は df['ColumnName'] と指定してください
            data_col = df.select_dtypes(include=[np.number]).iloc[:, 0]
            
            combined_data.extend(data_col.dropna().tolist())
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return np.array(combined_data)

# データをロード
data_exp = load_and_combine_data(csv_paths_exp)
data_un = load_and_combine_data(csv_paths_un)

# ---------------------------------------------------------
# 3. 分布曲線（KDE）の作成と交点の計算
# ---------------------------------------------------------
# 評価するためのX軸の範囲を決定
x_min = min(data_exp.min(), data_un.min())
x_max = max(data_exp.max(), data_un.max())
x_grid = np.linspace(x_min, x_max, 1000) # 1000分割で滑らかに

# KDE（カーネル密度推定）で滑らかな確率密度関数を計算
kde_exp = gaussian_kde(data_exp)(x_grid)
kde_un = gaussian_kde(data_un)(x_grid)

# 交点を見つける（差分の符号が反転する場所を探す）
diff = kde_exp - kde_un
idx = np.argwhere(np.diff(np.sign(diff))).flatten()

# 交点の座標を取得（y値が低すぎるノイズのような交点は除外する処理を入れるとより堅牢です）
intersections = []
for i in idx:
    # 簡易的な線形補間でより正確なXを求める
    x1, x2 = x_grid[i], x_grid[i+1]
    y1, y2 = diff[i], diff[i+1]
    # 0になるxを求める: x = x1 + (x2 - x1) * |y1| / (|y1| + |y2|)
    x_cross = x1 + (x2 - x1) * abs(y1) / (abs(y1) + abs(y2))
    
    # そのX地点での高さ(Y)をKDEから再取得
    y_cross = gaussian_kde(data_exp)(x_cross)[0]
    
    # ピークの高さの1%以上ある交点のみを採用（裾野のノイズ対策）
    max_peak = max(kde_exp.max(), kde_un.max())
    if y_cross > max_peak * 0.01:
        intersections.append((x_cross, y_cross))

# 主な交点（通常は中央付近の1つ）を取得。複数ある場合はリスト化
cross_points_str = ", ".join([f"{p[0]:.2f}" for p in intersections])

# ---------------------------------------------------------
# 4. グラフ描画
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# データセット1 (Exp) のプロット
label_exp = f'Exp (N={len(data_exp)})'
sns.lineplot(x=x_grid, y=kde_exp, label=label_exp, color='blue', linewidth=2)

# データセット2 (Un) のプロット
label_un = f'Un (N={len(data_un)})'
sns.lineplot(x=x_grid, y=kde_un, label=label_un, color='orange', linewidth=2)

# 交点のプロット
for (cx, cy) in intersections:
    plt.plot(cx, cy, 'ro', markersize=8, zorder=10) # 赤い点
    # グラフ上に値をテキスト表示したい場合は以下をコメントアウト解除
    # plt.text(cx, cy, f'{cx:.2f}', color='red', verticalalignment='bottom')

# 凡例に交点情報を追加するためのダミープロット
plt.plot([], [], 'ro', label=f'Intersection X: {cross_points_str}')

# グラフ装飾
plt.title("Smoothed Distribution of Fluorescence Intensities", fontsize=15)
plt.xlabel("Fluorescence Intensity", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()
