import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, BLOB
from scipy.stats import spearmanr
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Baseクラスの定義
Base = declarative_base()

# Cellクラスの定義
class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(Float)
    area = Column(Float)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(Float)
    center_y = Column(Float)

# データベースファイルとCSVファイルのリスト
db_files = ["sqlite:///sk398_01_bta_0001-completed (1).db", "sqlite:///sk402_01_bta_0-completed.db"]
csv_files = ["sk398_01_bta_0001-completed.db_median_fluo_intensities (1).csv", "sk402_01_bta_0-completed.db_median_fluo_intensities.csv"]

# データを格納するリスト
all_db_values = []
all_csv_values = []
colors = []

# データベースファイルとCSVファイルを個別に処理
for i, (db_file, csv_file) in enumerate(zip(db_files, csv_files)):
    # エンジンの作成とセッションの設定
    engine = create_engine(db_file)
    Session = sessionmaker(bind=engine)
    session = Session()

    # manual_labelが1のareaを抽出し、cell_idが特定の順序で並べ替え
    try:
        results = session.query(Cell.cell_id, Cell.area).filter(Cell.manual_label == 1).all()
        
        # cell_idをカスタムソート
        def custom_sort_key(cell):
            parts = cell[0].split('C')
            f_part = int(parts[0][1:])  # 'F'の後の数字
            c_part = int(parts[1])      # 'C'の後の数字
            return (f_part, c_part)
        
        results.sort(key=custom_sort_key)
        
        # 結果をリストに変換して追加
        db_values = [result[1] for result in results]
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        db_values = []
    finally:
        session.close()

    # # db_valuesを0から1で正規化
    # scaler = MinMaxScaler()
    # db_values_normalized = scaler.fit_transform(np.array(db_values).reshape(-1, 1)).flatten()

    # CSVファイルからデータを読み込む
    csv_data = pd.read_csv(csv_file, header=None, skiprows=1)
    csv_values = csv_data[0].tolist()  # 1列目のデータをリストに変換

    # リストの長さを確認
    if len(db_values) != len(csv_values):
        print(f"エラー: db_values_normalizedの長さ({len(db_values)})とcsv_valuesの長さ({len(csv_values)})が一致しません。")
    else:
        # データセットA, Bを作成
        all_db_values.extend(db_values)
        all_csv_values.extend(csv_values)
        # 色を設定
        if i == 0:
            colors.extend(['red'] * len(db_values))
        else:
            colors.extend(['blue'] * len(db_values))

# all_db_valuesを0から1で正規化
scaler = MinMaxScaler()
all_db_values_normalized = scaler.fit_transform(np.array(all_db_values).reshape(-1, 1)).flatten()

# all_db_values_normalizedとall_csv_valuesの数値の組み合わせを表示
for db_value, csv_value in zip(all_db_values_normalized, all_csv_values):
    print(f"db_value: {db_value}, csv_value: {csv_value}")

# データフレームに変換
df = pd.DataFrame({'area': all_db_values_normalized, 'median': all_csv_values, 'color': colors})

# Seabornを用いて散布図をプロット
plt.figure(figsize=(8, 8))
sns.set(style="darkgrid")
scatter_plot = sns.scatterplot(x='area', y='median', hue='color', palette=['red', 'blue'], data=df, s=60)
plt.ylim(-0.05, 1.05)
plt.xlim(-0.05, 1.05)

# 凡例のラベルをカスタマイズ
handles, labels = scatter_plot.get_legend_handles_labels()
scatter_plot.legend(handles=handles, labels=['sk398', 'sk402'], loc='lower right', fontsize=15)

# スピアマンの相関係数とp値を計算
spearman_corr, p_value = spearmanr(all_db_values_normalized, all_csv_values)

# 相関係数とp値を表示
plt.text(0.05, 0.95, f'Spearman = {spearman_corr:.2f}, p-value = {p_value:.2e}', 
         transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

plt.xlabel("area")
plt.ylabel("median")
plt.title("Correlation between area and median")
plt.savefig("fig_plot_cor_398402_area-med_241128.png")
plt.show()