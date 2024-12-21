import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, BLOB

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

# データベースファイルのパス
db_path = "sqlite:///sk398_01_bta_exp-completed-2024-12-20.db"  # 適切なパスに変更してください

# エンジンの作成とセッションの設定
engine = create_engine(db_path)
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
    
    # 結果をリストに変換
    db_values = [result[1] for result in results]
    print(f"抽出された数値の個数: {len(db_values)}")
    # 抽出したareaを表示
    for value in db_values:
        print(value)
except Exception as e:
    print(f"エラーが発生しました: {e}")
    db_values = []
finally:
    session.close()

# CSVファイルから数値の集団を読み込む（1行目をスキップ）
csv_file = "sk398_01_bta_exp-completed.db_median_fluo_intensities.csv"  # 適切なCSVファイルのパスに変更してください
csv_data = pd.read_csv(csv_file, header=None, skiprows=1)
csv_values = csv_data[0].tolist()  # 1列目のデータをリストに変換

# db_valuesとcsv_valuesの数値の組み合わせを表示
for db_value, csv_value in zip(db_values, csv_values):
    print(f"db_value: {db_value}, csv_value: {csv_value}")

# データフレームに変換
df = pd.DataFrame({'area': db_values, 'median': csv_values})

# Seabornを用いて散布図をプロット
plt.figure(figsize=(8, 8))
sns.set(style="darkgrid")
scatter_plot = sns.scatterplot(x='area', y='median', data=df, s=60, color='blue')

plt.xlabel("area")
plt.ylabel("median")
plt.title("Correlation between area and median")
plt.show()
plt.savefig("plot_cor_398exp_area-med_241217.png")