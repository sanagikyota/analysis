import sys
import os
import pandas as pd

default_csv = "398_bta_exp_1015-completed.db_median_fluo1_fluo_intensities.csv"
csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

# ファイル名を表示
print(f"Filename: {os.path.basename(csv_path)}")


# 1行目がヘッダなので skiprows=1
df = pd.read_csv(csv_path, header=None, skiprows=1, names=["median"])
vals = pd.to_numeric(df["median"].dropna(), errors="coerce").dropna()

if vals.empty:
    print("No numeric data found.")
    sys.exit(1)

threshold = vals.quantile(0.05)
print(f"5th percentile threshold: {threshold}")