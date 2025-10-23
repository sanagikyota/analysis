import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# データの準備
values = [0.05, 0.7857142857142857, 1.0, 0.9733333333333334, 0.9897959183673469]  # 各ファイルの割合
labels = ['0', '1.0', '1.3', '1.5', '1.8']  # 任意のラベルに変更可能

# データフレームに変換
df = pd.DataFrame({'Label': labels, 'Value': values})

# seabornで棒グラフを描画
sns.set(style='whitegrid')
sns.barplot(x='Label', y='Value', data=df)

# グラフの装飾
plt.title('sk398')
plt.ylim(0, 1)  # y軸の範囲を0〜1に固定
plt.ylabel('Formation Rate')
plt.xlabel('Butanol Concentration (v/v%)')
plt.savefig('sk398_barplot_251007.png', dpi=300, bbox_inches='tight')
plt.show()
