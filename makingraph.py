import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# データの準備
values = [0.05, 0.2430939226519337, 0.22857142857142856, 0.8066298342541437]
labels = ['0', '0.7', '1.0', '1.5']  # 任意のラベルに変更可能

# データフレームに変換
df = pd.DataFrame({'Label': labels, 'Value': values})

# seabornで棒グラフを描画
sns.set(style='whitegrid')
sns.barplot(x='Label', y='Value', data=df)

# グラフの装飾
plt.title('pKJE7')
plt.ylim(0, 1)  # y軸の範囲を0〜1に固定
plt.ylabel('value')
plt.xlabel('Label')
plt.savefig('398_barplot_250715.png', dpi=300, bbox_inches='tight')
plt.show()
