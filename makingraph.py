import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# データの準備
values = [0.05, 0.35964912280701755, 0.3237410071942446, 0.47435897435897434]
labels = ['0', '1.0', '1.3', '1.5']  # 任意のラベルに変更可能

# データフレームに変換
df = pd.DataFrame({'Label': labels, 'Value': values})

# seabornで棒グラフを描画
sns.set(style='whitegrid')
sns.barplot(x='Label', y='Value', data=df)

# グラフの装飾
plt.title('pKJE7')
plt.ylim(0, 1)  # y軸の範囲を0〜1に固定
plt.ylabel('Formation rate')
plt.xlabel('Butanol concentration')
plt.savefig('398_barplot_250724.png', dpi=300, bbox_inches='tight')
plt.show()
