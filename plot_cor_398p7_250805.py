import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

def multiply_by_100(data):
    return [x * 100 for x in data]

sns.set()

c_black = multiply_by_100([0.05, 0.59375, 0.7961165048543689, 1.0, 0.9640718562874252])
f_black = multiply_by_100([0, 0.348101266, 1, 1, 1])

c_red = multiply_by_100([0.05, 0.3620178041543027, 0.46875, 0.7592592592592593, 0.5056179775280899])
f_red = multiply_by_100([0, 0.173611111, 0.634920635, 1, 1])

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(c_black, f_black, color='deepskyblue', label='SK398_empty vector')
ax.scatter(c_red, f_red, color='darkviolet', label='pKJE7_DnaKJ_grpE')

ax.grid(True)
ax.set_xlabel('formation rate(%)')
ax.set_ylabel('dead cell rate(%)')
ax.set_ylim(-5, 105)
ax.set_xlim(-5, 105)
ax.set_title('Correlation diagram')

slope_black, _ = np.polyfit(c_black, f_black, 1)
line_of_best_fit = slope_black * np.array(c_black)
correlation_coefficient, _ = pearsonr(c_black, f_black)
fit_equation = f'f(x) = {slope_black:.4f} * x'
fit_equation_f = f'{fit_equation}\n(R²= {correlation_coefficient:.4f})'
# ax.plot(c_black, line_of_best_fit, color='cyan', linewidth=0, label=fit_equation_f)

slope_red, _ = np.polyfit(c_red, f_red, 1)
line_of_best_fit = slope_red * np.array(c_red)
correlation_coefficient, _ = pearsonr(c_red, f_red)
fit_equation = f'f(x) = {slope_red:.4f} * x'
fit_equation_f = f'{fit_equation}\n(R²= {correlation_coefficient:.4f})'
# ax.plot(c_red, line_of_best_fit, color='darkviolet', linewidth=0, label=fit_equation_f)

x = np.linspace(0, 100, 100)
ax.plot(x, x, color='black', linestyle='--', linewidth=0.8, label='x = y')

ax.legend(loc='lower right', fontsize='8')

# 保存
fig.savefig("fig_bta_cor_398p7_250805.png")

plt.show()
