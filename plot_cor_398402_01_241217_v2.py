import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

def multiply_by_100(data):
    return [x * 100 for x in data]

sns.set()

c_black = multiply_by_100([0.05, 0.05042016806722689, 0.0625, 0.1419753086419753, 0.42990654205607476, 0.38028169014084506])
f_black = multiply_by_100([0, 0.109489051, 0.288659794, 0.852348993, 0.9921875, 1])

c_red = multiply_by_100([0.05, 0.1686746987951807, 0.2777777777777778, 0.3391304347826087, 0.4935064935064935, 0.5163398692810458])
f_red = multiply_by_100([0.04950495, 0.223404255, 0.984924623, 1, 1, 1])

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(c_black, f_black, color='deepskyblue', label='SK398_empty vector')
ax.scatter(c_red, f_red, color='darkviolet', label='SK402_GroESL')

ax.grid(True)
ax.set_xlabel('formation rate(%)')
ax.set_ylabel('dead cell rate(%)')
ax.set_ylim(-5, 105)
ax.set_xlim(-5, 105)
ax.set_title('IPTG 0.1mM')

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
fig.savefig("fig_bta_cor_398402_01_241217_v2.png")

plt.show()
