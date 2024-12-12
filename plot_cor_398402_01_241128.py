import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

def multiply_by_100(data):
    return [x * 100 for x in data]

sns.set()

c_black = multiply_by_100([0.05, 0.04, 0.3712871287128713, 0.6384180790960452, 0.8646288209606987, 0.8313725490196079])
f_black = multiply_by_100([0, 0.109489051, 0.288659794, 0.852348993, 0.9921875, 1])

c_red = multiply_by_100([0.05, 0.07083333333333333, 0.7593360995850622, 0.7876106194690266, 0.6636771300448431, 0.5061224489795918])
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
fig.savefig("fig_bta_cor_398402_01_241128.png")

plt.show()
