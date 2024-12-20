import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

def multiply_by_100(data):
    return [x * 100 for x in data]

sns.set()

c_black = multiply_by_100([0.05, 0.17488789237668162, 0.2198581560283688, 0.6566265060240963, 0.9758064516129032, 0.933852140077821])
f_black = multiply_by_100([0,	0.094420601, 0.12371134, 0.2, 0.975409836, 1])

c_red = multiply_by_100([0.05, 0.05357142857142857, 0.5726495726495726, 0.9689655172413794, 0.9678714859437751, 0.9482071713147411])
f_red = multiply_by_100([0.013043478, 0.067708333, 0.592039801, 0.994413408, 1, 1])

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(c_black, f_black, color='black', label='SK398_empty vector', s=90)
ax.scatter(c_red, f_red, color='red', label='SK402_GroESL', s=90)

ax.grid(True)
ax.set_xlabel('Formation Rate(%)')
ax.set_ylabel('Death Rate(%)')
ax.set_ylim(-5, 105)
ax.set_xlim(-5, 105)
ax.set_title('IPTG 0.05mM')

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

ax.legend(loc='lower right', fontsize='12')

# 保存
fig.savefig("fig_bta_cor_398402_005_241128.png")

plt.show()
