import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

def multiply_by_100(data):
    return [x * 100 for x in data]

sns.set()

c_black = multiply_by_100([0.05, 0.43386243386243384, 0.3333333333333333, 0.7692307692307693, 0.8853503184713376, 0.9390862944162437])
f_black = multiply_by_100([0, 0.623655914, 0.87755102, 0.942528736, 1, 1])

c_red = multiply_by_100([0.05, 0.7298850574712644, 0.5073529411764706, 0.6861702127659575, 0.9615384615384616, 0.8863636363636364])
f_red = multiply_by_100([0, 0.722772277, 0.831858407, 1, 1, 1])

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(c_black, f_black, color='deepskyblue', label='SK398_empty vector')
ax.scatter(c_red, f_red, color='darkviolet', label='SK394_GroESL')

ax.grid(True)
ax.set_xlabel('formation rate(%)')
ax.set_ylabel('dead cell rate(%)')
ax.set_ylim(-5, 105)
ax.set_xlim(-5, 105)

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
fig.savefig("fig_bta_cor_394398__240314.png")

plt.show()
