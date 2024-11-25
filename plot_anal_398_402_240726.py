import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr


meds_c = []
with open('sk25_pro_FITC_c_meds.txt') as f:
    for i in f.readlines():
        meds_c.append(float(i.split(',')[0]))
meds_5 = []
with open('sk402_005_bta18_meds.txt') as f:
    for i in f.readlines():
        meds_5.append(float(i.split(',')[0]))
meds_6 = []
with open('sk402_01_bta18_meds.txt') as f:
    for i in f.readlines():
        meds_6.append(float(i.split(',')[0]))
meds_7 = []
with open('sk402_bta18_meds.txt') as f:
    for i in f.readlines():
        meds_7.append(float(i.split(',')[0]))
meds_8 = []
with open('sk398_bta18_meds.txt') as f:
    for i in f.readlines():
        meds_8.append(float(i.split(',')[0]))
# meds_10 = []
# with open('sk25_pro_FITC_8_meds.txt') as f:
#     for i in f.readlines():
#         meds_10.append(float(i.split(',')[0]))

data1,data2,data3,data4,data5= meds_c,meds_5,meds_6,meds_7,meds_8
sns.set()
fig = plt.figure(figsize = [8,8])
plt.boxplot([data1,data2,data3,data4,data5],sym="")
for i, data in enumerate([data1,data2,data3,data4,data5], start=1):
    x = np.random.normal(i, 0.04, size=len(data))  
    plt.plot(x, data, 'o', alpha=0.1)
plt.xticks([1, 2, 3, 4,5], [f'cntr. (n={len(data1)})', f'4.0% (n={len(data2)})', f'5.0% (n={len(data3)})', f'6.0% (n={len(data4)})', f'7.0% (n={len(data5)})'])
plt.xlabel('butanol conc.(%)')
plt.ylabel('meds(-)')
plt.grid(True)
fig.savefig("fig_bta_meds_240713.png",dpi = 500)

data1,data2,data3,data4,data5= meds_c,meds_5,meds_6,meds_7,meds_8
lower_percentile_95 = np.percentile(data1, 5)
prob_below_lower_percentile_95 = [0.05] + [float(np.mean(data < lower_percentile_95)) for data in [data2, data3, data4, data5]]
prob_below_lower_percentile_95_100 = [x * 100 for x in prob_below_lower_percentile_95]
print(f"下限95パーセンタイルを下回るサンプルの発生確率: {prob_below_lower_percentile_95_100}")
def multiple_by_100(data):
    return[y * 100 for y in data]
x = [0, 5, 6, 7, 8, 10]
y = prob_below_lower_percentile_95_100
sns.set()
fig, ax = plt.subplots()
sns.lineplot(x=x, y=y, marker='o', linestyle='-', ax=ax)
ax.grid(True)
ax.set_xlabel('butanol conc.(%)')
ax.set_ylabel('formation rate(%)')
ax.set_ylim(-5, 100)
fig.savefig('fig_bta_95_240713.png')

# x_input = prob_below_lower_percentile_95_100
# y_input = [0, 13.04347826, 23.75, 40, 97.8021978, 100]
# fig, ax =plt.subplots(figsize=(6,6))
# ax.scatter(x_input[0],y_input[0],color = 'cyan',label = '0%')
# ax.scatter(x_input[1],y_input[1],color = 'deepskyblue',label = '4.0%')
# ax.scatter(x_input[2],y_input[2],color = 'dodgerblue',label = '5.0%')
# ax.scatter(x_input[3],y_input[3],color = 'mediumslateblue',label = '6.0%')
# ax.scatter(x_input[4],y_input[4],color = 'slateblue',label = '7.0%')
# ax.scatter(x_input[5],y_input[5],color = 'darkviolet',label = '8.0%')  
# sns.set()
# ax.grid(True)
# ax.set_xlabel('formation rate(%)')
# ax.set_ylabel('death cell rate(%)')
# ax.set_xlim(-5,105)
# ax.set_ylim(-5,105)

# correlation_coefficient, _ = pearsonr(x_input, y_input)
# fit_params = np.polyfit(x_input, y_input, 1)
# fit_equation = f'f(x) = {fit_params[0]:.4f} * x + {fit_params[1]:.4f}'
# fit_equation_f = f'{fit_equation}\n(R²= {correlation_coefficient:.4f})'

# ax.plot(x_input, np.polyval(fit_params, x_input), color='black', linewidth=0.7, label=fit_equation_f)
# ax.legend(loc='lower right', fontsize = '8')
# fig.savefig('fig_pro_cor_240130.png')

plt.show()