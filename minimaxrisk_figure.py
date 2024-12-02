import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import algorithms
import helper
import matplotlib

from helper import get_robust_cost

import pandas as pd
#Back-end to use depends on the system
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

'''

Creating a figure comparing Regret against the order level

'''

# Set up the plot style

sns.set_style("white")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')

# Data to evaluate the expectations
eval_data = np.random.exponential(scale=80, size=int(1e7))


# List of rho values to compare
rho_list = [0.2, 0.3, 0.4, 0.5, 0.6]

# Commensurate b values when h = 1
b_list = [(-1) * rho / (rho - 1) for rho in rho_list]
print(b_list)
h = 1


qbar = 200
lam = np.quantile(eval_data, 0.4)
q_list = np.linspace(0, qbar, 300) # spacing of order levels q to check


# NOTE: Uncomment the following code to generate a dataset that will be plotted

# data = []

# for q in q_list:
#     for b in b_list:
#         rho = b / (b+h)
#         # print(f'Rho: {rho}')
#         cost = get_robust_cost(q, b, h, lam, qbar, eval_data) # gets the regret for order level q
#         data.append({'b': h, 'h': h, 'rho': rho, 'q': q, 'cost': cost})

# df = pd.DataFrame(data)
# df.to_csv(f'./data/minimaxrisk.csv', index=False)

tick_list = []
label_list = []

tick_list.append(lam)
label_list.append(r'$\lambda$')


# Checks each of the rho values and checks if it is identifiable or unidentifiable
# and adds a tick mark for either \qopt or q^\dagger depending on the regime
i = 2

y_tick_list = []
y_label_list = []
for b in b_list:
    rho = b / (b+h)
    if rho == 0.4: # this is the observable boundary, do not plot
        i += 1
        continue
    else:
        Gminus = np.mean(eval_data < lam)
        if Gminus >= rho:
            print(f'rho: {rho} identifiable')
            qopt = helper.get_optimal_quantile(b, h, eval_data)
            tick_list.append(qopt)
            label_list.append(rf'$q^\star_{int(i)}')
        if Gminus < rho:
            print(f'rho: {rho} unidentifiable')
            qcrit = helper.get_optimal_robust(b, h, lam, qbar, eval_data)
            delta = helper.get_robust_cost(qcrit, b, h, lam, qbar, eval_data)
            print(f'qcrit: {qcrit}')
            tick_list.append(qcrit)
            label_list.append(rf'$q^\dag_{int(i)}$')
            y_tick_list.append(delta)
            y_label_list.append(rf'$\Delta_{int(i)}$')
        i += 1


# Reads in the saved data
df = pd.read_csv(f'./data/minimaxrisk.csv')
df['rho'] = df['rho'].round(1)

print(f'rho list: {rho_list}')
print(f'df rho values: {df["rho"].unique()}')
epsilon = 1e-5  # Define a small tolerance to round rho for numerical precision
df = df[df['rho'].apply(lambda x: any(np.isclose(x, rho, atol=epsilon) for rho in rho_list))]
df = df[df['q'] <= 100]

# Create the seaborn line plot with x-axis 'q', y-axis 'cost', and color-coded by 'rho'
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='q', y='cost', hue='rho')


for cusp in tick_list: # adds the cusp to the figure
    plt.axvline(x=cusp, color='gray', linestyle='--', linewidth=0.5)

for cusp in y_tick_list:
    plt.axhline(y = cusp, color='gray', linestyle='--', linewidth=0.5)

# Add a horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)


plt.xticks(ticks=tick_list, labels=label_list)
plt.yticks(ticks=y_tick_list, labels=y_label_list)  # Keep y-ticks removed as requested

# Customize the plot
plt.xlabel(r'$q$')
plt.ylabel(r'$\textup{Regret}(q)$')


plt.legend(
    title=r'$\rho$', 
    loc='upper left', 
    bbox_to_anchor=(1, 1),  # Place legend outside the plot
    frameon=False  # Remove legend frame if preferred
)

# saves the figure
plt.savefig(f'./figures/minimaxrisk.pdf', bbox_inches = 'tight')

plt.close('all')