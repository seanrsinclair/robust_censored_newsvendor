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

Creates two figures comparing q^\Delta and \Delta versus the observable boundary \lambda

'''



# Set up the plot style
sns.set_style("white")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')


# Gets the data to evaluate the expectations
eval_data = np.random.exponential(scale=80, size=int(1e7))

# List of rho values to check when h = 1
rho_list = [0.1, 0.3, 0.5, 0.7, 0.9]
b_list = [(-1) * rho / (rho - 1) for rho in rho_list]
print(b_list)
h = 1
# rho_list = [b / (b+h) for b in b_list]
qbar = 200

# NOTE: Uncomment the following in order to get the data that is used to feed into the plotting code

# lam_list = np.linspace(0, qbar, 300)
# data = []

# for lam in lam_list:
#     for b in b_list:
#         rho = b / (b+h)
#         qcrit = helper.get_optimal_robust(b, h, lam, qbar, eval_data)
#         cost = helper.get_robust_cost(qcrit, b, h, lam, qbar, eval_data)

#         data.append({'b': h, 'h': h, 'rho': rho, 'lam': lam, 'qrisk': qcrit, 'cost': cost})


# df = pd.DataFrame(data)
# print(df.head(5))

# df.to_csv(f'./data/delta_qcrit.csv', index=False)

# Loops through the b values and gets the cusps, being \qopt
rho_cusp_list = []
for b in b_list:
    rho = b / (b+h)
    cusp = helper.get_optimal_quantile(b, h, eval_data)
    rho_cusp_list.append(cusp)
print(rho_cusp_list)

# Loops through the b values to get the qcrit values:
# q_crit_cusp_list = []
# for b in b_list:
#     rho = b / (b+h)
#     qcrit = helper.get_q_crit( , lam, )

# FIGURE ONE: q^\Delta


# Generate tick labels in the desired format
tick_labels = [rf'$q^\star_{{{2*i+1}}}$' for i in range(len(rho_cusp_list))]


df = pd.read_csv(f'./data/delta_qcrit.csv')

# Create the seaborn line plot with x-axis 'q', y-axis 'cost', and color-coded by 'rho'
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='lam', y='qrisk', hue='rho')

# Customize the plot
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$q^{\Delta}$')

# Add thin, dotted vertical lines at each rho_cusp_list value
for cusp in rho_cusp_list:
    plt.axvline(x=cusp, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=cusp, color='gray', linestyle='--', linewidth=0.5)


# Set x-ticks at specified positions
plt.xticks(ticks=rho_cusp_list, labels=tick_labels)
plt.yticks(ticks=rho_cusp_list, labels=tick_labels)  # Keep y-ticks removed as requested




# Update legend title to display as LaTeX notation for rho
# Remove the legend
plt.legend([], [], frameon=False)

# saves the figure
plt.savefig(f'./figures/qrisk.pdf', bbox_inches = 'tight')

# FIGURE TWO: \Delta


# Create the seaborn line plot with x-axis 'q', y-axis 'cost', and color-coded by 'rho'
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='lam', y='cost', hue='rho')

# Set x-ticks at specified positions
plt.xticks(ticks=rho_cusp_list, labels=tick_labels)
plt.yticks([])  # Keep y-ticks removed as requested

# Add a horizontal line at y = 0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)


# Add thin, dotted vertical lines at each rho_cusp_list value
for cusp in rho_cusp_list:
    plt.axvline(x=cusp, color='gray', linestyle='--', linewidth=0.5)


# Customize the plot
plt.xlabel(r'$\lambda$')
plt.ylabel('$\Delta$')
plt.legend(title=r'$\rho$')

plt.savefig(f'./figures/delta.pdf', bbox_inches = 'tight')