import numpy as np
import pandas as pd


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms
import helper

import freshnet_helper

import matplotlib.pyplot as plt
import seaborn as sns

'''

Computes combined histogram plot for different products

'''

plt.style.use('PaperDoubleFig.mplstyle.txt')

plt.rc('text', usetex=True)
# Add amsmath to the LaTeX preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'



num_samples = 100

# starting from 70 is the old list!
product_list = [74, 803, 239, 832, 70, 215]



samples_dict = {}

for product_id in product_list:
    qbar_list, b_list, rho_list, order_levels, censored_demands = freshnet_helper.get_parameters(product_id) # gets the evaluation parameters


    lam = max(order_levels)


    cdf = freshnet_helper.get_km_cdf(product_id, qbar = qbar_list[1], dataset='eval')
    # eval_order_levels, _ = freshnet_helper.get_censored_sales_data(product_id, dataset='eval')
    print(f'Product ID: {product_id} and value at 5: {cdf(5)}')

    # Evaluate the CDF directly on a grid
    grid_size = 2000
    # x_vals = np.linspace(0, max(eval_order_levels), grid_size)
    x_vals = np.linspace(0, qbar_list[1], grid_size)
    y_vals = np.array([cdf(x) for x in x_vals])

    samples_dict[product_id] = {'x': x_vals, 'y': y_vals}

# --- Overlaid CDFs for all products ---
plt.figure(figsize=(10,6))
sns.set_palette('husl')
palette = sns.color_palette('husl', n_colors=len(samples_dict))

all_x = np.concatenate([vals['x'] for vals in samples_dict.values()])
xmin = np.min(all_x)
xmax = np.max(all_x)

for (product_id, vals), color in zip(samples_dict.items(), palette):
    x = vals['x']
    y = vals['y']

    # Main CDF curve
    plt.plot(x, y, color=color)

    # # Horizontal line at 0 from xmin to the first x-value
    # plt.hlines(0, xmin, x[0], colors=color)

    # # Horizontal line at 1 from the last x-value to xmax
    # plt.hlines(1, x[-1], xmax, colors=color)

# plt.xlim(xmin, xmax)
plt.xlim(0, 15)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.xlabel(r'$q$')
plt.ylabel(r'$G(q)$')
plt.savefig('./figures/multi_product_cdf.pdf', bbox_inches='tight', pad_inches=0)
plt.close()
