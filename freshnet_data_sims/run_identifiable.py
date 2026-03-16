import numpy as np
import pandas as pd


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms
import helper

import freshnet_helper

import matplotlib.pyplot as plt

"""
generate_identifiable_products.py

Purpose
-------
Construct a dataset that flags which FreshRetailNet products are *identifiable*
(as defined by a G^-(lambda) vs rho condition) across different qbar levels, and
save the resulting product × qbar summary for downstream analysis/plots.

What it does
------------
For each product and each candidate qbar:
- Load the product’s observed order levels and censored demands.
- Set lam := max(observed order levels) for that product.
- Estimate Gminus(lam) = P(D < lam) using a large sample drawn from the product’s
  KM-based demand model (evaluation dataset).
- For each b (or each rho = b/(b+h)), mark identifiability:
      identifiable = 1{ Gminus(lam) >= rho }
- Save a tidy table with (at least) product, qbar, lam, b/rho, Gminus(lam),
  and the resulting identifiable flag.

Output
------
Writes a CSV (long or product-summary format depending on implementation) that
can be joined with algorithm results to filter to “identifiable products” at
each qbar level.
"""


DEBUG = False
INCLUDE_TRUE = True

# Meta parameter specification for the simulations

df = pd.read_csv('./data/product_data.csv')
product_list = df['product'].unique()


h = 1

np.random.seed(5)

# Loops over each distribution



data = []


for product_id in product_list:
    print(f'#### RUNNING FOR: {product_id} ####')
 

    file_name = f'./data/freshnet_identifiable.csv'

    qbar_list, b_list, rho_list, order_levels, censored_demands = freshnet_helper.get_parameters(product_id) # gets the evaluation parameters

    lam = max(order_levels)

    # specified for that distribution

    for qbar in qbar_list: # Loop over each parameter

        gminus_calc_list = freshnet_helper.sample_from_km_cdf(num_samples = int(1e7), qbar = qbar, product_id = product_id, dataset='eval')

        for b in b_list:

            rho = b / (b+h)
            print(f'############# Running for product: {product_id} b: {b}, lam: {lam} and qbar: {qbar} #############')
            id = 1 if np.mean(gminus_calc_list < lam) >= rho else 0

            qopt = helper.get_optimal_quantile(b, h, gminus_calc_list)
            opt_cost = helper.get_newsvendor_cost(qopt, b, h, gminus_calc_list)

            mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
            mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)

            common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar}


            data.append({
            'metric': 'identifiable',
            'value': id,
            **common_params
            })

            data.append({
            'metric': 'qopt',
            'value': qopt,
            **common_params
            })

            data.append({
            'metric': 'qdelta',
            'value': mm_opt,
            **common_params
            })



# # Convert list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# # Save the DataFrame as a CSV file
df.to_csv(file_name, index=False)  # index=False to avoid writing row numbers

print(f"Data has been saved to {file_name}")
