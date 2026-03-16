import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from helper import sample_dist

"""
synthetic_var_tables.py

Purpose
-------
Generate LaTeX tables summarizing performance of algorithms on the synthetic
(non–well-separated) VAR-style experiments for a single distribution
(currently: distribution_name = 'normal').

The script:
1) Loads simulation results from ./data/{distribution_name}.csv
2) Filters to the largest sample size N in the CSV
3) Augments the dataset with two Monte Carlo “oracle” benchmarks:
      - 'delta'   : robust min-max cost (minmax_cost)
      - 'optimal' : true (oracle) newsvendor cost (true_cost)
4) For each algorithm (excluding baselines), computes two table metrics:
      - table_additive : additive gap vs the appropriate benchmark
      - table_relative : percent gap vs the appropriate benchmark
   The benchmark is chosen based on an identifiability heuristic using delta:
      - if delta_value > 0.05  => treated as UNidentifiable (identifiable = 0)
          compare minmax_cost to delta (robust benchmark)
      - else                   => treated as identifiable (identifiable = 1)
          compare true_cost to optimal (oracle benchmark)
5) Pivots the table into a LaTeX-ready format with columns indexed by:
      (lam, identifiable, param, metric)
   and rows indexed by algorithm (mapped to LaTeX macros).

Key outputs
-----------
- Printed LaTeX pivot table to stdout.
"""



h = 1
distribution_name = 'normal'

qbar, param_list, b_list, rho_list, lam_list = helper.get_parameters(distribution_name)

algo_list = ['true_saa', 'ignorant_saa', 'robust', 'km', 'subsample_saa', 'joint_order_saa']



print(f'#### TABLES FOR: {distribution_name} ####')
file_loc = f'./data/{distribution_name}.csv'
df = pd.read_csv(file_loc)

# Getting the b values we evaluate
b_values = df['b'].unique()

# Getting the parameter values we evaluate
param_values = df['param'].unique()
lam_list = df['lam'].unique()

print(param_values, lam_list, b_values)


# Augmenting the dataset to include the risk values
delta = []
for lam in lam_list:
    for param in param_values:
        gminus_calc_list = sample_dist(int(1e7), distribution_name, param)
        for b in b_values:
            rho = b / (b + h)
            mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
            mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
            delta.append({'distribution': distribution_name, 'algorithm':'delta', 'param': param, 'b': b, 'h': h, 'lam': lam, 'metric': 'minmax_cost', 'value': mm_cost})

            true_opt = helper.get_optimal_quantile(b, h, gminus_calc_list)
            true_cost = helper.get_newsvendor_cost(true_opt, b, h, gminus_calc_list)
            delta.append({'distribution': distribution_name, 'param': param, 'algorithm':'optimal', 'b': b, 'h': h, 'lam': lam, 'metric': 'true_cost', 'value': true_cost})


df = pd.read_csv(file_loc)
df = df[df['N'] == df['N'].max()] # filtering to the most amount of data
df = pd.concat([df, pd.DataFrame(delta)], ignore_index=True) # including the risk values

df = df[df['lam'].isin(lam_list)] # filtering to the desired lambda value
# df = df[df['lam'] == lam_list[2]]


df = df[df['b'] == 9] # filtering to b = 9

# Separate 'delta' data for easier reference
delta_df = df[df['algorithm'] == 'delta']
optimal_df = df[df['algorithm'] == 'optimal']

# Compute the 'relative' metric and add rows
relative_rows = []

for (algorithm, lam, param), group in df.groupby(['algorithm', 'lam', 'param']):
    if algorithm != 'delta' and algorithm != 'optimal':  # Skip 'delta' and 'optimal' itself
        print(f'Algo: {algorithm} lam: {lam}, param: {param}')

        delta_value = delta_df.loc[(delta_df['lam'] == lam) & (delta_df['param'] == param) & (delta_df['metric'] == 'minmax_cost'), 'value']
        delta_value = delta_value.values[0]

        optimal_value = optimal_df.loc[(optimal_df['lam'] == lam) & (optimal_df['param'] == param) & (optimal_df['metric'] == 'true_cost'), 'value']
        optimal_value = optimal_value.values[0]

        # print(f'Delta value: {delta_value}')

        if delta_value > 0.05: # problem is identifiable, add tolerance due to choice of lambda
            identifiable = 0
            additive_value = group.loc[group['metric'] == 'minmax_cost', 'value'].mean() - delta_value
            relative_value = 100*((group.loc[group['metric'] == 'minmax_cost', 'value'].mean() - delta_value) / delta_value)
        else: # problem is identifiable
            identifiable = 1
            additive_value = group.loc[group['metric'] == 'true_cost', 'value'].mean() - optimal_value
            relative_value = 100*((group.loc[group['metric'] == 'true_cost', 'value'].mean() - optimal_value) / optimal_value)

        relative_rows.append({
            'algorithm': algorithm,
            'lam': lam,
            'param': param,
            'metric': 'table_relative',
            'value': relative_value,
            'identifiable': identifiable
        })


        relative_rows.append({
            'algorithm': algorithm,
            'lam': lam,
            'param': param,
            'metric': 'table_additive',
            'value': additive_value,
            'identifiable': identifiable
        })

df_test = pd.DataFrame(relative_rows)

# Append the new rows to the original dataframe
df_extended = pd.concat([df, pd.DataFrame(relative_rows)], ignore_index=True)


mapping = {
'delta': '$\risk$',
'ignorant_saa': '\IgnorantSAA',
'true_saa': '\TrueSAA',
'joint_order_saa': '\CensoredSAA',
'km': '\KM',
'robust': '\ALG',
'subsample_saa': '\SubsampleSAA'
}

# Rename the column values
df_extended['algorithm'] = df_extended['algorithm'].replace(mapping)


# Create the pivot table based on the specified conditions
filtered_df = df_extended[
    (df_extended['algorithm'] != 'delta') &
    (df_extended['algorithm'] != 'optimal') &
    (df_extended['metric'].isin(['table_additive', 'table_relative']))
]


pivot_table = filtered_df.pivot_table(
    index='algorithm', 
    columns=['lam', 'identifiable', 'param', 'metric'], 
    values='value', 
    aggfunc='mean'
)

# Sort the columns with 'identifiable' decreasing and 'lam' increasing
pivot_table = pivot_table.sort_index(axis=1, level=['lam', 'identifiable', 'param', 'metric'], ascending=[True, False, True, True])


print(pivot_table.to_latex(index=True,
            # float_format="{:.1f}".format,
            float_format=lambda x: helper.float_format_sig(x, sig=2)
))
