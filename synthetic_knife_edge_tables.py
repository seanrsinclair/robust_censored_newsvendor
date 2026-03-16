import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from helper import sample_dist

"""
synthetic_vanilla_lam_tables.py

Purpose
-------
Create LaTeX tables for the paper summarizing algorithm performance on the
"vanilla" regret for the distributions (continuous_uniform, exponential, poisson).
This script focuses on a fixed underage cost level (currently b=9) and the
largest available sample size N in the simulation output CSVs.

For each distribution in distribution_list, the script:
1) Loads the corresponding CSV containing simulation outputs.
2) Computes Monte Carlo benchmark baselines per (param, lam):
     - 'delta'   baseline: robust min-max cost (metric='minmax_cost')
     - 'optimal' baseline: oracle true newsvendor cost (metric='true_cost')
   and appends these baselines to the dataset as extra "algorithm" rows.
3) Computes two paper-facing performance metrics for each algorithm:
     - table_additive : additive gap vs oracle (currently always oracle)
     - table_relative : percent gap vs oracle (currently always oracle)
   producing a pivoted LaTeX table with rows = algorithm and
   columns = (param, lam, metric).

Key outputs
-----------
- Printed LaTeX table to stdout for each distribution (and each b in b_values).
"""


PLOT_REGRET = True
h = 1
np.random.seed(1)

distribution_list = ['uniform', 'exponential', 'poisson']

algo_list = ['true_saa', 'ignorant_saa', 'robust', 'km', 'subsample_saa', 'joint_order_saa']


# Loop through each distribution name and read the CSV file
for distribution_name in distribution_list:

    print(f'Distribution: {distribution_name}')

    qbar, param_list, b_list, rho_list, lam_list = helper.get_parameters(distribution_name)
    if distribution_name == 'uniform':
        qbar = 100
    qopt = helper.get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
    print(f'qopt value: {qopt}')
    lam_list = np.linspace(0.9*qopt, 1.1*qopt, 8)


    file_loc = f'./data/{distribution_name}_knife_edge.csv'
    df = pd.read_csv(file_loc)



    print(f'Lambda values: {df["lam"].unique()}')
    print(f'b values: {df["b"].unique()}')




    delta = []

    # AUGMENTING TO INCLUDE DELTA VALUES AS AN ``ALGORITHM''
    param_values = df['param'].unique()
    b_values = df['b'].unique()
    lam_values = df['lam'].unique()
    N_values = df['N'].unique()


    for param in param_values:
        if distribution_name == 'negative_binomial':
            param = eval(param) # converts into a proper tuple
        gminus_calc_list = sample_dist(int(1e7), distribution_name, param)

        for b in b_values:
            rho = b / (b + h)
            # Calculating \Delta for each \lambda value in the plot
            for lam in lam_values:
                rho = b / (b + h)
                mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
                mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
                delta.append({'distribution': distribution_name, 'param': param, 'algorithm':'delta', 'b': b, 'h': h, 'lam': lam, 'metric': 'minmax_cost', 'value': mm_cost})

                true_opt = helper.get_optimal_quantile(b, h, gminus_calc_list)
                true_cost = helper.get_newsvendor_cost(true_opt, b, h, gminus_calc_list)
                delta.append({'distribution': distribution_name, 'param': param, 'algorithm':'optimal', 'b': b, 'h': h, 'lam': lam, 'metric': 'true_cost', 'value': true_cost})


                if mm_cost > 0:
                    delta.append({'distribution': distribution_name, 'param': param, 'algorithm':'true value', 'b': b, 'h': h, 'lam': lam, 'metric': 'relative', 'value': mm_cost})
                else:
                    delta.append({'distribution': distribution_name, 'param': param, 'algorithm':'true value', 'b': b, 'h': h, 'lam': lam, 'metric': 'relative', 'value': true_cost})


    for b in b_values:
        print(f'### FOR VALUE: {b} ###')

        df = pd.read_csv(file_loc)


        for N in N_values:
            print(f'Computing for N: {N}')
            df = pd.read_csv(file_loc)
            df = df[df['N'] == N] # filtering to the specified value of N
            df = pd.concat([df, pd.DataFrame(delta)], ignore_index=True) # including the risk values
            df = df[df['lam'].isin(lam_values)] # filtering to the desired lambda value

            df = df[df['b'] == b] # filtering to b


            # Group by 'algorithm', 'lam', 'param', and 'metric', and calculate the average of 'value'

            df = df[df['metric'].isin(['true_cost', 'minmax_cost'])] # filters out identifiability metric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            averaged_df = df.groupby(['algorithm', 'lam', 'param', 'metric'])['value'].mean().reset_index()

            # Separate 'delta' data for easier reference
            delta_df = df[df['algorithm'] == 'delta']
            optimal_df = df[df['algorithm'] == 'optimal']

            # Compute the 'relative' metric and add rows
            relative_rows = []

            for (algorithm, lam, param), group in df.groupby(['algorithm', 'lam', 'param']):
                if algorithm != 'delta' and algorithm != 'optimal' and algorithm != 'true value':  # Skip 'delta' and 'optimal' and 'true value' itself
                    delta_value = delta_df.loc[(delta_df['lam'] == lam) & (delta_df['param'] == param) & (delta_df['metric'] == 'minmax_cost'), 'value']
                    delta_value = delta_value.values[0]

                    optimal_value = optimal_df.loc[(optimal_df['lam'] == lam) & (optimal_df['param'] == param) & (optimal_df['metric'] == 'true_cost'), 'value']


                    if delta_value > 0: # problem is unidentifiable
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

            # Append the new rows to the original dataframe
            df_extended = pd.concat([df, pd.DataFrame(relative_rows)], ignore_index=True)


            mapping = {
            'delta': '\risk',
            'ignorant_saa': '\IgnorantSAA',
            'true_saa': '\TrueSAA',
            'joint_order_saa': '\CensoredSAA',
            'km': '\KM',
            'robust': '\ALG',
            'subsample_saa': '\SubsampleSAA'
            }

            # Rename the column values
            df_extended['algorithm'] = df_extended['algorithm'].replace(mapping)


            
            filtered_df = df_extended[
                (df_extended['algorithm'] != 'delta') &
                (df_extended['algorithm'] != 'optimal') &
                # (df_extended['metric'] == 'minmax_cost')
                (df_extended['metric'].isin(['table_additive', 'table_relative']))
            ]

            print(filtered_df.head(5))

            pivot_table = filtered_df.pivot_table(
                index='algorithm', 
                columns=['param', 'lam', 'metric'], 
                values='value', 
                aggfunc='mean'
            )

            # Sort the columns with 'identifiable' decreasing and 'lam' increasing
            pivot_table = pivot_table.sort_index(axis=1, level=['param', 'lam', 'metric'], ascending=[True, True, True])


            print(pivot_table.to_latex(index=True,
                        # float_format="{:.1f}".format,
                        float_format=lambda x: helper.float_format_sig(x, sig=2)
            ))

