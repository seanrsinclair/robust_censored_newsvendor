import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from helper import sample_dist

PLOT_REGRET = True
h = 1
np.random.seed(1)

distribution_list = ['uniform', 'exponential', 'poisson']


algo_list = ['true_saa', 'ignorant_saa', 'robust', 'km', 'subsample_saa', 'joint_order_saa']


# Loop through each distribution name and read the CSV file
for distribution_name in distribution_list:



    print(f'Distribution: {distribution_name}')

    qbar, param_list, b_list, rho_list, lam_list = helper.get_parameters(distribution_name)

    file_loc = f'./data/{distribution_name}_full_b.csv'
    df = pd.read_csv(file_loc)
    print(f'Lambda values: {df["lam"].unique()}')
    print(f'b values: {df["b"].unique()}')




    delta = []

    # AUGMENTING TO INCLUDE DELTA VALUES AS AN ``ALGORITHM''
    param_values = df['param'].unique()
    b_values = df['b'].unique()
    lam_values = df['lam'].unique()

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
        df = df[df['N'] == df['N'].max()] # filtering to the most amount of data
        df = pd.concat([df, pd.DataFrame(delta)], ignore_index=True) # including the risk values
        df = df[df['lam'].isin(lam_values)] # filtering to the desired lambda value

        df = df[df['b'] == b] # filtering to b = 9



        # Group by 'algorithm', 'lam', 'param', and 'metric', and calculate the average of 'value'
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


                if delta_value > 0: # problem is identifiable
                    identifiable = 0
                    relative_value = 100*((group.loc[group['metric'] == 'minmax_cost', 'value'].mean() - delta_value) / delta_value)
                else:
                    identifiable = 1
                    relative_value = 100*((group.loc[group['metric'] == 'true_cost', 'value'].mean() - optimal_value) / optimal_value)

                relative_rows.append({
                    'algorithm': algorithm,
                    'lam': lam,
                    'param': param,
                    'metric': 'relative',
                    'value': relative_value,
                    'identifiable': identifiable
                })

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
            (df_extended['metric'] == 'relative')
        ]
        

        pivot_table = filtered_df.pivot_table(
            index='algorithm', 
            columns=['param', 'identifiable', 'lam'], 
            values='value', 
            aggfunc='mean'
        )

        # Sort the columns with 'identifiable' decreasing and 'lam' increasing
        pivot_table = pivot_table.sort_index(axis=1, level=['param', 'identifiable', 'lam'], ascending=[True, True, True])


        print(pivot_table.to_latex(index=True,
                    float_format="{:.2f}".format,
        ))



