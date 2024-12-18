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
qbar = 25
table_b = 9

algo_list = ['true_saa', 'ignorant_saa', 'robust', 'robust_plus', 'km', 'subsample_saa', 'joint_order_saa']

demand_df = pd.read_csv('./datasets/train.csv')
grouped_df = demand_df.groupby(['Order Date', 'Category']).size().reset_index(name='sales')


def sample_dist(N, grouped_df, category):
    filtered_df = grouped_df[grouped_df['Category'] == category]

    if N > len(filtered_df):
        raise ValueError("Sample size cannot be greater than the number of available rows.")
    return filtered_df['sales'].sample(n=N, replace=False).reset_index(drop=True)


for category in grouped_df['Category'].unique():

    print(f'###### CATEGORY: {category} ######')
    distribution_name = f'kaggle_sales_data_{category}'

    gminus_calc_list = grouped_df[grouped_df['Category'] == category]['sales'].to_numpy()


    file_loc = f'./data/{distribution_name}_plus.csv'
    print(f'Reading data from: {file_loc}')
    df = pd.read_csv(file_loc)
    print(f'Data frame distribution name: {df["distribution"].unique()}')
    b_values = df['b'].unique()
    # print(b_values)
    lam_values = df['lam'].unique()


    delta = []

    # AUGMENTING TO INCLUDE DELTA VALUES AS AN ``ALGORITHM''
    # b_values = df['b'].unique()
    b_values = [9]
    lam_values = df['lam'].unique()

    for b in b_values:
        rho = b / (b + h)
        # Calculating \Delta for each \lambda value in the plot
        for lam in lam_values:
            mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
            mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
            delta.append({'distribution': distribution_name, 'algorithm':'delta', 'b': b, 'h': h, 'lam': lam, 'metric': 'minmax_cost', 'value': mm_cost})

            true_opt = helper.get_optimal_quantile(b, h, gminus_calc_list)
            true_cost = helper.get_newsvendor_cost(true_opt, b, h, gminus_calc_list)
            delta.append({'distribution': distribution_name, 'algorithm':'optimal', 'b': b, 'h': h, 'lam': lam, 'metric': 'true_cost', 'value': true_cost})

        print(f'Category: {category}, b: {b}, opt: {true_opt}')

    lam = lam_values[0]

    for table_b in b_values:
        print(f'### TABLE FOR B: {table_b} ###')
        df = pd.read_csv(file_loc)
        N = df['N'].max()
        df = df[df['N'] == 500]

        df = pd.concat([df, pd.DataFrame(delta)], ignore_index=True)
        
        # Filter by b value equal to 9
        df = df[df['b'] == table_b]




        # Group by 'algorithm', 'lam', and 'metric', and calculate the average of 'value'
        averaged_df = df.groupby(['algorithm', 'order_level_two', 'metric'])['value'].mean().reset_index()
        print(averaged_df.head(5))

        # Separate 'delta' data for easier reference
        delta_df = df[df['algorithm'] == 'delta']
        optimal_df = df[df['algorithm'] == 'optimal']

        # Compute the 'relative' metric and add rows
        relative_rows = []

        for (algorithm, q), group in df.groupby(['algorithm', 'order_level_two']):
            if algorithm != 'delta' and algorithm != 'optimal':  # Skip 'delta' and 'optimal' itself
                delta_value = delta_df.loc[(delta_df['lam'] == lam) & (delta_df['metric'] == 'minmax_cost'), 'value']
                delta_value = delta_value.values[0]

                optimal_value = optimal_df.loc[(optimal_df['lam'] == lam) & (optimal_df['metric'] == 'true_cost'), 'value']

                if delta_value > 0: # problem is identifiable
                    identifiable = 0
                    relative_value = 100*((group.loc[group['metric'] == 'minmax_cost', 'value'].mean() - delta_value) / delta_value)
                else:
                    identifiable = 1
                    relative_value = 100*((group.loc[group['metric'] == 'true_cost', 'value'].mean() - optimal_value) / optimal_value)

                relative_rows.append({
                    'algorithm': algorithm,
                    'lam': lam,
                    'order_level_two': q,
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
        'subsample_saa': '\SubsampleSAA',
        'robust_plus': '\ALGplus',
        # 'robust_bonus': '\ALG^{b}'
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
            columns=['identifiable', 'order_level_two'], 
            values='value', 
            aggfunc='mean'
        )

        # Sort the columns with 'identifiable' decreasing and 'lam' increasing
        pivot_table = pivot_table.sort_index(axis=1, level=['identifiable', 'order_level_two'], ascending=[True, True])

        print(pivot_table.to_latex(index=True,
                    float_format="{:.1f}".format,
        ))

