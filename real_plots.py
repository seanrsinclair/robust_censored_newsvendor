import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress



PLOT_REGRET = True

h = 1
qbar = 10



plt.style.use('PaperDoubleFig.mplstyle.txt')

plt.style.use('PaperDoubleFig.mplstyle.txt')

algo_list = ['true_saa', 'ignorant_saa', 'robust', 'km', 'subsample_saa', 'joint_order_saa']


sns.set_palette('husl', len(algo_list))


# Line styles dictionary
line_styles = {
    'ignorant_saa': 'dotted',
    'robust': 'dashed',
    'km': 'dashdot',
    'subsample_saa': (0, (3, 1, 1, 1)),  # Custom dash pattern
    'true_saa': (0, (5, 5)),             # Another custom dash pattern
    'joint_order_saa': (0, (2, 2, 10, 2))  # New custom dash pattern
}

# Markers dictionary
markers = {
    'ignorant_saa': 'o',  # Circle
    'robust': 's',         # Square
    'km': 'D',             # Diamond
    'subsample_saa': '^',  # Triangle
    'true_saa': 'P',       # Plus (filled)
    'joint_order_saa': 'X' # Cross (filled)
}



df = pd.read_csv('./datasets/train.csv')
grouped_df = df.groupby(['Order Date', 'Category']).size().reset_index(name='sales')


def sample_dist(N, grouped_df, category):
    filtered_df = grouped_df[grouped_df['Category'] == category]

    if N > len(filtered_df):
        raise ValueError("Sample size cannot be greater than the number of available rows.")
    return filtered_df['sales'].sample(n=N, replace=False).reset_index(drop=True)


for category in grouped_df['Category'].unique():
# for category in ['Office Supplies']:
    print(f'###### CATEGORY: {category} ######')
    distribution_name = f'kaggle_sales_data_{category}'

    gminus_calc_list = grouped_df[grouped_df['Category'] == category]['sales'].to_numpy()




    file_loc = f'./data/{distribution_name}.csv'
    print(f'Reading data from: {file_loc}')
    df = pd.read_csv(file_loc)
    print(f'Data frame distribution name: {df["distribution"].unique()}')
    b_values = df['b'].unique()
    # print(b_values)
    lam_values = df['lam'].unique()


    # algo_list = ['ignorant_saa', 'robust', 'cave', 'km', 'subsample_saa', 'aim', 'burnetas_smith']
    algo_list = ['ignorant_saa', 'robust', 'km', 'subsample_saa']

    sns.set_palette('husl', len(algo_list))


    for b in b_values:
        for lam in lam_values:
            print(f'Creating plot for: b = {b} and lam = {lam}')
            plot_df = df[(df['b'] == b) & (df['lam'] == lam)]

            print(plot_df.head(5))


            # eval_demand_list = np.random.poisson(lam = mean, size = int(1e7))
            eval_demand_list = gminus_calc_list
            rho = b / (b + h)
            print(f'Gminus(lambda): {np.mean(eval_demand_list < lam)} and rho: {rho}')


            qopt = helper.get_optimal_quantile(b, h, eval_demand_list)
            opt_cost = helper.get_newsvendor_cost(qopt, b, h, eval_demand_list)
            print(f'Empirical Optimal order: {qopt} and cost: {opt_cost}')

            mm_opt = helper.get_optimal_robust(b, h, lam, qbar, eval_demand_list)
            mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, eval_demand_list)
            print(f'Optimal robust order: {mm_opt} and cost: {mm_cost}')


            # Create the subplots
            # fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            fig, axes = plt.subplots(1, 2, figsize=(25, 10), sharey=True)

            # Plot for 'true_cost' metric
            for algorithm, style in line_styles.items():
                subset = plot_df[plot_df['algorithm'] == algorithm]
                marker = markers[algorithm]


                if PLOT_REGRET:
                    sns.lineplot(data=subset[subset['metric'] == 'true_cost'].assign(value=lambda df: df['value'] - opt_cost),
                                x='N', y='value', ax=axes[0], 
                                label=f"{algorithm}", marker = marker, linestyle=style)
                else:
                    sns.lineplot(data=subset[subset['metric'] == 'true_cost'], 
                                x='N', y='value', ax=axes[0], 
                                label=f"{algorithm}", linestyle=style)

            if PLOT_REGRET:
                axes[0].axhline(y=0.0, color='black', linestyle='-', 
                                label=f'Optimal Order {np.round(qopt, decimals=3)}')
            else:
                axes[0].axhline(y=opt_cost, color='black', linestyle='-', 
                                label=f'Optimal Order {np.round(qopt, decimals=3)}')
            axes[0].set_xlabel('N')
            axes[0].set_ylabel('Cost')

            # Plot for 'robust_cost' metric
            for algorithm, style in line_styles.items():
                subset = plot_df[plot_df['algorithm'] == algorithm]
                marker = markers[algorithm]

                sns.lineplot(data=subset[subset['metric'] == 'minmax_cost'], 
                            x='N', y='value', ax=axes[1], 
                            label=f"{algorithm}", linestyle=style, marker=marker)

            axes[1].axhline(y=mm_cost, color='black', linestyle='-', 
                            label=f'Optimal Order {np.round(mm_opt, decimals=3)}')
            axes[1].set_xlabel('N')
            axes[1].set_ylabel('MinMax Cost')

            # Add legend to the first plot
            axes[0].legend()
            axes[1].legend()

            # Adjust layout for better spacing
            fig.suptitle(fr'{distribution_name}, $\lambda = {lam}$, $G^-(\lambda) = {np.round(np.mean(gminus_calc_list < lam),decimals=4)}$, $\rho = {np.round(rho, decimals=4)}$')
            plt.tight_layout()


            # Show the plot
            fig_name = f'./figures/real_data/{distribution_name}_{b}_{lam}.pdf'
            plt.savefig(fig_name)
            plt.close('all')
