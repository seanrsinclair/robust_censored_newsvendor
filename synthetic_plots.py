import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from helper import sample_dist


PLOT_REGRET = True
MAKE_LEGEND = True
h = 1


distribution_list = ['uniform']

plt.style.use('PaperDoubleFig.mplstyle.txt')

plt.rc('text', usetex=True)
# Add amsmath to the LaTeX preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

algo_list = ['true_saa', 'ignorant_saa', 'robust', 'km', 'subsample_saa', 'joint_order_saa']


sns.set_palette('husl', len(algo_list))


# Line styles dictionary
line_styles = {
       'true_saa': (0, (5, 5)),             # Another custom dash pattern 
        'robust': 'dashed',
        'joint_order_saa': (0, (2, 2, 10, 2)),  # New custom dash pattern
            'ignorant_saa': 'dotted',
            'km': 'dashdot',
    'subsample_saa': (0, (3, 1, 1, 1)),  # Custom dash pattern
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

mapping = {
    'ignorant_saa': r'\textsf{Naive SAA}',
    'true_saa': r'\textsf{True SAA}',
    'joint_order_saa': r'\textsf{Censored SAA}',
    'km': r'\textsf{Kaplan-Meier}',
    'robust': r'\textsf{RCN}',
    'subsample_saa': r'\textsf{Subsample SAA}'
    }

color_mapping = {
    'true_saa': (0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
    'robust': (0.7350228985632719, 0.5952719904750953, 0.1944419133847522),
    'joint_order_saa': (0.3126890019504329, 0.6928754610296064, 0.1923704830330379),
    'ignorant_saa': (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
    'km': (0.23299120924703914, 0.639586552066035, 0.9260706093977744),
    'subsample_saa': (0.9082572436765556, 0.40195790729656516, 0.9576909250290225)
}


for distribution_name in distribution_list:
    print(f'#### PLOTS FOR: {distribution_name} ####')
    file_loc = f'./data/{distribution_name}.csv'
    df = pd.read_csv(file_loc)

    param_values = df['param'].unique()
    print(param_values)

    for param in param_values:
    # for param in [param_values[0]]:
        df = pd.read_csv(file_loc)
        df = df[df['param'] == param]
        b_values = df['b'].unique()
        lam_values = df['lam'].unique()


        if distribution_name == 'negative_binomial':
            param = eval(param) # converts into a proper tuple

        qbar, param_list, b_list, rho_list, lam_list = helper.get_parameters(distribution_name)
        gminus_calc_list = sample_dist(int(1e7), distribution_name, param)

        for b in b_values:
        # for b in [b_values[0]]:
            for lam in reversed(lam_values):
            # for lam in [lam_values[-1]]:
                print(f'Creating plot for: b = {b} and lam = {lam}')
                plot_df = df[(df['b'] == b) & (df['lam'] == lam)]

                # eval_demand_list = np.random.poisson(lam = mean, size = int(1e7))
                eval_demand_list = sample_dist(int(1e7), distribution_name, param)
                rho = b / (b + h)
                print(f'Gminus(lambda): {np.mean(eval_demand_list < lam)} and rho: {rho}')


                qopt = helper.get_optimal_quantile(b, h, eval_demand_list)
                opt_cost = helper.get_newsvendor_cost(qopt, b, h, eval_demand_list)
                print(f'Empirical Optimal order: {qopt} and cost: {opt_cost}')

                mm_opt = helper.get_optimal_robust(b, h, lam, qbar, eval_demand_list)
                mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, eval_demand_list)
                print(f'Optimal robust order: {mm_opt} and cost: {mm_cost}')


                if mm_cost > 0:
                    IDENTIFIABLE = False
                else:
                    IDENTIFIABLE = True



                if IDENTIFIABLE:
                    # print(plot_df.head(5))

                    # Filter to rows where metric is 'minmax_cost'
                    minmax_df = plot_df[plot_df["metric"] == "true_cost"]

                    # Step 1: Group by algorithm and take the average of the 'value' metric
                    average_values = (
                        plot_df[plot_df["metric"] == "true_cost"]
                        .groupby(["algorithm", "N"])["value"]
                        .mean()
                        .reset_index()
                        .rename(columns={"value": "avg_value"})
                    )

                    # Step 2: Calculate relative percent increase over mm_cost
                    print(f'Minmax cost: {mm_cost}')
                    average_values["relative_increase"] = (average_values["avg_value"] - opt_cost) / opt_cost


                    print(average_values)

                    minmax_df["condition_met"] = (minmax_df["value"] - mm_cost) / mm_cost <= 0.1


                    smallest_N = (
                    average_values[average_values["relative_increase"] <= 0.1]
                    .groupby("algorithm")["N"]
                    .min()
                    .reindex(average_values["algorithm"].unique(), fill_value=float('nan'))
                    .reset_index()
                    .rename(columns={"N": "smallest_N"})
                    )

                    print(smallest_N)










                if not IDENTIFIABLE:
                    # Determine the sample complexity for error <= 5%
                    # print(plot_df.head(5))

                    # Filter to rows where metric is 'minmax_cost'
                    minmax_df = plot_df[plot_df["metric"] == "minmax_cost"]

                    # Step 1: Group by algorithm and take the average of the 'value' metric
                    average_values = (
                        plot_df[plot_df["metric"] == "minmax_cost"]
                        .groupby(["algorithm", "N"])["value"]
                        .mean()
                        .reset_index()
                        .rename(columns={"value": "avg_value"})
                    )

                    # Step 2: Calculate relative percent increase over mm_cost
                    print(f'Minmax cost: {mm_cost}')
                    average_values["relative_increase"] = (average_values["avg_value"] - mm_cost) / mm_cost


                    print(average_values)

                    minmax_df["condition_met"] = (minmax_df["value"] - mm_cost) / mm_cost <= 0.1


                    smallest_N = (
                    average_values[average_values["relative_increase"] <= 0.1]
                    .groupby("algorithm")["N"]
                    .min()
                    .reindex(average_values["algorithm"].unique(), fill_value=float('nan'))
                    .reset_index()
                    .rename(columns={"N": "smallest_N"})
                    )

                    print(smallest_N)



                # Create a single plot
                fig = plt.figure(figsize=(8, 6))
                ax = plt.gca()  # Get the current axes

                for algorithm, style in line_styles.items():
                    if (not IDENTIFIABLE and algorithm != 'true_saa') or IDENTIFIABLE:
                        subset = plot_df[plot_df['algorithm'] == algorithm].copy()
                        marker = markers.get(algorithm, None)  # Use `.get()` for safety
                        
                        # Subtract 'mm_cost' from 'value' for each entry
                        subset['adjusted_value'] = subset['value'] - mm_cost

                        if MAKE_LEGEND:
                            algo_label = mapping[algorithm]
                        else:
                            algo_label = algorithm

                        sns.lineplot(
                            data=subset[subset['metric'] == 'minmax_cost'], 
                            x='N', y='adjusted_value',  # Use the adjusted value column
                            ax=ax, 
                            label=algo_label, 
                            linestyle=style, 
                            color = color_mapping.get(algorithm, None),
                            marker=marker
                        )                # Add legend to the plot

                # Remove x and y labels
                ax.set_xlabel(r"$N$")
                ax.set_ylabel(r"$\textup{Regret}(q) - \Delta$")

                # Ensure the y-axis always includes 0
                y_min, y_max = ax.get_ylim()
                if y_min > 0:
                    ax.set_ylim(0, y_max)  # Extend to include 0
                elif y_max < 0:
                    ax.set_ylim(y_min, 0)  # Extend to include 0

                ax.set_xticks([0, 250, 500])

                # Adjust layout for better spacing
                plt.tight_layout()

                if not MAKE_LEGEND:
                    # Assuming `ax` is the current axes
                    if ax.get_legend():
                        ax.get_legend().remove()
                    fig_name = f'./figures/{distribution_name}/{param}_{b}_{int(lam)}.pdf'
                    plt.savefig(fig_name)
                    plt.close('all')

                else:
                    # Add legend to the plot
                    ax.legend()

                    legend = ax.legend(ncol = 3, loc= 'lower center', bbox_to_anchor=(0.5, -0.5, 0, 0))

                    # Increase the line width of legend lines
                    for line in legend.get_lines():
                        line.set_linewidth(5)  # Adjust line width as needed (e.g., 5)

                    # Increase the font size of legend text
                    # for text in legend.get_texts():
                        # text.set_fontsize(14)  # Adjust font size as needed (e.g., 14)         
                    helper.export_legend(legend, filename="./figures/legend.pdf")


