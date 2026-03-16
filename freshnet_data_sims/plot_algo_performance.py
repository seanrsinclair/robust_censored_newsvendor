import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import random
import math
import pandas as pd
import pickle
import bisect
from lifelines import KaplanMeierFitter
import statsmodels.api as sm
from scipy import stats


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helper


from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

"""
plot_algo_performance.py

Purpose
-------
Generate summary figures/tables that characterize algorithm performance on the
FreshRetailNet real-world dataset using the consolidated results file:
    ./data/freshnet_algo_data.csv

What this script produces
-------------------------
1) Identifiability rates by b:
   - Computes the fraction of products labeled identifiable at each b.

2) Distribution diagnostics (product-level):
   - Histogram of G^-(lambda) across products (saved to ./figures/gminslam.pdf).
   - Histogram of Delta across products, stratified by b (saved to ./figures/Delta_hist.pdf).

3) Algorithm performance summaries:
   - Computes additive_regret and relative_regret from minmax-cost results.
     * If identifiable==0: relative_regret compares (cost - Delta)/Delta.
     * If identifiable==1: relative_regret compares cost/qopt_cost.
   - Selects a representative qbar per product (the “middle” qbar when multiple
     are available) and filters to that qbar before plotting.

4) Regret bar charts (averaged over products):
   - Identifiable products: bar chart of mean additive regret by algorithm and b
     (saved to ./figures/id_regret.pdf) and a standalone legend PDF
     (saved to ./figures/freshnet_legend.pdf).
   - Unidentifiable products: bar chart of mean (regret - Delta) by algorithm and b
     (saved to ./figures/uid_regret.pdf).

5) Robustness vs stockout frequency:
   - Joins training data (./datasets/train.csv) to compute per-product OOS
     frequency and reports Pearson correlations between OOS frequency and
     additive regret, by algorithm and b.

Inputs
------
- ./data/freshnet_algo_data.csv : long-format algorithm evaluation results.
- ./datasets/train.csv          : raw training data used to compute OOS frequency.

Outputs
-------
- PDFs saved under ./figures/ (see filenames above).
- Printed summary tables to stdout (identifiability rates, Delta stats, correlations).
"""



# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')
plt.rc('text', usetex=True)
# Add amsmath to the LaTeX preamble
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'




res = pd.read_csv("./data/freshnet_algo_data_updated_tune.csv")


train = pd.read_csv("./datasets/train.csv")


# Report fraction of identifiable for each value of $\rho$


tmp = res[["product", "b", "identifiable"]].drop_duplicates()
fraction_identifiable = (
    tmp.groupby("b")["identifiable"]
       .mean()
       .reset_index(name="fraction_identifiable")
)
print(fraction_identifiable.head())




# Merge the three results datasets

print(f'### CALCULATING IDENTIFIABILITY RATES PER B ###')
res_final = res
print(res_final.head())


# Focusing on minmax cost

minmax_df = res_final[res_final['metric'] == 'minmax_cost']
minmax_df = minmax_df[minmax_df['qopt_cost'] > 0].copy()
minmax_df['value'] = pd.to_numeric(minmax_df['value'], errors='coerce')
minmax_df['additive_regret'] = minmax_df['value']-minmax_df['Delta']
minmax_df["relative_regret"] = np.where(
    minmax_df["identifiable"] == 0,
    (minmax_df["value"] - minmax_df["Delta"]) / minmax_df["Delta"],
    (minmax_df["value"]) / minmax_df["qopt_cost"]
)
minmax_df['Gminuslam-rho'] = np.maximum(0,(minmax_df['b']/(minmax_df['b']+minmax_df['h']))-minmax_df['Gminuslam'])


# Plot distribution of $G^-(\lambda)$
print(f'### CREATING DISTRIBUTION PLOT OF G^-(\lambda) ###')
tmp = minmax_df[['product', 'Gminuslam']].drop_duplicates()
mean_val = np.mean(tmp['Gminuslam'])
plt.figure(figsize=(10,6))
plt.hist(tmp["Gminuslam"], bins=50)
plt.xlabel(r"$G^-(\lambda)$")
plt.ylabel("Number of Products")
plt.axvline(mean_val, color='grey', linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.legend()
# plt.title("Histogram of Product OOS Frequencies")
plt.savefig('./figures/gminslam.pdf', bbox_inches='tight')
plt.close('all')
# plt.show()


filtered_df = minmax_df.copy()
print(filtered_df["qbar"].unique())


# Plot distribution of $\Delta$

print(f'### CREATING DISTRIBUTION PLOT OF DELTA ###')




# 1. Get unique rows at (product, b) level
tmp = filtered_df[["product", "b", "Delta"]].drop_duplicates()
tmp = tmp[tmp['Delta'] > 0].copy()

# 2. Prepare plot
plt.figure(figsize=(10, 6))
sns.set_palette("husl", len(tmp["b"].unique()))

# 3. Plot histogram for each value of b

for b in tmp['b'].unique():
    subset = tmp[tmp["b"] == b]
    sns.histplot(subset["Delta"], kde=False, stat = 'probability', label=b, alpha=0.5, bins=50)


# 4. Labels and legend
plt.xlabel(r"$\Delta$")
plt.ylabel("Frequency")
plt.legend(title=r"$b$", bbox_to_anchor=(1, 1.05), loc='upper left')
plt.tight_layout()
plt.savefig('./figures/Delta_hist.pdf', bbox_inches='tight')
# plt.show()
plt.close('all')



mean_deltas = tmp.groupby("b")["Delta"].mean()
median_deltas = tmp.groupby("b")["Delta"].median()
tmp["Delta_gt_10"] = (tmp["Delta"] >= 56).astype(int)

# Compute fraction for each b
fraction_exceeds = (
    tmp.groupby("b")["Delta_gt_10"]
       .mean()            # since it's 0/1, mean = fraction
       .rename("fraction_exceeding_10")
)

print(mean_deltas)
print(fraction_exceeds)

# Plot average regret across products, by identifiability class


algs = ['robust_1', 'robust_plus_1']
algs += ['km', 'joint_order_saa']
# algs = filtered_df['algorithm'].unique()

print(algs)
sns.set_palette('husl', len(algs))
alg_names ={'km': r'$\textsf{Kaplan-Meier}$', 'joint_order_saa': r'\textsf{Censored SAA}', 'robust_plus_1': r'\textsf{RCN+}', 'robust_1': r'\textsf{RCN}', 'robust_sol': r'\textsf{Robust}'}
# Optional: LaTeX / pretty legend names
# alg_names = {e: e for e in algs}
rhos = {3: 0.75, 9: 0.9, 49: 0.98}
fewer_algs = filtered_df[filtered_df['algorithm'].isin(algs)]

avg_minimax_cost_per_id = (
    fewer_algs
    .groupby(["algorithm", "b", "identifiable"])
    .agg(
        additive_regret_mean=("additive_regret", "mean"),
        additive_regret_sem=("additive_regret", "sem"),
        relative_regret=("relative_regret", "mean"),
        gminuslam_rho=("Gminuslam-rho", "mean"),
    )
    .reset_index()
)
avg_minimax_cost_per_id["additive_regret_sem"] = avg_minimax_cost_per_id["additive_regret_sem"].fillna(0.0)



print(avg_minimax_cost_per_id)

# Pivot to get algorithms as columns and b as index
df_plot = avg_minimax_cost_per_id[avg_minimax_cost_per_id['identifiable'] == 1].copy()
df_plot['rho'] = df_plot['b'].map(rhos)


# Convert algorithm to pretty printable version
df_plot['alg_pretty'] = df_plot['algorithm'].map(alg_names).fillna(df_plot['algorithm'])

# --- Pivot for grouped bar chart ---
pivot_mean = df_plot.pivot_table(
    index='b',
    columns='alg_pretty',
    values='additive_regret_mean'
)
pivot_sem = df_plot.pivot_table(
    index='b',
    columns='alg_pretty',
    values='additive_regret_sem'
)

# Sort by rho for cleaner plotting
pivot_mean = pivot_mean.sort_index()
pivot_sem = pivot_sem.reindex(index=pivot_mean.index, columns=pivot_mean.columns).fillna(0.0)




print(f'### CREATING ID REGRET PLOTS ###')

# --- Plot ---
plt.figure(figsize=(10, 6))
pivot_mean.plot(
    kind='bar',
    figsize=(10, 6),
    legend=False,
    yerr=pivot_sem,
    capsize=4,
    error_kw={"elinewidth": 2.5, "capthick": 2.5}
)

plt.xlabel(r'$b$')
plt.ylabel(r'$\textup{Regret}(q)$')
# plt.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
plt.xticks(rotation=0)
plt.ylim(0,30)
plt.tight_layout()
plt.savefig('./figures/id_regret.pdf', bbox_inches='tight')
# plt.show()
plt.close('all')



print(f'### CREATING ID REGRET LEGEND ###')

# Keep pivot variable for downstream legend code
pivot = pivot_mean.copy()
print(pivot.columns)

# prints out this: ['robust_3', 'robust_plus_3', 'km', 'joint_order_saa']
# rename using mapping

mapping = {
    'robust_1': r'\textsf{RCN}',
    'robust_plus_1':r'\textsf{RCN}$^+$',
    'joint_order_saa': r'\textsf{Censored SAA}',
    'km': r'\textsf{Kaplan-Meier}',
    }

pivot = pivot.rename(columns=mapping)
print(pivot.columns)


# --- Legend-only figure (no bar plot), with husl colors ---
import matplotlib.patches as mpatches

labels = list(pivot.columns)  # after rename(mapping)

# Match seaborn husl palette used elsewhere
palette = sns.color_palette("husl", len(labels))

fig, ax = plt.subplots(figsize=(12, 1.8))  # larger canvas
ax.axis("off")

handles = [
    mpatches.Patch(facecolor=palette[i], label=lab)
    for i, lab in enumerate(labels)
]

legend = ax.legend(
    handles=handles,
    labels=labels,
    ncol=len(labels),
    loc="center",
    frameon=False,
    handlelength=1.5,
    columnspacing=2.0,
)

# --- Save legend only ---
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bbox = legend.get_window_extent(renderer=renderer).expanded(1.10, 1.40)
bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

fig.savefig(
    "./figures/freshnet_legend.pdf",
    dpi="figure",
    bbox_inches=bbox,
    pad_inches=0.15,
)

plt.close(fig)



print(f'## FINISHED MAKING FIGURE ##')



print(f'### CREATING UNID REGRET PLOTS ###')




# Pivot to get algorithms as columns and b as index
df_plot = avg_minimax_cost_per_id[avg_minimax_cost_per_id['identifiable'] == 0].copy()
rhos = {3: 0.75, 9: 0.9, 49: 0.98}
df_plot['rho'] = df_plot['b'].map(rhos)


# Convert algorithm to pretty printable version
df_plot['alg_pretty'] = df_plot['algorithm'].map(alg_names).fillna(df_plot['algorithm'])

# --- Pivot for grouped bar chart ---
pivot_mean = df_plot.pivot_table(
    index='b',
    columns='alg_pretty',
    values='additive_regret_mean'
)
pivot_sem = df_plot.pivot_table(
    index='b',
    columns='alg_pretty',
    values='additive_regret_sem'
)

# Sort by rho for cleaner plotting
pivot_mean = pivot_mean.sort_index()
pivot_sem = pivot_sem.reindex(index=pivot_mean.index, columns=pivot_mean.columns).fillna(0.0)

# --- Plot ---
plt.figure(figsize=(10, 6))
pivot_mean.plot(
    kind='bar',
    figsize=(10, 6),
    legend=False,
    yerr=pivot_sem,
    capsize=4,
    error_kw={"elinewidth": 2.5, "capthick": 2.5}
)

plt.xlabel(r'$b$')
plt.ylabel(r'$\textup{Regret}(q)-\Delta$')
# plt.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
plt.xticks(rotation=0)
plt.ylim(0,30)
plt.tight_layout()
plt.savefig('./figures/uid_regret.pdf', bbox_inches='tight')
# plt.show()
plt.close('all')

# Explore robustness to OOS frequency


print(f'### OOS FREQUENCY RESULTS ###')


oos_freq = train.groupby("product_id")["censored"].mean()
oos_freq_df = oos_freq.reset_index(name="oos_frequency")
oos_freq_df.head()

# Join the two to get a table that includes results with OOS frequency

merged = fewer_algs.merge(oos_freq_df, left_on="product", right_on="product_id", how="left")
merged = merged.drop(columns=["product_id"])
algs = merged['algorithm'].unique().tolist()

merged.head()



# Initialize a list to store your results
results_list = []

for b in [3, 9, 49]:
    for alg in algs:
        # Filter data
        robust_res = merged[merged['algorithm'] == alg]
        tmp = robust_res[(robust_res['b'] == b)]
        clean_data = tmp[['oos_frequency', 'additive_regret']].dropna()
        
        # Calculate Correlation only if we have data points
        if len(clean_data) > 1:
            corr, p_val = stats.pearsonr(clean_data['oos_frequency'], clean_data['additive_regret'])
            signif = p_val < 0.05
        else:
            # Handle empty/insufficient data cases
            corr, p_val, signif = None, None, False

        # Append result to list
        results_list.append({
            'algorithm': alg,
            'b': b,
            'correlation': corr,
            'p_value': p_val,
            'is_significant': signif,
            'significance_label': "(significant)" if signif else "(not significant)"
        })

# Create the DataFrame once the loop is finished
df_results = pd.DataFrame(results_list)

# View the final table
print(df_results)


plt.close('all')
