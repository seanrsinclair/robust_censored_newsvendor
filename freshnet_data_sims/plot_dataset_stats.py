import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import random
import math
import pandas as pd

import pickle


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms
import helper

import freshnet_helper


from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

"""
plot_dataset_stats.py

Purpose
-------
Produce basic descriptive statistics and plots for the FreshRetailNet datasets
(train/eval), focusing on censoring intensity and the structure of historical
order levels ("selling seasons") per product.

Outputs (saved under ./figures/)
--------------------------------
- oos_freq.pdf              : out-of-stock (censored) frequency per product
- num_ordering_levels.pdf   : number of distinct order levels per product
- nk_hist.pdf               : N_K at the max order level λ per product
- mean_samples_vs_rank.pdf  : avg sample count vs rank of order levels (high→low)

Caching
-------
Builds and caches a per-product dictionary of censored sales keyed by order
level to ./datasets/training_sales_and_order_levels.pkl to avoid recomputation.
"""



# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')


print(f'Reading in training and evaluation datasets')
train, test = pd.read_csv("./datasets/train.csv"), pd.read_csv('./datasets/eval.csv')


product_ids = train["product_id"].unique()
training_sales_and_order_levels = {}

print(f'Creating dataset per product IDs')

pkl_path = "./datasets/training_sales_and_order_levels.pkl"

if os.path.exists(pkl_path):
    print(f"Loading cached training_sales_and_order_levels from {pkl_path}")
    with open(pkl_path, "rb") as f:
        training_sales_and_order_levels = pickle.load(f)
else:
    print(f"Cache not found. Creating dataset per product IDs")
    for product_id in product_ids:
        print(f"Product: {product_id}")

        order_levels, censored_demands = freshnet_helper.get_censored_sales_data(
            product_id, dataset='train'
        )
        training_sales_and_order_levels[product_id] = {
            level: demands
            for level, demands in zip(order_levels, censored_demands)
        }

    with open(pkl_path, "wb") as f:
        pickle.dump(training_sales_and_order_levels, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved training_sales_and_order_levels to {pkl_path}")


print(f'Finished setting up the data and saving to pickle for future changes')


# Plot histogram of censorship % per product in training set


oos_freq = train.groupby("product_id")["censored"].mean()
oos_freq_df = oos_freq.reset_index(name="oos_frequency")
mean_val = oos_freq_df['oos_frequency'].mean()

plt.figure(figsize=(10,6))
plt.hist(oos_freq_df["oos_frequency"], bins=50)
plt.xlabel("Out-of-Stock Frequency")
plt.ylabel("Number of Products")
plt.axvline(mean_val, color='grey', linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.legend()
# plt.title("Histogram of Product OOS Frequencies")
plt.savefig('./figures/oos_freq.pdf', bbox_inches='tight')
plt.close('all')
# plt.show()



values = oos_freq_df["oos_frequency"]

median_val = values.median()
q1 = values.quantile(0.25)
q3 = values.quantile(0.75)
min_val = values.min()
max_val = values.max()

median_val, q1, q3, min_val, max_val, len([e for e in values if e > 0.5])/len(values)


# Plot histogram of number of order levels across all products in training set


# Get number of order levels for each product
product_ids = training_sales_and_order_levels.keys()
n_order_levels = {}
for product_id in product_ids:
    n_order_levels[product_id] = len(training_sales_and_order_levels[product_id].keys())
    
mean_val = np.mean(list(n_order_levels.values()))

plt.figure(figsize=(10,6))
plt.hist(n_order_levels.values(), bins=50)
plt.xlabel("Number of Selling Seasons")
plt.ylabel("Number of Products")
plt.axvline(mean_val, color='grey', linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.legend()
# plt.title("Histogram of Product OOS Frequencies")
plt.savefig('./figures/num_ordering_levels.pdf', bbox_inches='tight')
plt.close('all')

# plt.show()


values = list(n_order_levels.values())

mean_val = np.mean(values)
median_val = np.median(values)
q1 = np.percentile(values, 25)
q3 = np.percentile(values, 75)
min_val = min(values)
max_val = max(values)

mean_val, median_val, q1, q3, min_val, max_val, np.std(values), len([e for e in values if e >= 50])/len(values)



# Get $N_K$ for each product

NK_dct = {}
for product_id in product_ids:
    lam = max(training_sales_and_order_levels[product_id].keys())
    NK_dct[product_id] = len(training_sales_and_order_levels[product_id][lam])
#     print(product_id, lam, NK_dct[product_id])

mean_val = np.mean(list(NK_dct.values()))

plt.figure(figsize=(10,6))
plt.hist(NK_dct.values(), bins=30)
plt.xlabel(r"$N_K$")
plt.ylabel("Number of Products")
plt.axvline(mean_val, color='grey', linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.legend()
# plt.title("Histogram of Product OOS Frequencies")
plt.savefig('./figures/nk_hist.pdf', bbox_inches='tight')
plt.close('all')

# plt.show()


values = list(NK_dct.values())

mean_val = np.mean(values)
median_val = np.median(values)
q1 = np.percentile(values, 25)
q3 = np.percentile(values, 75)
min_val = min(values)
max_val = max(values)

mean_val, median_val, q1, q3, min_val, max_val, len([e for e in values if e >= 20])/len(values)


# Plot $N_k$ vs rank

# For each k (rank), store a list of sample counts across all products
rank_counts = {}  # rank → list of N_k values across products

for product_id in product_ids:
    # Sorted keys from high → low
    keys_sorted = sorted(training_sales_and_order_levels[product_id].keys(), reverse=True)

    # Iterate over ranks
    for k in range(0, len(keys_sorted)):  # k = 1 → second highest, 2 → third highest, ...
        key_k = keys_sorted[k]
        count_k = len(training_sales_and_order_levels[product_id][key_k])

        if k not in rank_counts:
            rank_counts[k] = []
        rank_counts[k].append(count_k)

mean_rank_counts = {k: np.mean(v) for (k, v) in rank_counts.items()}

max_rank = 20
ranks = [k for k in sorted(mean_rank_counts.keys()) if k < max_rank]
means = [mean_rank_counts[k] for k in ranks]

plt.figure(figsize=(10,6))
plt.plot(ranks, means,marker='o')
plt.xlabel("Rank")
plt.ylabel("Avg. Number of Samples")
# plt.grid(True)
# plt.xticks(ranks)
plt.savefig("./figures/mean_samples_vs_rank.pdf", bbox_inches="tight")
plt.close('all')
