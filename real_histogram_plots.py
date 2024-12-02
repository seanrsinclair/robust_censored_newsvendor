import numpy as np
import pandas as pd
import algorithms
import helper
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid",palette='husl')
plt.style.use('PaperDoubleFig.mplstyle.txt')

DEBUG = False
INCLUDE_TRUE = True

qbar = 10
h = 1
num_iters = 50

# Reading in the dataset
df = pd.read_csv('./datasets/train.csv')
grouped_df = df.groupby(['Order Date', 'Category']).size().reset_index(name='sales')


max_sales = grouped_df["sales"].max()
print(f'Maximum sales value: {grouped_df["sales"].max()}')

# Get unique categories
categories = grouped_df['Category'].unique()

# Determine x and y limits
x_min, x_max = grouped_df['sales'].min(), grouped_df['sales'].max()
y_max = 0

# Find the maximum frequency for consistent y-axis limits
for category in categories:
    data = grouped_df[grouped_df['Category'] == category]
    counts, _ = np.histogram(data['sales'], bins=10)
    y_max = max(y_max, counts.max())


print(grouped_df.head(5))

# Prepare the data for CDF plot
categories = grouped_df['Category'].unique()
plt.figure(figsize=(10, 6))

for category in categories:
    sales_data = grouped_df[grouped_df['Category'] == category]['sales']
    sorted_sales = np.sort(sales_data)
    cdf = np.arange(1, len(sorted_sales) + 1) / len(sorted_sales)
    plt.plot(sorted_sales, cdf, label=category)

# Customize plot
plt.title("Empirical CDF of Sales by Category")
plt.xlabel("Sales")
plt.ylabel("CDF")
plt.legend(title="Category")
plt.grid(True)
plt.show()

# # Create a histogram for each category using a for loop
# for category in categories:

#     current_df = grouped_df[grouped_df['Category'] == category]
#     sales_mean = current_df["sales"].mean()
#     sales_std = current_df["sales"].std()

#     result = {
#         "Mean of sales": sales_mean,
#         "Standard deviation of sales": sales_std,
#     }

#     print(result)



#     plt.figure(figsize=(5, 5))
#     sns.histplot(data=grouped_df[grouped_df['Category'] == category], x="sales", bins=10, kde=True)
#     # plt.title(f'Sales Distribution for {category}')
#     # plt.xlabel("Sales")
#     plt.xlabel(None)  # Remove x-axis label
#     plt.ylabel(None)  # Remove y-axis label
#     plt.xlim(x_min, x_max)  # Set consistent x-axis limits
#     plt.ylim(0, y_max)      # Set consistent y-axis limits

#     # plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.savefig(f'./figures/demand_{category}_histogram.pdf',bbox_inches='tight')
#     plt.close('all')