# prompt: sample 10000 random variables from D = max(0,X) where X = Normal(80, sigma^2). Do this for \sigma^2 \in \{20,25,30,35,40\}, then plot the empirical CDFs for each of these 400 sampled distributions. Use the husl seaborn color palette. Place a dashed line at D = 118.46 and call it lambda

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import matplotlib

from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)


sns.set_style("white")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')

# Set the sigmas
sigmas = [20, 25, 30, 35, 40]

# Create the plot
plt.figure(figsize=(12, 6))

# Use husl color palette
colors = sns.color_palette("husl", len(sigmas))

# Iterate over sigmas
for i, sigma in enumerate(sigmas):
    # Sample from D
    X = np.random.normal(80, (sigma), 1000000)  # Sample from X ~ N(80, sigma)
    D = np.maximum(0, X)

    # Calculate the empirical CDF
    x_values = np.sort(D)
    y_values = np.arange(1, len(D) + 1) / len(D)

    # Plot the ECDF
    plt.plot(x_values, y_values, label=rf"$\sigma = {sigma}$", color=colors[i])

# Add a dashed line at D = 118.46

# Add a horizontal line at y = 0
plt.axvline(x=118.46, color='black', linestyle='--', linewidth=2.0)
plt.xticks(ticks=[118.46], labels=[r"$\lambda$"])

plt.axhline(y=0.9, color='black',linestyle='--',linewidth=2.0)
plt.yticks(ticks=[0.9], labels=[r"$\rho$"])  # Keep y-ticks removed as requested





# Customize the plot
plt.xlabel(r"$q$")
plt.ylabel(r"$G(q)")
# plt.title("Empirical CDFs of D for different values of sigma")
plt.legend(
    loc='upper left', 
    bbox_to_anchor=(1, 1),  # Place legend outside the plot
    frameon=False  # Remove legend frame if preferred
)
# saves the figure
plt.savefig(f'./figures/normalfigures.pdf', bbox_inches = 'tight')

plt.close('all')