# The Data-Driven Censored Newsvendor Problem

In this repository we include all of the code used for generating the figures and numerical simulations.

## Supplementary Files
- `algorithms.py` implements all of the algorithms
- `helper.py` implements helper functions to evaluate the newsvendor costs, solve for the empirical quantile, etc

## Figure Files
- `delta_qcrit_figure.py` creates the figure outputting $q^{\Delta}$ and $\Delta$ versus the $\lambda$ value
- `minimaxrisk_figure.py` creates the figure comparing $\textsf{Regret}(q)$ versus the order level $q$

## Running a Simulation
- `run_synthetic.py` runs the simulations on the synthetically generated data, saving the outputs from each iteration
- `run_real.py` runs the simulations on the Kaggle superstore sales dataset, saving the outputs from each iteration

## Creating the Figures and Tables
- `real_histogram_plots.py` plots the histograms for the three categories in the Kaggle data
- `real_lam_tables.py` creates the pivot tables comparing the algorithms on relative regret under the different $\lambda$ values for the Kaggle data
- `synthetic_var_tables.py` creates the pivot tables comparing the algorithms on relative regret under different $\sigma^2$ values with Normally distributed demand
- `synthetic_lam_tables.py` creates the pivot tables comparing the algorithms on relative regret under the different $\lambda$ values of the synthetic demand data
- `synthetic_plots.py` creates the line plots comparing the algorithms on true minmax regret versus the number of datapoints $N$
- `real_q_2_tables.py` creates the pivot tables comparing RCN+ on relative regret under different historical ordering quantities