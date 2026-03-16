import numpy as np
import pandas as pd
import algorithms
import helper
from helper import sample_dist

"""
run_well_separated.py

Purpose
-------
Run Monte Carlo simulations for the *well-separated* synthetic censoring setup and
write a long-format CSV of algorithm outcomes. This is the data-generation script
used to produce `./data/{distribution}_well_separated.csv`, which downstream
table/plot scripts consume for the paper.

High-level experiment design
----------------------------
Each instance corresponds to a distribution family and parameter `param`, an
underage cost `b` (holding cost fixed at h=1), and a robustness parameter
`lam` that also acts as an order/censoring level for the first selling season.

For each instance, we generate a two-season dataset of size N:
  - Season 1 uses order level q1 = lam and censored demands min(D1, lam).
  - Season 2 uses a random order level q2 drawn uniformly from [lam/4, 3lam/4]
    and censored demands min(D2, q2).

We evaluate multiple learning algorithms on this censored data. For each run, we
record:
  - the solution (order quantity) returned by the algorithm,
  - the *true* expected newsvendor cost under the full demand distribution,
  - the *robust/min-max* cost under the well-separated robust objective,
  - an identifiability label produced by the evaluation routine.

Outputs
-------
Creates a CSV at:
    ./data/{distribution_name}_well_separated.csv
in long format with columns including (at least):
    ['N','b','h','distribution','param','lam','order_level_two','qbar',
     'algorithm','metric','value']

Each algorithm contributes multiple rows per simulation replicate:
    metric == 'true_cost'   : expected true newsvendor cost
    metric == 'minmax_cost' : robust/min-max cost for the chosen order
    metric == 'id'          : categorical identifiability label
                              {'identifiable','knife_edge','unidentifiable'}

"""


DEBUG = False
INCLUDE_TRUE = True

# Meta parameter specification for the simulations
distribution_list = ['continuous_uniform', 'truncated_exponential']

num_iters = 100 # normally 100
h = 1
# N_list = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_list = [500]

np.random.seed(5)

# Loops over each distribution
for distribution_name in distribution_list:
    print(f'#### RUNNING FOR: {distribution_name} ####')
    data = []

    file_name = f'./data/{distribution_name}_well_separated.csv'


    qbar, param_list, b_list, rho_list, lam_list, gamma = helper.get_well_separated_parameters(distribution_name) # gets the evaluation parameters


    print(f'Running for: M = {qbar}, params: {param_list}, b values: {b_list}, and lam: {lam_list}')
            # specified for that distribution
    gminus_calc_list = sample_dist(int(1e7), distribution_name, param_list[0]) # data used for sampling distribution


    # list of algorithms to evaluate
    algorithm_list = [
        # ('ignorant_saa', algorithms.ignorant_saa),
        # ('subsample_saa', algorithms.subsample_saa),
        # ('km', algorithms.km_estimate),
        # ('joint_order_saa', algorithms.joint_order_saa),
        ('robust', lambda order_levels, censored_data, b, h: algorithms.robust_well_separated_saa(order_levels, censored_data, b, h, qbar, gamma, np.mean(gminus_calc_list < lam), delta=0.05)),
        ('robust_plus',lambda order_levels, censored_data, b, h: algorithms.robust_plus_well_separated_saa(order_levels, censored_data, b, h, qbar, gamma, np.mean(gminus_calc_list < lam), delta=0.05))
    ]

    for param in param_list: # Loop over each parameter for the distribution type
        gminus_calc_list = sample_dist(int(1e7), distribution_name, param)
        if 'robust' in algorithm_list: # just gotta reset the Gminus estimate for a potential tighter confidence term
            algorithm_list['robust'] = lambda order_levels, censored_data, b, h: algorithms.robust_well_separated_saa(order_levels, censored_data, b, h, qbar, gamma, np.mean(gminus_calc_list < lam), delta=0.05)
            algorithm_list['robust_plus'] = lambda order_levels, censored_data, b, h: algorithms.robust_plus_well_separated_saa(order_levels, censored_data, b, h, qbar, gamma, np.mean(gminus_calc_list < lam), delta=0.05)

        for b in b_list: # Loop over each parameter
            rho = b / (b+h)
            for lam in lam_list:
                for N in N_list:

                    print(f'############# Running for N: {N}, b: {b}, lam: {lam}, param: {param} #############')
                    if DEBUG: # Output vanilla optimal and robust optimal order
                        print(f'Gminus(lambda): {np.mean(gminus_calc_list < lam)} and rho: {rho}')
                        qopt = helper.get_optimal_quantile(b, h, gminus_calc_list)
                        opt_cost = helper.get_newsvendor_cost(qopt, b, h, gminus_calc_list)
                        print(f'Empirical Optimal order: {qopt}, vanilla_regret: {opt_cost - opt_cost} and mm_regret: {helper.get_well_separated_robust_cost(qopt, b, h, lam, qbar, gamma, gminus_calc_list)}')

                        mm_opt = helper.get_well_separated_optimal_robust(b, h, lam, qbar, gamma, gminus_calc_list)
                        mm_cost = helper.get_well_separated_robust_cost(mm_opt, b, h, lam, qbar, gamma, gminus_calc_list)
                        print(f'Optimal robust order: {mm_opt}, vanilla_regret: {helper.get_newsvendor_cost(mm_opt, b, h, gminus_calc_list) - opt_cost} and mm_regret: {mm_cost}')

                    for _ in range(num_iters):
                        order_level_two = float(np.random.randint(int((lam / 4)), int((3*lam / 4))))
                        # sample a dataset
                        true_demand_list_one = sample_dist(N, distribution_name, param)
                        true_demand_list_two = sample_dist(N, distribution_name, param)
                        eval_demand_list = gminus_calc_list

                        # censor the dataset by the two selling seasons
                        censored_demand_list_one = np.minimum(lam, true_demand_list_one)
                        censored_demand_list_two = np.minimum(order_level_two, true_demand_list_two)
                        
                        order_levels = [lam, order_level_two]
                        censored_demands = [censored_demand_list_one, censored_demand_list_two]

                        for algo_name, algo_func in algorithm_list: # Loops over each algorithm and evaluates it
                            sol, true_cost, mm_cost, ind = helper.evaluate_algorithm(
                                algo_name, algo_func, order_levels, censored_demands, b, h, qbar, eval_demand_list, gamma = gamma
                            )
                            if DEBUG: print(f'Algo: {algo_name}, sol: {sol}, costs: {(true_cost, mm_cost)}')
                            # Add true and minmax cost results to data
                            common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'param': param, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                            helper.add_to_data(data, algo_name, 'true_cost', true_cost, **common_params)
                            helper.add_to_data(data, algo_name, 'minmax_cost', mm_cost, **common_params)
                            if ind == 1:
                                helper.add_to_data(data, algo_name, 'id', 'identifiable', **common_params)
                            elif ind == 0:
                                helper.add_to_data(data, algo_name, 'id', 'knife_edge', **common_params)
                            else:
                                helper.add_to_data(data, algo_name, 'id', 'unidentifiable', **common_params)
                            
                        if INCLUDE_TRUE: # augmenting the two lists for the full distribution
                            sol, true_cost, mm_cost = helper.evaluate_true_saa(lam, np.concatenate((true_demand_list_one,true_demand_list_two)), b, h, qbar, eval_demand_list, gamma = gamma)
                            common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'param': param, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                            helper.add_to_data(data, 'true_saa', 'true_cost', true_cost, **common_params)
                            helper.add_to_data(data, 'true_saa', 'minmax_cost', mm_cost, **common_params)
                            helper.add_to_data(data, 'true_saa', 'id', 'identifiable', **common_params)


    # # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # # Save the DataFrame as a CSV file
    df.to_csv(file_name, index=False)  # index=False to avoid writing row numbers

    print(f"Data has been saved to {file_name}")