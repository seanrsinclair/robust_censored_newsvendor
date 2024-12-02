import numpy as np
import pandas as pd
import algorithms
import helper
from helper import sample_dist


DEBUG = False
INCLUDE_TRUE = True

# Meta parameter specification for the simulations

# distribution_list = ['uniform', 'exponential', 'poisson', 'negative_binomial', 'normal']
# distribution_list = ['uniform', 'exponential']
# distribution_list = ['exponential']
# distribution_list = ['uniform', 'poisson']
# distribution_list = ['poisson']
# distribution_list = ['normal']
# distribution_list = ['uniform']
# 
# distribution_list = ['uniform']
# distribution_list = 
distribution_list = ['uniform', 'exponential', 'poisson']


num_iters = 100
h = 1
# N_list = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_list = [500]

np.random.seed(5)

# Loops over each distribution
for distribution_name in distribution_list:
    print(f'#### RUNNING FOR: {distribution_name} ####')
    data = []

    file_name = f'./data/{distribution_name}_full_b.csv'


    qbar, param_list, b_list, rho_list, lam_list = helper.get_parameters(distribution_name) # gets the evaluation parameters
    print(f'Running for: M = {qbar}, params: {param_list}, b values: {b_list}, and lam: {lam_list}')
            # specified for that distribution
    gminus_calc_list = sample_dist(int(1e7), distribution_name, param_list[0]) # data used for sampling distribution


    # list of algorithms to evaluate
    algorithm_list = [
        ('ignorant_saa', algorithms.ignorant_saa),
        ('subsample_saa', algorithms.subsample_saa),
        ('robust', lambda lam, order_level_two, censored_one, censored_two, b, h: algorithms.robust_saa(lam, order_level_two, censored_one, censored_two, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05)),
        ('km', algorithms.km_estimate),
        ('joint_order_saa', algorithms.joint_order_saa)
    ]

    for param in param_list: # Loop over each parameter for the distribution type
        gminus_calc_list = sample_dist(int(1e7), distribution_name, param)
        if 'robust' in algorithm_list: # just gotta reset the Gminus estimate for a potential tighter confidence term
            algorithm_list['robust'] = lambda lam, order_level_two, censored_one, censored_two, b, h: algorithms.robust_saa(lam, order_level_two, censored_one, censored_two, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05),


        for b in b_list: # Loop over each parameter
            rho = b / (b+h)
            for lam in lam_list:
                for N in N_list:
                    print(f'############# Running for N: {N}, b: {b}, lam: {lam}, param: {param} #############')
                    if DEBUG: # Output vanilla optimal and robust optimal order
                        print(f'Gminus(lambda): {np.mean(gminus_calc_list < lam)} and rho: {rho}')
                        qopt = helper.get_optimal_quantile(b, h, gminus_calc_list)
                        opt_cost = helper.get_newsvendor_cost(qopt, b, h, gminus_calc_list)
                        print(f'Empirical Optimal order: {qopt}, cost: {opt_cost} and mm_cost: {helper.get_robust_cost(qopt, b, h, lam, qbar, gminus_calc_list)}')

                        mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
                        mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
                        print(f'Optimal robust order: {mm_opt}, cost: {helper.get_newsvendor_cost(mm_opt, b, h, gminus_calc_list)} and mm_cost: {mm_cost}')

                    for _ in range(num_iters):
                        order_level_two = float(np.random.randint(int((lam / 4)), int((3*lam / 4))))
                        # sample a dataset
                        true_demand_list_one = sample_dist(N, distribution_name, param)
                        true_demand_list_two = sample_dist(N, distribution_name, param)
                        eval_demand_list = gminus_calc_list

                        # censor the dataset by the two selling seasons
                        censored_demand_list_one = np.minimum(lam, true_demand_list_one)
                        censored_demand_list_two = np.minimum(order_level_two, true_demand_list_two)
                    

                        for algo_name, algo_func in algorithm_list: # Loops over each algorithm and evaluates it
                            sol, true_cost, mm_cost = helper.evaluate_algorithm(
                                algo_name, algo_func, lam, order_level_two, censored_demand_list_one, censored_demand_list_two, b, h, qbar, eval_demand_list
                            )
                            
                            # Add true and minmax cost results to data
                            common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'param': param, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                            helper.add_to_data(data, algo_name, 'true_cost', true_cost, **common_params)
                            helper.add_to_data(data, algo_name, 'minmax_cost', mm_cost, **common_params)
                            
                        if INCLUDE_TRUE: # augmenting the two lists for the full distribution
                            sol, true_cost, mm_cost = helper.evaluate_true_saa(lam, np.concatenate((true_demand_list_one,true_demand_list_two)), b, h, qbar, eval_demand_list)
                            common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'param': param, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                            helper.add_to_data(data, 'true_saa', 'true_cost', true_cost, **common_params)
                            helper.add_to_data(data, 'true_saa', 'minmax_cost', mm_cost, **common_params)


    # # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # # Save the DataFrame as a CSV file
    df.to_csv(file_name, index=False)  # index=False to avoid writing row numbers

    print(f"Data has been saved to {file_name}")