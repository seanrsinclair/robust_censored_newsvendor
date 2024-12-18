import numpy as np
import pandas as pd
import algorithms
import helper


DEBUG = False # Debug output
INCLUDE_TRUE = True # Include the True SAA solution


# Meta parameter specification for the simulations
qbar = 25
h = 1
num_iters = 100 # number of iterations for each simulation pair of (algo, N, distribution, etc)

# N_list = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_list = [500]

b_list = [3, 9, 49]
# b_list = [3]

rho_list = [b / (b+h) for b in b_list]

lam_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# lam_list = [15]

np.random.seed(5)


# Reading in the dataset 
df = pd.read_csv('./datasets/train.csv')
grouped_df = df.groupby(['Order Date', 'Category']).size().reset_index(name='sales') # Grouping by order date and category


def sample_dist(N, grouped_df, category): # Helper function which samples from the historical Kaggle dataset for a given category
    filtered_df = grouped_df[grouped_df['Category'] == category]

    if N > len(filtered_df):
        raise ValueError("Sample size cannot be greater than the number of available rows.")
    return filtered_df['sales'].sample(n=N, replace=True).reset_index(drop=True) # Samples from the data without replacement

for category in grouped_df['Category'].unique(): # Loops over each category
    data = [] # Empty dataset
    distribution_name = f'kaggle_sales_data_{category}' # File data name
    print(f'############# Category: {category} #############')

    gminus_calc_list = grouped_df[grouped_df['Category'] == category]['sales'].to_numpy() # Data used to estimate the true \Gminus(lambda)

    algorithm_list = [ # List of algorithms to evaluate
        ('ignorant_saa', algorithms.ignorant_saa),
        ('subsample_saa', algorithms.subsample_saa),
        ('robust', lambda lam, order_level_two, censored_one, censored_two, b, h: algorithms.robust_saa(lam, order_level_two, censored_one, censored_two, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05)),
        ('robust_plus', lambda lam, order_level_two, censored_one, censored_two, b, h: algorithms.robust_plus_saa(lam, order_level_two, censored_one, censored_two, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05)),
        ('robust_bonus', lambda lam, order_level_two, censored_one, censored_two, b, h: algorithms.robust_bonus_saa(lam, order_level_two, censored_one, censored_two, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05)),
        ('km', algorithms.km_estimate),
        ('joint_order_saa', algorithms.joint_order_saa)
    ]


    for b in b_list: # Loops over each parameter to run the simulation
        rho = b / (b+h)
        for lam in lam_list:
            for N in N_list:
                print(f'############# Running for N: {N}, b: {b}, lam: {lam} #############')
                if DEBUG: # DEBUG by outputting optimal vanilla and maxmin order quantity
                    print(f'Gminus(lambda): {np.mean(gminus_calc_list < lam)} and rho: {rho}')
                    qopt = helper.get_optimal_quantile(b, h, gminus_calc_list)
                    opt_cost = helper.get_newsvendor_cost(qopt, b, h, gminus_calc_list)
                    print(f'Empirical Optimal order: {qopt}, cost: {opt_cost} and mm_cost: {helper.get_robust_cost(qopt, b, h, lam, qbar, gminus_calc_list)}')

                    mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
                    mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
                    print(f'Optimal robust order: {mm_opt}, cost: {helper.get_newsvendor_cost(mm_opt, b, h, gminus_calc_list)} and mm_cost: {mm_cost}')

                for _ in range(num_iters): # Loops over each iteration
                    if (3*lam / 4) < 1:
                        order_level_two = 0.0
                    else:
                        order_level_two = float(np.random.randint(int((lam / 4)), int((3*lam / 4))))
                    # sample a dataset
                    true_demand_list_one = sample_dist(N, grouped_df, category)
                    true_demand_list_two = sample_dist(N, grouped_df, category)
                    eval_demand_list = gminus_calc_list

                    # censor the dataset by the two selling seasons
                    censored_demand_list_one = np.minimum(lam, true_demand_list_one)
                    censored_demand_list_two = np.minimum(order_level_two, true_demand_list_two)
                

                    for algo_name, algo_func in algorithm_list: # Loops over each algorithm and evaluates it
                        sol, true_cost, mm_cost = helper.evaluate_algorithm(
                            algo_name, algo_func, lam, order_level_two, censored_demand_list_one, censored_demand_list_two, b, h, qbar, eval_demand_list
                        )
                        
                        # Add true and minmax cost results to data
                        common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                        helper.add_to_data(data, algo_name, 'true_cost', true_cost, **common_params)
                        helper.add_to_data(data, algo_name, 'minmax_cost', mm_cost, **common_params)
                        
                    if INCLUDE_TRUE: # augmenting the two lists for the full distribution
                        sol, true_cost, mm_cost = helper.evaluate_true_saa(lam, np.concatenate((true_demand_list_one,true_demand_list_two)), b, h, qbar, eval_demand_list)
                        common_params = {'N': N, 'b': b, 'h': h, 'distribution': distribution_name, 'lam': lam, 'order_level_two': order_level_two, 'qbar': qbar}
                        helper.add_to_data(data, 'true_saa', 'true_cost', true_cost, **common_params)
                        helper.add_to_data(data, 'true_saa', 'minmax_cost', mm_cost, **common_params)

    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a CSV file
    df.to_csv(f'./data/{distribution_name}.csv', index=False)  # index=False to avoid writing row numbers

    print(f"Data has been saved to 'data/{distribution_name}.csv'")