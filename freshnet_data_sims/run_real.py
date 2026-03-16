import numpy as np
import pandas as pd


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms
import helper

import freshnet_helper

import matplotlib.pyplot as plt

"""
run_real.py

Purpose
-------
Run algorithm evaluations on real-world FreshRetailNet product data (censored
demands + observed order levels) and save results to:
    ./data/freshnet_algo_data.csv

Workflow (per product)
----------------------
For each product_id in freshnet_helper.PRODUCT_LIST:
1) Load problem parameters and training data:
     qbar_list, b_list, rho_list, order_levels, censored_demands
   Set lam := max(order_levels).
2) For each qbar in qbar_list:
   - Sample a large evaluation demand list from the product’s KM CDF
     (freshnet_helper.sample_from_km_cdf, dataset='eval').
   - Build a list of algorithms (SAA baselines, KM, robust variants).
3) For each b in b_list:
   - Compute reference quantities from evaluation data:
       qopt_G (oracle quantile), qrisk_G (robust-optimal), Delta (robust cost at qrisk_G)
   - Evaluate each algorithm via helper.evaluate_algorithm(...) to get:
       (qalg, true_cost, minmax_cost, predict_id)
   - Add additional fixed policies (rho*qbar, lam, qopt_G, qrisk_G) for comparison.
4) Append all results in long format and write a single CSV.

Recorded metrics
----------------
For each (product, qbar, b, algorithm) the script stores rows with:
  - metric='true_cost'   : expected newsvendor cost on eval demand sample
  - metric='minmax_cost' : robust/min-max cost under (lam, qbar)
  - metric='predict_id'  : {'identifiable','knife_edge','unidentifiable'}

Common parameters saved with every row include:
  b, h, product, lam, qbar, identifiable (ground-truth via Gminus(lam) >= rho),
  Gminuslam, qalg, qrisk_G, Delta, qopt_G, qopt_cost.

Output
------
Creates ./data/freshnet_algo_data.csv (long format), used by downstream tables/plots.
"""

np.random.seed(1)

DEBUG = False
INCLUDE_TRUE = True

# Meta parameter specification for the simulations

product_list = freshnet_helper.PRODUCT_LIST

h = 1

np.random.seed(5)

# Loops over each distribution

data = []

scale_list = [1, 2, 3, 4, 5]

for product_id in product_list:
    if DEBUG: print(f'#### RUNNING FOR: {product_id} ####')
 

    file_name = f'./data/freshnet_algo_data_updated_tune.csv'

    qbar_list, b_list, rho_list, order_levels, censored_demands = freshnet_helper.get_parameters(product_id) # gets the evaluation parameters


    lam = max(order_levels)
    if DEBUG:
        print(f'Training Data')
        print(f'Order Levels: {order_levels}')
        print(censored_demands)

    # specified for that distributio


    for qbar in qbar_list: # Loop over each parameter


        gminus_calc_list = freshnet_helper.sample_from_km_cdf(num_samples = int(1e7), qbar = qbar, product_id=product_id, dataset='eval') # data used for sampling distribution
        # print(f'Demand samples from evaluation data: {gminus_calc_list}')
        if DEBUG: print(f"Mean: {np.mean(gminus_calc_list):.4f}, Variance: {np.var(gminus_calc_list):.4f}, Max: {np.max(gminus_calc_list):.4f}, Min: {np.min(gminus_calc_list):.4f}")

        # list of algorithms to evaluate
        algorithm_list = [
            ('km', algorithms.km_estimate),
            ('joint_order_saa', algorithms.joint_order_saa),
        ]

        algorithm_list += [
            ('robust_plus_'+str(k), lambda order_levels, censored_demands, b, h: algorithms.robust_plus_saa(order_levels, censored_demands, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05, conf_cons=1/k)) for k in scale_list
        ]

        algorithm_list += [
            ('robust_'+str(k), lambda order_levels, censored_demands, b, h: algorithms.robust_saa(order_levels, censored_demands, b, h, qbar, np.mean(gminus_calc_list < lam), delta=0.05, conf_cons=1/k)) for k in scale_list
        ]



        for b in b_list:

            rho = b / (b+h)
            if DEBUG: print(f'############# Running for product: {product_id} b: {b}, lam: {lam} and qbar: {qbar} #############')
            if DEBUG: # Output vanilla optimal and robust optimal order
                print(f'Gminus(lambda): {np.mean(gminus_calc_list < lam)} and rho: {rho}')
                qopt = helper.get_optimal_quantile(b, h, gminus_calc_list)
                opt_cost = helper.get_newsvendor_cost(qopt, b, h, gminus_calc_list)
                print(f'Empirical Optimal order: {qopt}, cost: {opt_cost} and mm_cost: {helper.get_robust_cost(qopt, b, h, lam, qbar, gminus_calc_list)}')

                mm_opt = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
                mm_cost = helper.get_robust_cost(mm_opt, b, h, lam, qbar, gminus_calc_list)
                print(f'Optimal robust order: {mm_opt}, cost: {helper.get_newsvendor_cost(mm_opt, b, h, gminus_calc_list)} and mm_cost: {mm_cost}')
                print(f'Costs for lam: {helper.get_newsvendor_cost(lam, b, h, gminus_calc_list)} and mm cost: {helper.get_robust_cost(lam, b, h, lam, qbar, gminus_calc_list)}')
            
            # Evaluates qrisk_G and \Delta

            # Adding on the cost of the qrisk solution
            qrisk_G = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
            Delta = helper.get_robust_cost(qrisk_G, b, h, lam, qbar, gminus_calc_list)


            # Adding on the cost of the qopt_G solution
            qopt_G = helper.get_optimal_quantile(b, h, gminus_calc_list)
            qopt_cost = helper.get_newsvendor_cost(qopt_G, b, h, gminus_calc_list)


            for algo_name, algo_func in algorithm_list: # Loops over each algorithm and evaluates it
                sol, true_cost, mm_cost, ind = helper.evaluate_algorithm(
                    algo_name, algo_func, order_levels, censored_demands, b, h, qbar, gminus_calc_list
                )
                if DEBUG: print(f'Algo: {algo_name}, sol: {sol}, costs: {(true_cost, mm_cost)}')
                identifiable = 1 if np.mean(gminus_calc_list < lam) >= rho else 0
                Gminuslam = np.mean(gminus_calc_list < lam)
                # Add true and minmax cost results to data
                common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar, 'identifiable':identifiable, 'Gminuslam': Gminuslam, 'qalg': sol, 'qrisk_G': qrisk_G, 'Delta': Delta, 'qopt_G': qopt_G, 'qopt_cost': qopt_cost}
                helper.add_to_data(data, algo_name, 'true_cost', true_cost, **common_params)
                helper.add_to_data(data, algo_name, 'minmax_cost', mm_cost, **common_params)
                if ind == 1:
                    helper.add_to_data(data, algo_name, 'predict_id', 'identifiable', **common_params)
                elif ind == 0:
                    helper.add_to_data(data, algo_name, 'predict_id', 'knife_edge', **common_params)
                else:
                    helper.add_to_data(data, algo_name, 'predict_id', 'unidentifiable', **common_params)

            # Adding on the cost of the robust solution: \rho M
            robust_sol = rho*qbar
            news_cost = helper.get_newsvendor_cost(robust_sol, b, h, gminus_calc_list)
            mm_cost = helper.get_robust_cost(robust_sol, b, h, lam, qbar, gminus_calc_list)
            common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar, 'identifiable':identifiable, 'Gminuslam': Gminuslam, 'qalg': robust_sol, 'qrisk_G': qrisk_G, 'Delta': Delta, 'qopt_G': qopt_G, 'qopt_cost': qopt_cost}
            helper.add_to_data(data, 'robust_sol', 'true_cost', news_cost, **common_params)
            helper.add_to_data(data, 'robust_sol', 'minmax_cost', mm_cost, **common_params)


            # Adding on the cost of the lam solution
            lam_sol = lam
            news_cost = helper.get_newsvendor_cost(lam_sol, b, h, gminus_calc_list)
            mm_cost = helper.get_robust_cost(lam_sol, b, h, lam, qbar, gminus_calc_list)
            common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar, 'identifiable':identifiable, 'Gminuslam': Gminuslam, 'qalg': lam, 'qrisk_G': qrisk_G, 'Delta': Delta, 'qopt_G': qopt_G, 'qopt_cost': qopt_cost}
            helper.add_to_data(data, 'lam_sol', 'true_cost', news_cost, **common_params)
            helper.add_to_data(data, 'lam_sol', 'minmax_cost', mm_cost, **common_params)

            # Adding on the cost of the qopt_G solution
            qopt_sol = helper.get_optimal_quantile(b, h, gminus_calc_list)
            news_cost = helper.get_newsvendor_cost(qopt_sol, b, h, gminus_calc_list)
            mm_cost = helper.get_robust_cost(qopt_sol, b, h, lam, qbar, gminus_calc_list)
            common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar, 'identifiable':identifiable, 'Gminuslam': Gminuslam, 'qalg': qopt_sol, 'qrisk_G': qrisk_G, 'Delta': Delta, 'qopt_G': qopt_G, 'qopt_cost': qopt_cost}
            helper.add_to_data(data, 'qopt_G', 'true_cost', news_cost, **common_params)
            helper.add_to_data(data, 'qopt_G', 'minmax_cost', mm_cost, **common_params)

            # Adding on the cost of the qrisk solution
            qrisk_sol = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)
            news_cost = helper.get_newsvendor_cost(qrisk_sol, b, h, gminus_calc_list)
            mm_cost = helper.get_robust_cost(qrisk_sol, b, h, lam, qbar, gminus_calc_list)
            common_params = {'b': b, 'h': h, 'product': product_id, 'lam': lam, 'qbar': qbar, 'identifiable':identifiable, 'Gminuslam': Gminuslam, 'qalg': qrisk_sol, 'qrisk_G': qrisk_G, 'Delta': Delta, 'qopt_G': qopt_G, 'qopt_cost': qopt_cost}
            helper.add_to_data(data, 'qrisk_G', 'true_cost', news_cost, **common_params)
            helper.add_to_data(data, 'qrisk_G', 'minmax_cost', mm_cost, **common_params)


# # Convert list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# # Save the DataFrame as a CSV file
df.to_csv(file_name, index=False)  # index=False to avoid writing row numbers

print(f"Data has been saved to {file_name}")
