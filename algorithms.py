import numpy as np
import helper

from lifelines import KaplanMeierFitter

'''
Algorithm implementation for censored newsvendor problem. Each of the algorithms takes as input:
- order level (lambda)
- censored demands: list of [min(demand, order)]
- b,h : cost parameters
- bonus_demands_list: list of [min(demand, LAM_CONS*order)] for the second selling season
'''


DEBUG = False

def km_estimate(order_level_one, order_level_two, censored_demands_one, censored_demands_two, b, h):
    '''
    Implements the KM estimate using the lifelines package
    '''
    lam = order_level_one

    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()
    
    event_one = [1 if di < order_level_one else 0 for di in censored_demands_one] # gets the events for if the demand was uncensored
    event_two = [1 if di < order_level_two else 0 for di in censored_demands_two] # repeats for the bonus sales data

    # Fit the data using the KM estimator
    kmf.fit(np.concatenate((censored_demands_one, censored_demands_two)), event_observed=np.concatenate((event_one, event_two)))
    
    # Extract the cumulative distribution function (1 - survival function)
    cdf = 1 - kmf.survival_function_

    # Find the optimal order quantile (cost ratio b / (b + h))
    rho = b / (b + h)

    # Find the corresponding order level from the KM-estimated CDF
    order_level = cdf[cdf['KM_estimate'] >= rho].index.min()

    # Check if order_level is NaN, and return lam if true
    if np.isnan(order_level):
        return lam
    
    return order_level

def true_saa(uncensored_demands, b, h):
    '''
    Takes as input an uncensored dataset list, and returns the empirical rho-quantile form the uncensored data
    '''
    return np.quantile(uncensored_demands, b/(b+h))

def ignorant_saa(order_level_one, order_level_two, censored_demands_one, censored_demands_two, b, h):
    '''
    Ignores the potential impact of censoring and returns the empirical rho-quantile form the censored data
    '''
    return np.quantile(np.concatenate((censored_demands_one, censored_demands_two)), b/(b+h))

def subsample_saa(order_level_one, order_level_two, censored_demands_one, censored_demands_two, b, h):
    '''
    Subsamples the dataset to only contain samples which are uncensored, and returns the empirical rho-quantile
    '''

    filtered_demands_one = [demand for demand in censored_demands_one if demand < order_level_one]
    filtered_demands_two = [demand for demand in censored_demands_two if demand < order_level_two]
    filtered_demands = np.concatenate((filtered_demands_one, filtered_demands_two))
    if len(filtered_demands) > 0:
        return np.quantile(filtered_demands, b/(b+h))
    else: # if no uncensored samples just output lambda
        return order_level_one

def joint_order_saa(order_level_one, order_level_two, censored_demands_one, censored_demands_two, b, h):
    '''
    Implements the Fan, Chen, Zhou (2022) algorithm with two selling seasons
        order_level_one: lambda, the maximum selling season
        order_level_two: second selling season <= lambda
        censored_demands_one: demand list censored at order_level_one
        censored_demands_two: demand list censored at order_leve_two
    '''
    # print(f'Running for: {order_level_two, order_level_one}')
    # Constants beta with epsilon = 1 / sqrt(N)
    beta = (min(b, h) / (18 * (h + b))) * (1 / np.sqrt(len(censored_demands_one)))
    rho = b / (b+h)
    actual_rho = max(0, rho- 2 * beta)
    lam = order_level_one
    N = len(censored_demands_one)

    # First we combine both datasets to estimate P(D <= order_level_two).
    combined_data = np.concatenate((censored_demands_one, censored_demands_two))

    p_1 = np.mean(combined_data < order_level_two) # calculates probability mass up to the first selling season
    if p_1 >= actual_rho: # then we know that the empirical rho-quantile is in [0, order_level_two], so just combine output quantile
        return np.quantile(combined_data, actual_rho)
    else:
        # now we need to augment with an estimate of P(order_level_two < D <= order_level_one)
        p_2 = np.mean((order_level_two <= censored_demands_one) & (censored_demands_one < order_level_one))
        if p_1 + p_2 >= actual_rho: # solution is within [order_level_two, order_level_one)
            remaining_data = np.asarray([di for di in censored_demands_one if di >= order_level_two])   
            # sol_1 = np.quantile(remaining_data, (N / len(remaining_data))*(actual_rho - p_1))
            # return sol_1
            # # print(f'Potential sol_1: {sol_1}')
            sorted_data = np.sort(np.unique(remaining_data))
            # print(f'Checking values: {sorted_data}')
            for q in sorted_data:
                remaining_mean = np.mean((order_level_two <= censored_demands_one) & (censored_demands_one <= q))
                if p_1 + remaining_mean >= actual_rho: # TODO: Running into setting where this case is never visited
                    # print(f'Output solution: {q}')
                    return float(q)
            # print(f'For some reason not finding a q value which satisfies?')
        else: # we still have not seen \rho mass, so let us just output lambda
            # print(f'Did not find sufficient mass, output: {order_level_one}')
            return float(order_level_one)

def robust_saa(order_level_one, order_level_two, censored_demands_one, censored_demands_two, b, h, qbar, Gminus, delta=0.3):
    '''
    Implements our algorithm

    Note: Added some "sandwhiching" to ensure the confidence interval estimates for Gminus are in [0,1]
    '''
    lam = order_level_one
    rho = b / (b+h)
    N = len(censored_demands_one)


    Gminushat = np.mean(censored_demands_one < lam)

    # Tests out three different confidence intervals to determine if one is tighter than the other
    conf_one = np.sqrt(np.log(2 / delta) / (2*N))
    conf_two = np.sqrt(1 / (4*N * delta))
    conf_three = np.sqrt((4*Gminus*(1 - Gminus) *np.log(2 / delta))/N) + (4*max(Gminus, 1 - Gminus)*np.log(2 / delta))/(3*N)

    if DEBUG:
        print(f'Bernstein: {conf_three}, Chernoff: {conf_two}, Hoeffding: {conf_one}')
        if conf_three <= min(conf_one, conf_two):
            print(f'Bernstein Smallest')
        elif conf_two <= min(conf_one, conf_three):
            print(f'Chernoff Smallest!')
        else:
            print(f'Hoeffding Smallest!')

    conf_radius = min(conf_one, conf_two, conf_three) # taking minimum of Hoeffding, Bernstein, and Chernoff

    if DEBUG: print(f'Gminushat: {Gminushat}, and confidence radius: {conf_radius}')
    if Gminushat >= np.minimum(1,rho + conf_radius): # Strictly identifiable regime
        if DEBUG: print(f'Outputting SAA Solution')
        return np.quantile(censored_demands_one, rho)
    elif Gminushat < np.maximum(0, rho - conf_radius): # Striclty unidentifiable regime
        if DEBUG: print(f'Outputting QCrit')
        return helper.get_q_crit(Gminushat, lam, b, h, qbar)
    else: # Knife-edge, outputting lambda
        if DEBUG: print(f'Outputting lam')
        return lam
    


'''
Implementing a version of the AIM and the Burnetas-Smith algorithm, but have poor performance since we have offline censored data
instead of able to collect data adaptively online
'''
def aim(order_level, censored_demands, b, h):
    order = order_level

    t = 0
    for d in censored_demands:
        t += 1
        epsilon = 10 / (max(b,h) * np.sqrt(t))
        if d < order_level:
            order = max(0, order - epsilon * h)
        else:
            order += epsilon * b
    return order

def burnetas_smith(order_level, censored_demands, b, h):
    order = order_level

    t = 0
    for d in censored_demands:
        t += 1
        if d < order_level:
            order *= (1 - (h / ((b+h)*t)))
        else:
            order *= (1 + (h / ((b+h)*t)))
    return order
