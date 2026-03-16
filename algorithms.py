import numpy as np
import helper

from lifelines import KaplanMeierFitter

'''
Algorithm implementation for censored newsvendor problem. Each of the algorithms takes as input:
- order_levels : list of order levels (qoff_k)
- censored_demands : list of lists/arrays, each containing censored demands for the corresponding season
- b,h : cost parameters
'''


DEBUG = False
PRINT_EXTRA = False

def km_estimate(order_levels, censored_demands, b, h):
    '''
    Implements the KM estimate using the lifelines package over multiple seasons.

    Parameters
    ----------
    order_levels : list[float]
        Censoring/order levels for each selling season.
    censored_demands : list[np.ndarray]
        List of arrays/lists; the k-th entry contains samples censored at order_levels[k].
    b, h : float
        Cost parameters.
    '''
    # Validate inputs
    assert isinstance(order_levels, (list, tuple)) and isinstance(censored_demands, (list, tuple)), "order_levels and censored_demands must be lists/tuples"
    assert len(order_levels) == len(censored_demands), "order_levels and censored_demands must have the same length"
    assert len(order_levels) >= 1, "At least one season is required"

    lam = max(order_levels)

    # Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # Build pooled data and observed-event indicators per season
    pooled = []
    observed = []
    for qoff, data in zip(order_levels, censored_demands):
        data_arr = np.asarray(data)
        pooled.append(data_arr)
        observed.append((data_arr < qoff).astype(int))

    pooled = np.concatenate(pooled)
    observed = np.concatenate(observed)

    # Fit the KM estimator using pooled samples with per-sample censor flags
    kmf.fit(pooled, event_observed=observed)

    # Extract the cumulative distribution function (1 - survival function)
    cdf = 1 - kmf.survival_function_

    # Target quantile
    rho = b / (b + h)

    # Find the corresponding order level from the KM-estimated CDF
    order_level = cdf[cdf['KM_estimate'] >= rho].index.min()

    # Fallback to the largest order level if KM grid doesn't cross rho
    if np.isnan(order_level):
        return lam, 1
    idx = -1
    if order_level < lam:
        idx = 1
    return float(order_level), idx

def true_saa(uncensored_demands, b, h):
    '''
    Takes as input an uncensored dataset list/array, and returns the empirical rho-quantile from the uncensored data
    '''
    return np.quantile(uncensored_demands, b/(b+h))

def ignorant_saa(order_levels, censored_demands, b, h):
    '''
    Ignores censoring differences across seasons; returns the empirical rho-quantile from the pooled censored data.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1
    pooled_data = np.concatenate([np.asarray(x) for x in censored_demands])
    return np.quantile(pooled_data, b/(b+h)), 1

def subsample_saa(order_levels, censored_demands, b, h):
    '''
    Uses only uncensored samples from each season and returns the empirical rho-quantile.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1
    
    pools = []
    for qoff, data in zip(order_levels, censored_demands):
        arr = np.asarray(data)
        pools.append(arr[arr < qoff])
    filtered = np.concatenate([p for p in pools if len(p) > 0]) if any(len(p) > 0 for p in pools) else np.array([])
    if len(filtered) > 0:
        return np.quantile(filtered, b/(b+h)), 1
    else:
        return max(order_levels), 1


def joint_order_saa(order_levels, censored_demands, b, h):
    '''
    Fan, Chen, Zhou (2022) multi-season censored SAA.
    For k = 1..K (with q_off sorted ascending), define
        \hat F^k(x) = (1 / (\sum_{k'≥k} N_{k'})) * \sum_{k'≥k} \sum_{i=1}^{N_{k'}} 1{ s^{off}_{k',i} < x }.
    Terminate at the first k such that \hat F^k(q_off_k) ≥ ρ − 2β, where
        β = (min(b,h) / (18 (b+h))) * ε,  with  ε = 1/√N_total.
    Output \hat q = inf{ q : \hat F^k(q) ≥ ρ − 2β } based on the same pooled data (k'≥k).
    If no such k exists, output λ (largest order level).
    Returns (value, -1) for compatibility with downstream code.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    # Sort seasons by increasing order level and align data
    idx = np.argsort(order_levels)
    qoffs = [float(order_levels[i]) for i in idx]
    data_lists = [np.asarray(censored_demands[i]) for i in idx]

    K = len(qoffs)
    lam = qoffs[-1]


    # Parameters
    Ns = [len(x) for x in data_lists]
    N = Ns[-1] # number of data points at lambda
    eps = 1.0 / np.sqrt(N)
    beta = (min(b, h) / (18.0 * (b + h))) * eps
    rho = b / (b+h)
    target = max(0.0, rho - 2.0 * beta)


    idx = -1
    # Iterate k = 0..K-1 (1..K in paper). For each k, pool seasons k..K-1
    for k in range(K):
        pooled = np.concatenate(data_lists[k:]) if sum(Ns[k:]) > 0 else np.array([])
        if pooled.size == 0:
            continue
        qk = qoffs[k]
        Fk_at_qk = np.mean(pooled < qk)
        if Fk_at_qk >= target:
            # Return the (rho-2β)-quantile of this pooled distribution
            order_level = float(np.quantile(pooled, target))
            if order_level < lam:
                idx = 1
            return float(np.quantile(pooled, target)), idx

    # If still not enough mass, output lam
    return float(lam), -1




def robust_km(order_levels, censored_demands, b, h, qbar, Gminus, delta=0.3, conf_cons = 1):
    '''
    Robust KM algorithm.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    lam_idx = int(np.argmax(order_levels))
    lam = float(order_levels[lam_idx])
    rho = b / (b + h)

    data = np.asarray(censored_demands[lam_idx])
    N = len(data)

    Gminushat = np.mean(data < lam)

    conf_one = np.sqrt(np.log(2 / delta) / (2 * max(1, N)))
    conf_two = np.sqrt(1 / (4 * max(1, N) * delta))
    conf_three = np.sqrt((4 * Gminus * (1 - Gminus) * np.log(2 / delta)) / max(1, N)) + (4 * max(Gminus, 1 - Gminus) * np.log(2 / delta)) / (3 * max(1, N))

    conf_radius = conf_cons * min(conf_one, conf_two, conf_three)


    if Gminushat >= min(1.0, rho + conf_radius):
        # likely identifiable, fall back to KM algorithm
        return km_estimate(order_levels, censored_demands, b, h)[0], 1
    
    elif Gminushat < max(0.0, rho - conf_radius):
        return helper.get_q_crit(Gminushat, lam, b, h, qbar), -1
    else:
        return float(lam), 0






def robust_saa(order_levels, censored_demands, b, h, qbar, Gminus, delta=0.3, conf_cons = 1):
    '''
    Our baseline robust algorithm.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    lam_idx = int(np.argmax(order_levels))
    lam = float(order_levels[lam_idx])
    rho = b / (b + h)

    data = np.asarray(censored_demands[lam_idx])
    N = len(data)

    Gminushat = np.mean(data < lam)

    conf_one = np.sqrt(np.log(2 / delta) / (2 * max(1, N)))
    conf_two = np.sqrt(1 / (4 * max(1, N) * delta))
    conf_three = np.sqrt((4 * Gminus * (1 - Gminus) * np.log(2 / delta)) / max(1, N)) + (4 * max(Gminus, 1 - Gminus) * np.log(2 / delta)) / (3 * max(1, N))

    conf_radius = conf_cons * min(conf_one, conf_two, conf_three)

    if Gminushat >= min(1.0, rho + conf_radius):
        return float(np.quantile(data, rho, method="higher")), 1
    elif Gminushat < max(0.0, rho - conf_radius):
        return helper.get_q_crit(Gminushat, lam, b, h, qbar), -1
    else:
        return float(lam), 0


def robust_plus_saa(order_levels, censored_demands, b, h, qbar, Gminus, delta=0.3, conf_cons = 1, INCLUDE_LOG = True):
    '''
    Multi-season version of our robust algorithm.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    rho = b / (b + h)
    lam_idx = int(np.argmax(order_levels))
    lam = float(order_levels[lam_idx])
    lam_data = np.asarray(censored_demands[lam_idx])
    N = len(lam_data)

    K = len(order_levels)

    # Per-season empirical G_k^-(L)
    Ghats = []
    for qoff, data in zip(order_levels, censored_demands):
        arr = np.asarray(data)
        Ghats.append(np.mean(arr < qoff))
    if DEBUG: print(f'Ghats: {Ghats}')
    # Hoeffding confidence for all of the selling seasons

    if INCLUDE_LOG:
        conf = [
            conf_cons * np.sqrt(
                np.log((2*(K-1)) / delta) / (max(1, 2*len(censored_demands[k])))
            )
            for k in range(len(order_levels))
        ]
    else:
        conf = [
            conf_cons * np.sqrt(
                np.log(2 / delta) / (max(1, 2*len(censored_demands[k])))
            )
            for k in range(len(order_levels))
        ]

    conf[lam_idx] = conf_cons * np.sqrt( # no log term for the lam_idx
            np.log(2 / delta) / (max(1, 2*len(censored_demands[lam_idx])))
        )

    if DEBUG: print(f'Conf terms: {conf}')
    # U_est := { k : \hat G^-_k(qoff_k) >= rho + eps_k }
    Uest = [k for k in range(len(order_levels)) if Ghats[k] >= min(1.0, rho + conf[k])]
    if DEBUG: print(f'Uest: {Uest}')
    # If U_est nonempty, pool the corresponding censored samples and output the rho-quantile
    if len(Uest) > 0:
        pooled = np.concatenate([np.asarray(censored_demands[k]) for k in Uest])
        return float(np.quantile(pooled, rho, method="higher")), 1        


    # Otherwise checks if \Gminushat(lambda) < rho - conf(lambda)
    elif Ghats[lam_idx] < rho - conf[lam_idx]: #
        Gminushat = Ghats[lam_idx]
        return helper.get_q_crit(Gminushat, lam, b, h, qbar), -1

    else: # default to outputting lambda
        return float(lam), 0





def robust_well_separated_saa(order_levels, censored_demands, b, h, qbar, gamma, Gminus, delta=0.3):
    '''
    Single-season version for well-separated distributions.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    lam_idx = int(np.argmax(order_levels))
    lam = float(order_levels[lam_idx])
    rho = b / (b + h)

    data = np.asarray(censored_demands[lam_idx])
    N = len(data)

    Gminushat = np.mean(data < lam)

    conf_one = np.sqrt(np.log(2 / delta) / (2 * max(1, N)))
    conf_two = np.sqrt(1 / (4 * max(1, N) * delta))
    conf_three = np.sqrt((4 * Gminus * (1 - Gminus) * np.log(2 / delta)) / max(1, N)) + (4 * max(Gminus, 1 - Gminus) * np.log(2 / delta)) / (3 * max(1, N))

    conf_radius = min(conf_one, conf_two, conf_three)

    if Gminushat >= min(1.0, rho + conf_radius):
        return float(np.quantile(data, rho)), 1
    elif Gminushat < max(0.0, rho - conf_radius):
        qcrit = helper.get_well_separated_q_crit(Gminushat, lam, b, h, qbar, gamma)
        return qcrit, -1
    else:
        return float(lam), 0



def robust_plus_well_separated_saa(order_levels, censored_demands, b, h, qbar, gamma, Gminus, delta=0.3):
    '''
    Single-season version for well-separated distributions with the additional checks.
    '''
    assert len(order_levels) == len(censored_demands) and len(order_levels) >= 1

    rho = b / (b + h)
    lam_idx = int(np.argmax(order_levels))
    lam = float(order_levels[lam_idx])

    lam_data = np.asarray(censored_demands[lam_idx])
    N = len(lam_data)

    # Per-season empirical G^-_k(qoff)
    Ghats = []
    for qoff, data in zip(order_levels, censored_demands):
        arr = np.asarray(data)
        Ghats.append(np.mean(arr < qoff))

    conf_one = np.sqrt(np.log(2 / delta) / (2 * max(1, N)))
    conf_two = np.sqrt(1 / (4 * max(1, N) * delta))
    conf_three = np.sqrt((4 * Gminus * (1 - Gminus) * np.log(2 / delta)) / max(1, N)) + (4 * max(Gminus, 1 - Gminus) * np.log(2 / delta)) / (3 * max(1, N))
    conf_radius = min(conf_one, conf_two, conf_three)

    Ghat_lam = Ghats[lam_idx]

    if Ghat_lam >= min(1.0, rho + conf_radius):
        return float(np.quantile(lam_data, rho)), 1

    # Extra-identifiability checks across seasons
    if (gamma * lam >= rho) or any(Ghats[k] >= rho + conf_radius - gamma * (lam - order_levels[k]) for k in range(len(order_levels))):
        return float(np.quantile(lam_data, rho)), 1

    if Ghat_lam > rho:
        qghat = float(np.quantile(lam_data, rho))
        if (lam - qghat) >= (conf_radius / gamma):
            return qghat, 1
        else:
            return float(lam), 0

    if Ghat_lam < max(0.0, rho - conf_radius):
        qcrit = helper.get_well_separated_q_crit(Ghat_lam, lam, b, h, qbar, gamma)
        return qcrit, -1

    return float(lam), 0