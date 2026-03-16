"""
This module extends the core functionality provided in helper.py by reusing its functions
and adding additional utilities specific to the FreshRetailNet dataset.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper import *

from lifelines import KaplanMeierFitter

import pandas as pd
import numpy as np

import bisect


df = pd.read_csv(f'./datasets/train.csv')
product_list = df['product_id'].unique()


BUCKET_STYLE = 'closest'


PRODUCT_LIST = product_list
# MULT_FACTORS = [1.05, 1.5, 2., 2.5, 3., 4., 5.]
MULT_FACTORS = [2.5]


DEBUG = False

# --- Extended FreshRetailNet utilities ---

def safe_max(lst):
    return max(lst) if lst else None


def get_parameters(product_id):

    if DEBUG: print(f'Getting the parameters!')
    order_levels, censored_demands = get_censored_sales_data(product_id, dataset='train')

    eval_order_levels, eval_demands = get_censored_sales_data(product_id, dataset='eval')

    if DEBUG: print(f'Order levels: {order_levels}')
    if DEBUG: print(f'Demands: {censored_demands}')
    b_list = [3, 9, 49]
    h = 1
    rho_list = [b / (b+h) for b in b_list]
    
    # safe max calls
    max_train = safe_max(order_levels)
    max_eval = safe_max(eval_order_levels)

    if max_train is None and max_eval is None:
        raise ValueError(f"No order-level data found for product {product_id}")

    base_qbar = max(x for x in [max_train, max_eval] if x is not None)
    qbar_list = [base_qbar * MULT_FACTORS[i] for i in range(len(MULT_FACTORS))]

    return qbar_list, b_list, rho_list, order_levels, censored_demands




def get_km_cdf(product_id, qbar, dataset='eval'):


    # Kaplan–Meier CDF estimator using lifelines

    # Load data and filter by product
    df = pd.read_csv(f'./datasets/{dataset}.csv')
    df = df[df['product_id'] == product_id].copy()

    # If no data for this product, return a trivial CDF
    if df.empty:
        return lambda x: 1.0

    # Observed times: min(D_i, q_i) stored in s_t
    times = df['s_t'].to_numpy(dtype=float)

    # Event indicator: 1 if uncensored (censored == 0), 0 if censored (censored == 1)
    events = (df['censored'] == 0).to_numpy()


    # Observed times: min(D_i, q_i) stored in s_t
    times = df['s_t']

    # Event indicator: 1 if uncensored (censored == 0), 0 if censored (censored == 1)
    events = (df['censored'] == 0).to_numpy()

    # Fit KM
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)

    # lifelines KM cumulative density is the CDF: F(t) = P(D ≤ t)
    cdf = kmf.cumulative_density_.reset_index()

    # Sort by timeline just in case
    cdf = cdf.sort_values('timeline').reset_index(drop=True)



    def get_eval_cdf(x):
        x = float(x)
        # lifelines KM returns survival S(x); CDF = 1 - S(x)
        Sx = float(kmf.predict(x))
        return 1.0 - Sx

    return get_eval_cdf    



def get_cdf(product_id, dataset='eval', method = 'km'):
    """
    Construct an empirical CDF estimator for a given product in the dataset.

    Parameters
    ----------
    product_id : int or str
        Product identifier to filter the evaluation dataset.

    Returns
    -------
    get_eval_cdf : Callable[[float], float]
        Function that takes x >= 0 and returns the empirical estimate of G(x)
        using all censored demand observations with order level ≥ x.
    """


    if method == 'pooling_original':
        # Retrieve order levels and censored demand observations
        order_levels, censored_demands = get_censored_sales_data(product_id, dataset)

        # Edge case: if no data, return a dummy CDF
        if len(order_levels) == 0:
            return lambda x: 0.0


        def get_eval_cdf(x):
            """
            Empirical CDF estimate of G(x) = P(D ≤ x).

            We use all censored demand observations up to x. If data are censored,
            these represent min(D, q), so the estimate is conservative below the
            smallest order level but accurate above observed censored thresholds.
            """

            pooled_data = []
            i = 0
            for obs_list in censored_demands:
                if order_levels[i] > x:
                    pooled_data.extend(obs_list)
                i += 1
            pooled_data = np.array(pooled_data)
            if len(pooled_data) > 0:
                return np.mean(pooled_data <= x)
            else:
                return 1
        return get_eval_cdf
    

    if method == 'pooling_monotone':
        # Retrieve order levels and censored demand observations
        order_levels, censored_demands = get_censored_sales_data(product_id, dataset)

        # Edge case: if no data, return a dummy CDF
        if len(order_levels) == 0:
            return lambda x: 0.0

        def estimate_cdf(x):
            """
            Empirical CDF estimate of G(x) = P(D ≤ x).

            We use all censored demand observations up to x. If data are censored,
            these represent min(D, q), so the estimate is conservative below the
            smallest order level but accurate above observed censored thresholds.
            """
            pooled_data = []
            i = 0
            for obs_list in censored_demands:
                if order_levels[i] > x:
                    pooled_data.extend(obs_list)
                i += 1
            pooled_data = np.array(pooled_data)
            if len(pooled_data) > 0:
                return np.mean(pooled_data <= x)
            else:
                return 1

        # Compute the raw CDF values on the discrete grid of order levels
        # using the original pooling rule, then enforce monotonicity via a
        # running maximum over this grid.
        raw_support = np.array([estimate_cdf(L) for L in order_levels])
        print(f'Raw support: {raw_support}')
        # Enforce monotonicity: G_mono(L_k) = max_{j ≤ k} G_raw(L_j)
        mono_support = np.maximum.accumulate(raw_support)
        print(f'Monotone support: {mono_support}')
        def get_monotone_cdf(x):
            """
            Monotone empirical CDF estimate of G(x) = P(D ≤ x).

            We first compute the raw CDF at each order level using the
            original pooling rule and then take the running maximum over
            these support points to enforce monotonicity. For a general
            x, we return the value at the largest order level L ≤ x.
            """
            x = float(x)

            # If x is to the left of all order levels, there is no grid
            # point L ≤ x; in the discretized version we take G(x) = 0.
            if x < order_levels[0]:
                return 0.0

            # Find the index of the rightmost order level ≤ x
            idx = np.searchsorted(order_levels, x, side='right') - 1
            if idx < 0:
                return 0.0
            if idx >= len(order_levels):
                idx = len(order_levels) - 1

            return mono_support[idx]

        return get_monotone_cdf

    
    elif method == 'uncensored':
        uncensored_demands = get_uncensored_data(product_id, dataset)

        def get_eval_cdf(x):
            if len(uncensored_demands) == 0:
                return 1
            return np.mean(uncensored_demands <= x)
        return get_eval_cdf
    
    elif method == 'km':
        # Kaplan–Meier CDF estimator using lifelines

        # Load data and filter by product
        df = pd.read_csv(f'./datasets/{dataset}.csv')
        df = df[df['product_id'] == product_id].copy()

        # If no data for this product, return a trivial CDF
        if df.empty:
            return lambda x: 1.0

        # Observed times: min(D_i, q_i) stored in s_t
        times = df['s_t'].to_numpy(dtype=float)

        # Event indicator: 1 if uncensored (censored == 0), 0 if censored (censored == 1)
        events = (df['censored'] == 0).to_numpy()

        # Fit KM
        kmf = KaplanMeierFitter()
        kmf.fit(times, event_observed=events)

        def get_eval_cdf(x):
            x = float(x)
            # lifelines KM returns survival S(x); CDF = 1 - S(x)
            Sx = float(kmf.predict(x))
            return 1.0 - Sx

        return get_eval_cdf    




def sample_from_km_cdf(num_samples, qbar, product_id, dataset='eval'):
    """
    Sample from the (estimated) demand distribution implied by the
    Kaplan–Meier CDF for a given product.

    We treat the KM CDF as a discrete distribution supported on its
    time grid (the 'timeline' in lifelines), form the associated PMF,
    and then draw `num_samples` i.i.d. samples from that PMF.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    product_id : int or str
        Product identifier to filter the dataset.
    dataset : str, default 'eval'
        CSV filename (without extension) to read from, e.g., 'train' or 'eval'.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) of sampled demand values.
    """
    
    # Load data and filter by product
    df = pd.read_csv(f'./datasets/{dataset}.csv')
    df = df[df['product_id'] == product_id].copy()

    # If no data for this product, return zeros
    if df.empty:
        return np.zeros(num_samples)

    # Observed times: min(D_i, q_i) stored in s_t
    times = df['s_t']

    # Event indicator: 1 if uncensored (censored == 0), 0 if censored (censored == 1)
    events = (df['censored'] == 0).to_numpy()

    # Fit KM
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)

    # lifelines KM cumulative density is the CDF: F(t) = P(D ≤ t)
    cdf = kmf.cumulative_density_.reset_index()

    # Ensure the CDF reaches probability 1 by appending a large-time point if needed.
    if cdf['KM_estimate'].max() < 1.0:
        if DEBUG:
            print(f'Augmenting CDF so it reaches one at: {qbar}')
            print(f"{cdf['KM_estimate'].max()}")
        new_row = pd.DataFrame([{
            "timeline": qbar,
            "KM_estimate": 1.0
        }])
        cdf = pd.concat([cdf, new_row], ignore_index=True)

    # Sort by timeline just in case
    cdf = cdf.sort_values('timeline').reset_index(drop=True)

    # Convert the CDF to a discrete PMF on the KM time grid:
    # pmf(t_i) = F(t_i) - F(t_{i-1})
    cdf['pmf'] = cdf['KM_estimate'].diff().fillna(cdf['KM_estimate'])

    # Numerical guard: clip small negatives and renormalize
    pmf = cdf['pmf'].to_numpy()
    if DEBUG:
        print(f'KM values: {cdf["timeline"].to_numpy()}')
        print(f'KM PMF: {pmf}')

    # Draw samples directly from the PMF representation
    samples = np.random.choice(
        a=cdf['timeline'].to_numpy(),
        size=num_samples,
        p=pmf
    )

    return samples


def sample_from_true_cdf(cdf_fun, num_samples=10_000, tol=1e-3, max_x=None, max_iter=60, rng=np.random.default_rng()):
    """
    CODE FROM CHATGPT:

    Draw samples from a distribution with CDF `cdf_fun` using inverse transform sampling.

    Parameters
    ----------
    cdf_fun: Estimate of the CDF function
    num_samples : int, default 10k
    tol : float, relative tolerance for inverse search
    max_x : float or None, optional hard cap for the search domain. If provided and F(max_x) < u for some u, the returned value is max_x.
    max_iter : int, default 60
        Maximum number of bisection iterations per sample.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples,) of samples approximately distributed according to `cdf_fun`.
    """

    n = int(num_samples)

    np.random.default_rng()

    
    u = rng.random(n)  # draws n samples from U(0,1)
    samples = np.empty(n, dtype=float)

    def inverse_cdf(p: float) -> float:
        # Handle edge cases fast
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            # Find a finite x with F(x) ≈ 1 if possible
            lo, hi = 0.0, 1.0
            # expand hi until CDF(hi) ≥ p or we hit max_x
            while True:
                Fhi = float(cdf_fun(hi))
                if Fhi >= p:
                    break
                if max_x is not None and hi >= max_x:
                    return max_x
                hi *= 2.0
                if hi > 1e12 and max_x is None:  # safety valve
                    break
            # fall through to bisection with p≈1
        else:
            lo, hi = 0.0, 1.0

        # Exponential search to find an upper bound with F(hi) ≥ p
        while True:
            Fhi = float(cdf_fun(hi))
            if Fhi >= p:
                break
            if max_x is not None and hi >= max_x:
                # If even at max_x we haven't reached p, clamp
                return max_x
            lo, hi = hi, hi * 2.0
            if hi > 1e12 and max_x is None:  # safety valve to avoid infinite search
                break

        # Bisection within [lo, hi]
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            Fmid = float(cdf_fun(mid))
            if Fmid >= p:
                hi = mid
            else:
                lo = mid
            if (hi - lo) <= tol * (1.0 + hi):
                break
        return hi

    # Compute inverse per u
    for i, p in enumerate(u):
        samples[i] = inverse_cdf(float(p))

    return samples



def get_censored_sales_data(product_id, dataset='train', bucket_style = BUCKET_STYLE):
    """
    Build (order_levels, censored_demands) for a given product.

    Parameters
    ----------
    product_id : int or str
        Product identifier to filter the dataset.
    dataset : str, default 'train'
        CSV filename (without extension) to read from, e.g., 'train' or 'eval'.

    Returns
    -------
    order_levels : np.ndarray
        Sorted unique order levels (q_t) used as buckets.
    censored_demands : list[np.ndarray]
        For each order level L in order_levels, an array of observed demands censored at L.
        - If censored == 1 (stockout), the observed value is L (i.e., min(d, L) = L).
        - If censored == 0, the observed value is d_t (uncensored).
        All uncensored rows are bucketed into the *maximum order level* bucket. If no order levels exist yet, a single bucket is created at the maximum uncensored demand.
    """

    # Load data and filter by product
    df = pd.read_csv(f'./datasets/{dataset}.csv')


    df = df[df['product_id'] == product_id].copy()
   
    if df.empty:
        return [], []

    censored_df = df[df['censored'] == 1]
    uncensored_df = df[df['censored'] == 0]


    # Start with order levels observed in censored rows
    order_levels = sorted(censored_df['s_t'].dropna().unique().tolist())

    # Initialize mapping level -> list of demands
    buckets = {L: [] for L in order_levels}

    def ensure_level(L):
        if L not in buckets:
            buckets[L] = []
            order_levels.append(L)
            order_levels.sort()

    # Add censored observations: observed value = L
    for _, row in censored_df.iterrows():
        L = row['s_t']
        ensure_level(L)
        buckets[L].append(L)

    if BUCKET_STYLE == 'max':
        # Assign ALL uncensored samples to the maximum order level bucket

        # Determine the max order level; if no levels yet, create one at max uncensored demand
        max_level = max(order_levels) if order_levels else None

        max_uncensored_val = uncensored_df['s_t'].max()
        if pd.isna(max_uncensored_val):
            max_uncensored = 0.0
        else:
            max_uncensored = max_uncensored_val

        if max_level is None or max_uncensored > max_level:
            ensure_level(max_uncensored)


        max_level = max(order_levels)


        for _, row in uncensored_df.iterrows():
            sales_obs = row['s_t']
            # Put every uncensored observation into the max-level bucket
            buckets[max_level].append(sales_obs)

    elif BUCKET_STYLE == 'closest':
        # Assign uncensored samples to the smallest bucket for which order_level >= sales_obs.

        if not uncensored_df.empty:
            # Compute maximum uncensored and censored levels
            max_uncensored_val = uncensored_df['s_t'].max()
            max_censored = max(order_levels) if order_levels else None

            # Case 1: no censored levels at all — create a single bucket at max uncensored
            if max_censored is None:
                if pd.isna(max_uncensored_val):
                    # No meaningful uncensored value either; fall back to returning empty
                    return [], []
                ensure_level(max_uncensored_val)
                # Assign all uncensored observations to this single bucket
                for _, row in uncensored_df.iterrows():
                    sales_obs = row['s_t']
                    buckets[max_uncensored_val].append(sales_obs)

            else:
                # There is at least one censored level
                if max_uncensored_val > max_censored:
                    ensure_level(max_uncensored_val)
                    max_bucket_level = max_uncensored_val
                    # Recompute max_censored since order_levels may have been updated
                    max_censored = max(order_levels)

                # Assign each uncensored observation
                order_levels.sort()
                for _, row in uncensored_df.iterrows():
                    sales_obs = row['s_t']

                    # Find the closest higher (or equal) censored bucket
                    idx = bisect.bisect_left(order_levels, sales_obs)
                    if idx < len(order_levels):
                        chosen_level = order_levels[idx]
                    else:
                        # No higher level (and no max bucket); assign to the largest existing level
                        chosen_level = order_levels[-1]
                    buckets[chosen_level].append(sales_obs)

    order_levels = sorted(order_levels)
    censored_demands = [np.array(buckets[L]) for L in order_levels]

    return order_levels, censored_demands




def get_uncensored_data(product_id, dataset='train'):
    """
    Build (uncensored_demands) for a given product.

    Parameters
    ----------
    product_id : int or str
        Product identifier to filter the dataset.
    dataset : str, default 'train'
        CSV filename (without extension) to read from, e.g., 'train' or 'eval'.

    Returns
    -------
    uncensored_demands : np.ndarray
    """

    # Load data and filter by product
    df = pd.read_csv(f'./datasets/{dataset}.csv')


    df = df[df['product_id'] == product_id].copy()
   

    uncensored_df = df[df['censored'] == 0]
    uncensored_demands = uncensored_df["s_t"].to_numpy()
    return uncensored_demands






def get_product_list():
    '''
    Loops through all of the product IDs to determine if there are:

    (1) Sufficient number of training samples
    (2) Sufficient number of evaluation samples
    (3) Low "variance" on the order_levels, which we are using as a proxy for ideally having a "strong" performance
    for our algorithm style

    Returns
    -------
    dict
        A mapping product_id -> summary dict with keys:
            - train_n: total number of censored-demand observations in train
            - eval_n: total number of censored-demand observations in eval
            - var_order_levels: variance of training-order-levels
            - order_levels: list of training order levels (sorted)
    '''

    # Pull product IDs from the training split
    train_df = pd.read_csv(f'./datasets/train.csv')
    product_list = train_df['product_id'].unique()

    eval_df = pd.read_csv(f'./datasets/eval.csv')
    # product_list = PRODUCT_LIST

    print(f'Final product list: {product_list}')

    final_list = []
    for product_id in product_list:
        # --- Max censored and uncensored ---
        train_product_df = train_df[train_df['product_id'] == product_id]
        eval_product_df = eval_df[eval_df['product_id'] == product_id]

        max_censored_train  = train_product_df.loc[train_product_df['censored'] == 1, 's_t'].max()
        max_uncensored_train = train_product_df.loc[train_product_df['censored'] == 0, 's_t'].max()

        max_censored_eval  = eval_product_df.loc[eval_product_df['censored'] == 1, 's_t'].max()
        max_uncensored_eval = eval_product_df.loc[eval_product_df['censored'] == 0, 's_t'].max()



        # --- Training stats ---
        training_order_levels, train_censored_demands = get_censored_sales_data(product_id, 'train')
        N_K = len(train_censored_demands[-1])
        train_n = sum(len(arr) for arr in train_censored_demands) if train_censored_demands else 0

        # --- Evaluation stats ---
        eval_order_levels, eval_censored_demands = get_censored_sales_data(product_id, 'eval')
        eval_n = sum(len(arr) for arr in eval_censored_demands) if eval_censored_demands else 0

        # --- Variance of order levels (use training set order levels) ---
        if len(training_order_levels) >= 2:
            var = float(np.var(np.array(training_order_levels, dtype=float)))
        else:
            var = 0.0

        print(f'Adding: {product_id}')
        final_list.append({
            'product': product_id,
            'train_n':train_n,
            'eval_n': eval_n,
            'max_censored_train': max_censored_train,
            'max_censored_eval': max_censored_eval,
            'max_uncensored_train': max_uncensored_train,
            'max_uncensored_eval': max_uncensored_eval,
            'N_K': N_K,
            'order_levels': training_order_levels,
            'num_order_levels': len(training_order_levels),
            'var_order_levels': var,
        })

    return final_list

# RUN THIS TO GENERATE PRODUCT_DATA

# data = get_product_list()

# df = pd.DataFrame(data)
# df.to_csv('./data/product_data.csv', index=False)  # index=False to avoid writing row numbers
