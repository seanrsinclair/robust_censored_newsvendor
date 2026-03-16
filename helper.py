import numpy as np
import math
import algorithms
import matplotlib.pyplot as plt

# DEBUG = True
DEBUG = False


def poisson_cdf(mu, a):
    # Calculate the cumulative probability for X < a for a Poisson distribution with mean mu
    cumulative_prob = 0
    for k in range(int(a)):  # Since Poisson is discrete, we sum up to floor(a)
        cumulative_prob += (math.exp(-mu) * (mu ** k)) / math.factorial(k)
    return cumulative_prob


def evaluate_algorithm(algo_name, algo_func, order_levels, censored_demands, b, h, qbar, eval_demand_list, gamma = None):
    """Compute solutions and costs for the given algorithm."""
    sol, ind = algo_func(order_levels, censored_demands, b, h)
    lam = max(order_levels)
    true_cost = get_newsvendor_cost(sol, b, h, eval_demand_list)
    if gamma is not None:
        mm_cost = get_well_separated_robust_cost(sol, b, h, lam, qbar, gamma, eval_demand_list)
    else:
        mm_cost = get_robust_cost(sol, b, h, lam, qbar, eval_demand_list)
    if DEBUG: print(f'Algo: {algo_name}, Sol: {sol}, true cost: {true_cost}, MM cost: {mm_cost}')
    return sol, true_cost, mm_cost, ind


def evaluate_true_saa(lam, true_demand_list, b, h, qbar, eval_demand_list, gamma = None):
    # Evaluates the true SAA cost
    sol = algorithms.true_saa(true_demand_list, b, h)
    true_cost = get_newsvendor_cost(sol, b, h, eval_demand_list)
    if gamma is not None:
        mm_cost = get_well_separated_robust_cost(sol, b, h, lam, qbar, gamma, eval_demand_list)
    else:
        mm_cost = get_robust_cost(sol, b, h, lam, qbar, eval_demand_list)
    return sol, true_cost, mm_cost


def add_to_data(data, algo_name, metric, cost, **kwargs):
    """Add result dictionary to the data list."""
    data.append({
        'algorithm': algo_name,
        'metric': metric,
        'value': cost,
        **kwargs
    })



def sample_dist(N, distribution_name, param):
    # Evaluates from distribution according to param
    if distribution_name == 'uniform':
        return np.random.randint(0, 100, size=N)
    elif distribution_name == 'continuous_uniform':
        return np.random.uniform(0,100, size=N)
    elif distribution_name == 'truncated_exponential':
        return np.clip(np.random.exponential(scale=param, size=N), 0, 325)
    elif distribution_name == 'exponential':
        return np.random.exponential(scale=param, size=N)
    elif distribution_name == 'vanilla_exponential':
        return np.random.exponential(scale=param, size=N)
    elif distribution_name == 'poisson':
        return np.random.poisson(lam=param, size=N)
    elif distribution_name == 'negative_binomial':
        return np.random.negative_binomial(n=param[0], p=param[1], size=N)
    elif distribution_name == 'normal':
        return np.maximum(np.random.normal(loc=80, scale=param, size=N), 0)


def get_well_separated_parameters(distribution_name):
    # Returns the M, mean, param list, b values, rho values, and lambda values for each
    # distribution that we run experiments to
    h = 1
    if distribution_name == 'continuous_uniform':
        qbar = 100
        mean = 50
        param_list = [mean]
        # b_list = [3, 9, 49]
        b_list = [9]
        rho_list = [(b / (b+h)) for b in b_list]

        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = np.linspace(0.98 * qopt, 1.05 * qopt, 8)
        # lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        gamma = (1 / (2*mean))
    
    elif distribution_name == 'truncated_exponential':
        qbar = 325
        mean = 80
        truncate = 325
        param_list = [mean]
        b_list = [9]
        rho_list = [(b / (b+h)) for b in b_list]
        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = np.linspace(0.95 * qopt, 1.1 * qopt, 8)
        # lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        gamma = (1 / mean) * np.exp(-truncate / mean)
    

    return qbar, param_list, b_list, rho_list, lam_list, gamma



def get_parameters(distribution_name):
    # Returns the M, mean, param list, b values, rho values, and lambda values for each
    # distribution that we run experiments to
    h = 1
    if distribution_name == 'uniform':
        qbar = 325
        mean = 50
        param_list = [mean]
        b_list = [3, 9, 49]
        rho_list = [(b / (b+h)) for b in b_list]

        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        # lam_list = [qopt/2 + (qopt*k)/7 for k in [2,3,4,5]]

    elif distribution_name == 'continuous_uniform':
        qbar = 100
        mean = 50
        param_list = [mean]
        b_list = [9]
        rho_list = [(b / (b+h)) for b in b_list]

        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        # lam_list = [qopt/2 + (qopt*k)/7 for k in [2,3,4,5]]


    elif distribution_name == 'exponential':
        # rho_list = [0.5]
        mean = 80
        param_list = [mean]
        qbar = 325
        b_list = [3, 9, 49]
        # b_list = [49]
        rho_list = [(b / (b+h)) for b in b_list]

        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        # lam_list = [qopt/2 + (qopt*k)/7 for k in [5,6,7]]

    elif distribution_name == 'vanilla_exponential':
        # rho_list = [0.5]
        mean = 80
        param_list = [mean/2, mean]
        qbar = 325
        b_list = [9]
        # b_list = [49]
        rho_list = [(b / (b+h)) for b in b_list]

        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        # lam_list = [qopt/2 + (qopt*k)/7 for k in [5,6,7]]



    elif distribution_name == 'poisson':
        mean = 80
        param_list = [mean]
        qbar = 325
        b_list = [3, 9, 49]
        rho_list = [(b / (b+h)) for b in b_list]
        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]

    elif distribution_name == 'negative_binomial':
        qbar = 325
        param_list = [(80, 1/2), (40, 1/3)]
        b_list = [3, 9, 49]
        rho_list = [(b / (b+h)) for b in b_list]
        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, param_list[-1]))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]

    elif distribution_name == 'normal':
        qbar = 325
        mean = 80
        param_list = np.arange(20, 41, 5)
        b_list = [3, 9, 49]
        rho_list = [(b / (b+h)) for b in b_list]
        qopt = get_optimal_quantile(9, 1, sample_dist(int(1e7), distribution_name, 30))
        print(f'qopt value: {qopt}')
        lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        # lam_list = [qopt/2, qopt, (3*qopt/2)]

    return qbar, param_list, b_list, rho_list, lam_list

def get_q_crit(Gminus, lam, b, h, qbar):
    # Returns q_crit on demand list
    return (b*qbar + h*lam - (b+h)*Gminus*qbar) / ((b+h)*(1 - Gminus))


def get_well_separated_q_crit(Gminus, lam, b, h, qbar, gamma):
    # Returns q_crit with the well-separated definition
    rho = b / (b+h)
    Mtilde = min(qbar, lam + (rho - Gminus) / gamma, 1 / gamma)
    term1 = (1 - Gminus)
    term2 = np.sqrt((1 - Gminus)**2 + (rho - Gminus - (gamma * (Mtilde - lam)))**2 - (rho - Gminus)**2)
    return lam + (term1 - term2) / gamma


def get_optimal_quantile(b, h, demand_list):
    # Returns optimal rho quantile on demand_list
    rho = b / (b+h)

    order = np.quantile(demand_list, rho, method='higher')
    return order


def get_newsvendor_cost(order, b, h, demand_list):
    # Evaluates C_G(order) using a dataset of demand_list

    # Compute (D - q)^+ and (q - D)^+
    term1 = np.maximum(demand_list - order, 0)  # (D - q)^+
    term2 = np.maximum(order - demand_list, 0)  # (q - D)^+
    
    # Compute the expression for each D_i
    total_cost = b * term1 + h * term2

    return np.mean(total_cost)


def get_robust_cost(order, b, h, lam, qbar, demand_list):
    # Evaluates our closed form expression for \sup_{F \in G(lambda)} C_F(q) - C_F(\qopt_F)

    rho = b / (b+h)
    opt_quantile = get_optimal_quantile(b, h, demand_list)
    Gminus = np.mean(demand_list < lam)

    if order < lam and Gminus >= rho:
        term1 = np.mean((order - demand_list)*(demand_list <= order).astype(int))
        term2 = np.mean((opt_quantile - demand_list) * (demand_list <= opt_quantile).astype(int))
        return b*(opt_quantile - order) + (b+h)*(term1 - term2)

    elif order >= lam and Gminus >= rho:
        term1 = np.mean((lam - demand_list)*(demand_list < lam).astype(int))
        term2 = np.mean((opt_quantile - demand_list)*(demand_list <= opt_quantile).astype(int))
        return b*(opt_quantile - order) + (b+h)*((order - lam) + term1 - term2)

    elif order < lam and Gminus < rho:
        term1 = np.mean((order - demand_list)*(demand_list <= order).astype(int))
        term2 = np.mean((qbar - demand_list)*(demand_list < lam).astype(int))
        return b*(qbar - order) + (b+h)*(term1 - term2)

    else: # order >= lam, Gminus < rho
        q_crit = get_q_crit(Gminus, lam, b, h, qbar)
        if order < q_crit:
            return (b - (b+h)*Gminus)*(qbar - order)
        else: # order >= qcrit
            return h*(order - lam)



def get_well_separated_robust_cost(order, b, h, lam, qbar, gamma, demand_list):

    rho = b / (b + h)
    Gminus = np.mean(demand_list < lam)
    
    Mtilde = min(qbar, lam + ((rho - Gminus) / gamma), 1 / gamma)
    d = np.asarray(demand_list)

    # Empirical CDF at lambda
    Gminus = np.mean(d < lam)

    # Identifiable regime: fall back to standard robust cost
    if Gminus >= rho:
        return get_robust_cost(order, b, h, lam, qbar, d)

    if order < lam:
        # E[(q - D) 1{D <= q}] and E[(Mtilde - D) 1{D < lam}]
        exp_q = np.mean((order - d) * (d <= order))
        exp_Mt = np.mean((Mtilde - d) * (d < lam))
        return (b + h) * (
            rho * (Mtilde - order)
            + exp_q
            - exp_Mt
            - 0.5 * gamma * (Mtilde - lam) ** 2
        )
    else:
        # q >= lam
        term1 = (1 - rho) * (order - lam)
        term2 = (Mtilde - order) * (rho - Gminus - gamma * (order - lam)) - 0.5 * gamma * (Mtilde - order) ** 2
        return (b + h) * max(term1, term2)        


def get_optimal_robust(b, h, lam, qbar, demand_list):
    # Evaluates q^\Delta, i.e. q_G^* when identifiable, q_G^\dag otherwise
    rho = b / (b+h)
    Gminus = np.mean(demand_list < lam)
    if Gminus >= rho:
        return get_optimal_quantile(b, h, demand_list)
    else:
        return get_q_crit(Gminus, lam, b, h, qbar)

def get_well_separated_optimal_robust(b, h, lam, qbar, gamma, demand_list):
    # Evaluates q^\Delta, i.e. q_G^* when identifiable, q_G^\dag otherwise
    rho = b / (b+h)
    Gminus = np.mean(demand_list < lam)
    if Gminus >= rho:
        return get_optimal_quantile(b, h, demand_list)
    else:
        return get_well_separated_q_crit(Gminus, lam, b, h, qbar, gamma)








def export_legend(legend, filename="LABEL_ONLY.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    plt.close('all')




def check_identifiable(row):
    if row["distribution"] == 'negative_binomial':
        param = eval(param)
    else:
        param = row["param"]
    gminus_calc_list = sample_dist(int(1e7), row["distribution"], param)
    mm_opt = get_optimal_robust(row["b"], row["h"], row["lam"], row["qbar"], gminus_calc_list)

    cost = get_robust_cost(
        mm_opt,
        row["b"],
        row["h"],
        row["lam"],
        row["qbar"],
        gminus_calc_list
    )
    return 1 if cost > 0 else 0


def float_format_sig(x, sig=2):
    """Format float x with specified number of significant digits."""
    # if x == 0:
    #     return "0"
    # return f"{x:.{sig-1}e}" if abs(x) < 0.01 or abs(x) >= 1000 else f"{x:.{sig}g}"
    if x > 10:
        return f"{x:.0f}"
    else:
        return f"{x:.1f}"

