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



def evaluate_algorithm(algo_name, algo_func, lam, order_level_two, censored_demand_list_one, censored_demand_list_two, b, h, qbar, eval_demand_list):
    """Compute solutions and costs for the given algorithm."""
    sol = algo_func(lam, order_level_two, censored_demand_list_one, censored_demand_list_two, b, h)
    true_cost = get_newsvendor_cost(sol, b, h, eval_demand_list)
    mm_cost = get_robust_cost(sol, b, h, lam, qbar, eval_demand_list)
    if DEBUG: print(f'Algo: {algo_name}, Sol: {sol}, true cost: {true_cost}, MM cost: {mm_cost}')
    return sol, true_cost, mm_cost


def evaluate_true_saa(lam, true_demand_list, b, h, qbar, eval_demand_list):
    # Evaluates the true SAA cost
    sol = algorithms.true_saa(true_demand_list, b, h)
    true_cost = get_newsvendor_cost(sol, b, h, eval_demand_list)
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
    elif distribution_name == 'exponential':
        return np.random.exponential(scale=param, size=N)
    elif distribution_name == 'poisson':
        return np.random.poisson(lam=param, size=N)
    elif distribution_name == 'negative_binomial':
        return np.random.negative_binomial(n=param[0], p=param[1], size=N)
    elif distribution_name == 'normal':
        return np.maximum(np.random.normal(loc=80, scale=param, size=N), 0)



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
        # lam_list = [qopt/2 + (qopt*k)/7 for k in range(8)]
        lam_list = [qopt/2, qopt, (3*qopt/2)]

    return qbar, param_list, b_list, rho_list, lam_list

def get_q_crit(Gminus, lam, b, h, qbar):
    # Returns q_crit on demand list
    return (b*qbar + h*lam - (b+h)*Gminus*qbar) / ((b+h)*(1 - Gminus))

def get_optimal_quantile(b, h, demand_list):
    # Returns optimal rho quantile on demand_list
    rho = b / (b+h)

    order = np.quantile(demand_list, rho)
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
        

def get_optimal_robust(b, h, lam, qbar, demand_list):
    # Evaluates q^\Delta, i.e. q_G^* when identifiable, q_G^\dag otherwise
    rho = b / (b+h)
    Gminus = np.mean(demand_list < lam)
    if Gminus >= rho:
        return get_optimal_quantile(b, h, demand_list)
    else:
        return get_q_crit(Gminus, lam, b, h, qbar)
    

def export_legend(legend, filename="LABEL_ONLY.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    plt.close('all')
