import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import random
import math
import pandas as pd

import helper


from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

'''

Creates two figures comparing q^\Delta and \Delta versus the observable boundary \lambda comparing well-separated to not well-separated distributions

'''



# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use('PaperDoubleFig.mplstyle.txt')


b = 0.9
h = 0.1

rho = b / (b+h)

qbar = 1



lambdas = [0.1,0.3,0.5,0.7,0.9]
lambdas = [round(e,2) for e in lambdas]

gammas = np.arange(0.01,1.01,0.01)


gminus_calc_list = np.random.uniform(0, 1, int(1e6))

# --- Create DataFrame for plotting ---
records = []
for lam in lambdas:
    Gminus = np.mean(gminus_calc_list < lam)

    for gam in gammas:

        Mtilde = min(qbar, lam + ((rho - Gminus) / gam), 1 / gam)

        q_val = helper.get_optimal_robust(b, h, lam, qbar, gminus_calc_list)

        # check here that q_val = rho up to 2 digits
        if not np.isclose(q_val, rho, atol=1e-2):
            print(f"[WARN] q_val ≈ {q_val:.3f} but expected ≈ rho={rho:.3f} (λ={lam})")

        delta_val = helper.get_robust_cost(q_val, b, h, lam, qbar, gminus_calc_list)

        delta_theory = (1 - rho) * (rho - lam) * (b + h)
        if not np.isclose(delta_val, delta_theory, atol=1e-2):
            print(f"[WARN] Δ mismatch: computed {delta_val:.3f}, expected {delta_theory:.3f} (λ={lam})")
        # check here that delta_val = (1-rho) * (rho-lam) ? 

        qws_val = helper.get_well_separated_optimal_robust(b, h, lam, qbar, gam, gminus_calc_list)
        
        # check here that qws = lam + ((1-lam)-np.sqrt((1-lam)**2 +(rho-lam-gam*(Mtilde-lam))**2-(rho-lam)**2))/gam
        qws_theory = lam + ((1 - lam) - np.sqrt((1 - lam)**2 + (rho - lam - gam*(Mtilde - lam))**2 - (rho - lam)**2)) / gam
        if not np.isclose(qws_val, qws_theory, atol=1e-2):
            print(f"[WARN] q_ws mismatch: computed {qws_val:.3f}, expected {qws_theory:.3f} (λ={lam}, γ={gam:.2f})")

        deltaws_val = helper.get_well_separated_robust_cost(qws_val, b, h, lam, qbar, gam, gminus_calc_list)

        # deltaws_theory = (1-rho) * (qws_val-lam)
        deltaws_theory = (b+h) * (1 - rho) * (qws_theory - lam)
        if not np.isclose(deltaws_val, deltaws_theory, atol=1e-2):
            print(f"[WARN] Δ_ws mismatch: computed {deltaws_val:.3f}, expected {deltaws_theory:.3f} (λ={lam}, γ={gam:.2f})")

        records.append({
            'lambda': lam,
            'gamma': gam,
            'q': q_val,
            'q_ws': qws_val,
            'delta': delta_val,
            'delta_ws': deltaws_val
        })

plot_df = pd.DataFrame(records)

print(plot_df.head(5))



# --- First plot: Δ−Δ_ws ---
plt.figure(figsize=(10, 6))
palette = sns.color_palette('husl', n_colors=len(lambdas))
sns.lineplot(data=plot_df, x='gamma', y=plot_df['delta'] - plot_df['delta_ws'], hue='lambda', palette=palette)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Delta-\Delta^{\mathsf{ws}}$')
plt.legend(loc='upper left', title=r'$\lambda$')
plt.tight_layout()
plt.savefig('./figures/ws_delta_diff.pdf', bbox_inches='tight')
plt.close()

# --- Second plot: q^Δ−q^{Δ_ws} ---
plt.figure(figsize=(10, 6))
palette = sns.color_palette('husl', n_colors=len(lambdas))
sns.lineplot(data=plot_df, x='gamma', y=plot_df['q'] - plot_df['q_ws'], hue='lambda', legend=False, palette=palette)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$q^{\Delta}-q^{\Delta^{\mathsf{ws}}}$')
plt.tight_layout()
plt.savefig('./figures/ws_q_delta_diff.pdf', bbox_inches='tight')
plt.close()