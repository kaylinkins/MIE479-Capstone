from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack


# SETTINGS

ALPHA = 0.95          # CVaR confidence level (worst 5% of days)
TRADING_DAYS = 252
BTC_CAP = 0.20        # max weight in BTC; set to None for no cap
N_FRONTIER = 6        # number of points on the CVaR-return frontier

output = Path(__file__).resolve().parent



# STEP 1: LOAD DATA


train = pd.read_csv(
    output / "portfolio_train.csv", index_col=0, parse_dates=True
).sort_index()
validation = pd.read_csv(
    output / "portfolio_validation.csv", index_col=0, parse_dates=True
).sort_index()

assets = list(train.columns)          # Equity, Bond, BTC
n_assets = len(assets)
R_train = train.values
R_val = validation[assets].values

# Upper bound on each weight (long-only, fully invested).
upper = [1.0] * n_assets
if BTC_CAP is not None and "BTC" in assets:
    upper[assets.index("BTC")] = BTC_CAP


# STEP 2: HELPER FUNCTIONS

def var_cvar(weights, returns, alpha=ALPHA):
    """Historical VaR and CVaR of a portfolio, expressed as positive losses."""
    port = returns @ weights
    cutoff = np.quantile(port, 1 - alpha)
    return -cutoff, -port[port <= cutoff].mean()


def min_cvar(returns, alpha=ALPHA, upper=None, min_return=None):
    """
    Rockafellar-Uryasev LP.

    Decision variables: [w (n assets), zeta (VaR level), u (N slack terms)]

        minimize   zeta + 1 / ((1 - alpha) * N) * sum(u)
        subject to u_i >= -(r_i . w) - zeta     (u_i = loss beyond zeta in scenario i)
                   u_i >= 0
                   sum(w) = 1
                   0 <= w_i <= upper_i
                   mean_return . w >= min_return   (optional)

    At the optimum, zeta is the VaR and the objective value is the CVaR.
    """
    N, n = returns.shape
    upper = upper or [1.0] * n

    # Objective: zeta has coefficient 1, each u_i has 1 / ((1 - alpha) N).
    c = np.concatenate([np.zeros(n), [1.0], np.full(N, 1.0 / ((1 - alpha) * N))])

    # Scenario constraints:  -r_i . w - zeta - u_i <= 0
    A_ub = hstack(
        [csr_matrix(-returns), csr_matrix(-np.ones((N, 1))), -identity(N, format="csr")],
        format="csr",
    )
    b_ub = np.zeros(N)

    # Optional minimum expected return:  -mean . w <= -min_return
    if min_return is not None:
        row = np.concatenate([-returns.mean(axis=0), [0.0], np.zeros(N)])
        A_ub = vstack([A_ub, csr_matrix(row)], format="csr")
        b_ub = np.append(b_ub, -min_return)

    # Weights sum to 1.
    A_eq = csr_matrix(np.concatenate([np.ones(n), [0.0], np.zeros(N)]))
    b_eq = [1.0]

    bounds = [(0.0, u) for u in upper] + [(None, None)] + [(0.0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR optimization failed: {res.message}")
    return res.x[:n]


def max_feasible_return(returns, upper):
    """Highest mean daily return achievable under the weight caps."""
    n = returns.shape[1]
    res = linprog(-returns.mean(axis=0), A_eq=[np.ones(n)], b_eq=[1.0],
                  bounds=[(0.0, u) for u in upper], method="highs")
    return -res.fun



# STEP 3: BUILD PORTFOLIOS


portfolios = {}

# Minimum-CVaR portfolio (the "main" portfolio, in its simplest form).
w_min = min_cvar(R_train, upper=upper)
portfolios["Min CVaR"] = w_min

# Points along the frontier: minimum CVaR for increasing required return.
mu_train = R_train.mean(axis=0)
low = float(mu_train @ w_min)
high = max_feasible_return(R_train, upper) * 0.98   # stay just inside feasibility
for target in np.linspace(low, high, N_FRONTIER)[1:]:
    w = min_cvar(R_train, upper=upper, min_return=target)
    portfolios[f"Min CVaR, return >= {target * TRADING_DAYS:.1%}/yr"] = w

# Simple benchmarks for reference.
portfolios["Equal weight"] = np.full(n_assets, 1.0 / n_assets)
if {"Equity", "Bond"} <= set(assets):
    w = np.zeros(n_assets)
    w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
    portfolios["60/40 equity/bond"] = w



# STEP 4: EVALUATE ON TRAIN AND VALIDATION


rows = []
for name, w in portfolios.items():
    var_tr, cvar_tr = var_cvar(w, R_train)
    var_va, cvar_va = var_cvar(w, R_val)
    row = {"Portfolio": name}
    row.update({a: w[i] for i, a in enumerate(assets)})
    row.update({
        "Train Ann. Return": (R_train @ w).mean() * TRADING_DAYS,
        "Train 95% VaR (daily)": var_tr,
        "Train 95% CVaR (daily)": cvar_tr,
        "Val Ann. Return": (R_val @ w).mean() * TRADING_DAYS,
        "Val 95% VaR (daily)": var_va,
        "Val 95% CVaR (daily)": cvar_va,
    })
    rows.append(row)

results = pd.DataFrame(rows).set_index("Portfolio")
results.to_csv(output / "cvar_frontier.csv")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(f"Train: {len(R_train)} days | Validation: {len(R_val)} days | "
      f"BTC cap: {BTC_CAP}")
print("\nWEIGHTS:")
print(results[assets].round(3).to_string())
print("\nRISK AND RETURN (CVaR/VaR are daily losses, in %):")
risk = results.drop(columns=assets).copy()
for col in risk.columns:
    risk[col] = (risk[col] * 100).round(2)
print(risk.to_string())
print("\nSaved: cvar_frontier.csv")
