# ============================================================
# CAPSTONE PROJECT: BLACK-LITTERMAN SHARED FUNCTIONS
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Holds the settings and functions used by:
#     bl_insample.py       BL inputs, baseline models and CVaR frontier on the
#                          training set, historical vs BL means
#     bl_walk_forward.py   monthly walk-forward of every model, historical vs BL
#   Nothing runs when this file is imported.
#
# Black-Litterman specification (one setting for each attribute):
#   Q      views = historical sample mean of each asset (the same estimate
#          the baseline models use), one absolute view per asset, so P = I.
#   Sigma  sample covariance of the same history (no shrinkage), as in the
#          baseline models. The same Sigma is used for pi, Omega, the BL
#          posterior and the optimizers that need a covariance.
#   pi     implied equilibrium returns, pi = lambda * Sigma * w_benchmark.
#   Omega  diagonal, Omega_jj = (1 - c_j) / c_j * p_j (tau Sigma) p_j'
#          (c = 0.5 gives He & Litterman's Omega = diag(P tau Sigma P')).
#   tau, lambda, benchmark weights and confidences are fixed below.
#
# Units: everything is in DAILY simple returns. Annual figures are daily
# means x 252 (arithmetic), as in the baseline scripts. The risk-free rate
# is 0, as in the baseline scripts, so total and excess returns coincide.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack, identity, vstack

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

ALPHA = 0.95          # CVaR confidence level
TRADING_DAYS = 252
RISK_FREE = 0.0       # annual risk-free rate, same as the baseline scripts
BTC_CAP = 0.20        # same cap as the baseline scripts; None for no cap

# Black-Litterman attributes (one iteration each).
BENCHMARK_WEIGHTS = {"Equity": 0.55, "Bond": 0.40, "BTC": 0.05}
RISK_AVERSION = 2.5   # lambda (He & Litterman, 1999)
TAU = 0.05            # tau (He & Litterman, 1999)
VIEW_CONFIDENCE = {"Equity": 0.50, "Bond": 0.50, "BTC": 0.25}

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

def find_data_dir():
    """Look for portfolio_train.csv beside this file, then in parent folders."""
    for folder in [HERE, *HERE.parents]:
        if (folder / "portfolio_train.csv").exists():
            return folder
    raise FileNotFoundError("portfolio_train.csv not found beside this script "
                            "or in any parent folder.")


def load(name):
    df = pd.read_csv(find_data_dir() / f"portfolio_{name}.csv",
                     index_col=0, parse_dates=True).sort_index()
    if df.index.hasnans or not df.index.is_unique or df.isna().any().any():
        raise ValueError(f"portfolio_{name}.csv has bad dates or missing values.")
    return df


def upper_bounds(assets):
    upper = np.ones(len(assets))
    if BTC_CAP is not None and "BTC" in assets:
        upper[assets.index("BTC")] = BTC_CAP
    return upper


# ------------------------------------------------------------
# COVARIANCE AND BLACK-LITTERMAN
# ------------------------------------------------------------

def sample_cov(R):
    """Daily sample covariance of the return matrix R (days x assets)."""
    return np.cov(R, rowvar=False)


def black_litterman(R, assets):
    """
    Fit Black-Litterman on the history R (days x assets, daily returns).

    Returns a dict of daily quantities: cov, q, pi, omega, mu_bl.
    """
    n = len(assets)
    cov = sample_cov(R)
    w_b = np.array([BENCHMARK_WEIGHTS[a] for a in assets], dtype=float)
    w_b = w_b / w_b.sum()
    rf = RISK_FREE / TRADING_DAYS

    # Prior: implied equilibrium excess returns.
    pi = RISK_AVERSION * cov @ w_b

    # Views: baseline sample mean of each asset (absolute views, P = I).
    P = np.eye(n)
    q = R.mean(axis=0) - rf * P.sum(axis=1)

    # View uncertainty from the stated confidence (diagonal).
    c = np.array([VIEW_CONFIDENCE[a] for a in assets], dtype=float)
    omega = np.diag((1 - c) / c * np.diag(P @ (TAU * cov) @ P.T))

    # Posterior mean (equivalent to the precision-weighted formula).
    A = TAU * cov
    mu_bl = pi + A @ P.T @ np.linalg.solve(P @ A @ P.T + omega, q - P @ pi)

    return {"cov": cov, "q": q + rf, "pi": pi + rf, "omega": omega,
            "mu_bl": mu_bl + rf, "w_benchmark": w_b}


# ------------------------------------------------------------
# PORTFOLIO MODELS (same formulations as the baseline scripts)
# ------------------------------------------------------------

def clean(w, upper):
    """Remove tiny solver noise: clip to the bounds and rescale to sum to 1."""
    w = np.clip(w, 0.0, upper)
    return w / w.sum()


def max_sharpe(mu, cov, upper):
    """mu, cov ANNUALIZED. Highest (return - risk-free) / volatility."""
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w), w0,
                   bounds=[(0.0, u) for u in upper],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   method="SLSQP")
    return clean(res.x, upper), res.success


def min_variance(cov, upper):
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: w @ cov @ w, w0, bounds=[(0.0, u) for u in upper],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   method="SLSQP")
    return clean(res.x, upper), res.success


def risk_parity(cov):
    """Equal risk contribution (convex formulation, rescaled to sum to 1). Not capped."""
    n = cov.shape[0]
    res = minimize(lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / n,
                   1.0 / np.sqrt(np.diag(cov)), bounds=[(1e-9, None)] * n,
                   method="L-BFGS-B")
    return res.x / res.x.sum()


def min_cvar(R, upper, mu=None, min_return=None):
    """
    Rockafellar-Uryasev LP on historical scenarios R (days x assets).

    mu is the DAILY expected-return vector used in the optional return
    constraint mu . w >= min_return (defaults to the sample mean of R).
    Returns are scaled to percent inside the LP for solver reliability.
    """
    N, n = R.shape
    mu = R.mean(axis=0) if mu is None else np.asarray(mu, dtype=float)
    Rp = R * 100.0
    c = np.concatenate([np.zeros(n), [1.0], np.full(N, 1.0 / ((1 - ALPHA) * N))])
    A_ub = hstack([csr_matrix(-Rp), csr_matrix(-np.ones((N, 1))),
                   -identity(N, format="csr")], format="csr")
    b_ub = np.zeros(N)
    if min_return is not None:
        row = np.concatenate([-mu * 100.0, [0.0], np.zeros(N)])
        A_ub = vstack([A_ub, csr_matrix(row)], format="csr")
        b_ub = np.append(b_ub, -min_return * 100.0)
    A_eq = csr_matrix(np.concatenate([np.ones(n), [0.0], np.zeros(N)]))

    message = ""
    for method in ("highs", "highs-ds", "highs-ipm"):
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0.0, u) for u in upper] + [(None, None)]
                      + [(0.0, None)] * N, method=method)
        if res.success:
            return clean(res.x[:n], upper)
        message = res.message
    raise RuntimeError(f"CVaR optimization failed: {message}")


def fit_models(R, assets, mu, cov, holding_days=1):
    """
    Fit every baseline model given expected returns and covariance.

    R, mu and cov are on the same horizon: daily returns (holding_days=1,
    in-sample analysis) or month-length scenarios (holding_days = trading
    days in the month, walk-forward). MVO and risk parity use (mu, cov) and
    are unaffected by the horizon scaling. Min CVaR uses the scenarios R only.
    CVaR (matched return) minimizes CVaR subject to earning at least the
    expected return (under mu) of the max-Sharpe portfolio, less a slack of
    1e-6 per holding day that keeps the target strictly feasible.
    Returns ({name: weights}, number of SLSQP fits that did not converge).
    """
    upper = upper_bounds(assets)
    scale = TRADING_DAYS / holding_days
    mu_a, cov_a = mu * scale, cov * scale
    w_ms, ok1 = max_sharpe(mu_a, cov_a, upper)
    w_mv, ok2 = min_variance(cov_a, upper)
    target = float(mu @ w_ms) - 1e-6 * holding_days
    models = {
        "MVO (max Sharpe)": w_ms,
        "MVO (min variance)": w_mv,
        "Risk parity": risk_parity(cov_a),
        "Min CVaR": min_cvar(R, upper),
        "CVaR (matched return)": min_cvar(R, upper, mu=mu, min_return=target),
    }
    return models, (not ok1) + (not ok2)


def historical_scenarios(history, holding_days):
    """
    Overlapping month-length scenarios from daily returns, as in the team's
    walk-forward: each scenario is the compounded return over holding_days
    consecutive trading days, rolling forward one day at a time.
    """
    growth = (1 + history).rolling(holding_days).apply(np.prod, raw=True)
    return (growth - 1).dropna().to_numpy()


def benchmarks(assets):
    n = len(assets)
    out = {"Equal weight": np.full(n, 1.0 / n)}
    if {"Equity", "Bond"} <= set(assets):
        w = np.zeros(n)
        w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
        out["60/40 equity/bond"] = w
    return out


# ------------------------------------------------------------
# PERFORMANCE
# ------------------------------------------------------------

def performance(r):
    """Metrics of a daily portfolio return series (array or Series)."""
    r = np.asarray(r, dtype=float)
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cutoff = np.quantile(r, 1 - ALPHA)
    wealth = np.cumprod(1 + r)
    return {
        "Total Return": wealth[-1] - 1,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe": (ann_ret - RISK_FREE) / ann_vol,
        "95% VaR (daily)": -cutoff,
        "95% CVaR (daily)": -r[r <= cutoff].mean(),
        "Max Drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
    }


def show_pct(df, keep=("Sharpe",)):
    """Copy of df with every column in % (2 dp) except those in keep."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind not in "fi":
            continue
        out[col] = out[col].round(2) if col in keep else (out[col] * 100).round(2)
    return out
