"""CVaR frontier using Black-Litterman expected returns.
Run this copy instead of the original cvar_optimization.py for the BL experiment.
Historical joint daily returns remain the equally weighted loss scenarios.
"""
from pathlib import Path
import json
from black_litterman import fit_black_litterman, save_black_litterman, validate_returns

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
# Edit the JSON assumptions before reporting results.




# STEP 1: LOAD DATA


def var_cvar(weights, returns, alpha=ALPHA):
    """Historical VaR and CVaR of a portfolio, expressed as positive losses."""
    port = returns @ weights
    cutoff = np.quantile(port, 1 - alpha)
    loss = -port
    zeta = -cutoff
    # Exact empirical Rockafellar-Uryasev CVaR, including fractional tail mass.
    cvar = zeta + np.maximum(loss - zeta, 0).mean() / (1 - alpha)
    return zeta, cvar


def min_cvar(returns, alpha=ALPHA, upper=None, min_return=None, expected_returns=None):
    """
    Rockafellar-Uryasev LP.

    Decision variables: [w (n assets), zeta (VaR level), u (N slack terms)]

        minimize   zeta + 1 / ((1 - alpha) * N) * sum(u)
        subject to u_i >= -(r_i . w) - zeta     (u_i = loss beyond zeta in scenario i)
                   u_i >= 0
                   sum(w) = 1
                   0 <= w_i <= upper_i
                   expected_returns . w >= min_return   (optional)

    At the optimum, zeta is the VaR and the objective value is the CVaR.
    """
    N, n = returns.shape
    upper = [1.0] * n if upper is None else list(upper)
    if not 0 < alpha < 1 or N == 0 or not np.isfinite(returns).all():
        raise ValueError("Invalid scenarios or confidence level.")
    if len(upper) != n or not np.isfinite(upper).all() or min(upper) < 0 or sum(upper) < 1:
        raise ValueError("Invalid or infeasible weight caps.")
    mu = returns.mean(axis=0) if expected_returns is None else np.asarray(expected_returns, dtype=float)
    if mu.shape != (n,) or not np.isfinite(mu).all():
        raise ValueError("Expected-return vector must match scenario columns.")

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
        row = np.concatenate([-mu, [0.0], np.zeros(N)])
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


def max_feasible_return(expected_returns, upper):
    """Highest BL daily expected return achievable under the same weight caps."""
    n = len(expected_returns)
    res = linprog(-expected_returns, A_eq=[np.ones(n)], b_eq=[1.0],
                  bounds=[(0.0, u) for u in upper], method="highs")
    if not res.success:
        raise RuntimeError(f"Return maximization failed: {res.message}")
    return -res.fun


# STEP 3: BUILD PORTFOLIOS


def annualized_sharpe(weights, returns, risk_free_annual=0.0):
    """Historical Sharpe, conventionally annualized with sqrt(252).

    Uses realized daily simple returns and a constant risk-free assumption.
    Annual risk-free input is arithmetic, matching the BL configuration.
    Returns NaN if fewer than two observations or zero volatility.
    """
    portfolio_returns = returns @ weights
    excess_returns = portfolio_returns - risk_free_annual / TRADING_DAYS
    if len(excess_returns) < 2:
        return np.nan
    volatility = excess_returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility <= 0:
        return np.nan
    return float(excess_returns.mean() / volatility * np.sqrt(TRADING_DAYS))


def main():
    config = json.loads((output / "bl_config_demo.json").read_text(encoding="utf-8"))
    risk_free_annual = float(config.get("risk_free_annual", 0.0))
    print("BL MODE:", config.get("mode", "unspecified"))
    train = pd.read_csv(
        output / "portfolio_train.csv", index_col=0, parse_dates=True
    ).sort_index()
    validation = pd.read_csv(
        output / "portfolio_validation.csv", index_col=0, parse_dates=True
    ).sort_index()
    
    train = validate_returns(train, "train")
    validation = validate_returns(validation, "validation")
    if set(train.columns) != set(validation.columns):
        raise ValueError("Training/validation asset names differ.")
    if train.index.max() >= validation.index.min():
        raise ValueError("All training dates must precede all validation dates.")
    
    assets = list(train.columns)          # Equity, Bond, BTC
    n_assets = len(assets)
    R_train = train.values
    R_val = validation[assets].values
    
    # BL fits exclusively to training data. Validation is evaluation-only.
    bl = fit_black_litterman(train, config)
    mu_bl = bl["estimates"].loc[assets, "bl_total_daily"].to_numpy()
    
    # Upper bound on each weight (long-only, fully invested).
    upper = [1.0] * n_assets
    if BTC_CAP is not None and "BTC" in assets:
        upper[assets.index("BTC")] = BTC_CAP
    
    
    # STEP 2: HELPER FUNCTIONS
    
    portfolios = {}
    
    # Minimum-CVaR portfolio (the "main" portfolio, in its simplest form).
    w_min = min_cvar(R_train, upper=upper)
    portfolios["Min CVaR"] = w_min
    
    # Points along the frontier: minimum CVaR for increasing required return.
    mu_train = mu_bl
    low = float(mu_bl @ w_min)
    max_return = max_feasible_return(mu_bl, upper)
    # Stay inside the feasible interval, even when all expected returns are negative.
    high = low + 0.98 * max(0.0, max_return - low)   # stay just inside feasibility
    for target in np.linspace(low, high, N_FRONTIER)[1:]:
        w = min_cvar(R_train, upper=upper, min_return=target, expected_returns=mu_bl)
        portfolios[f"Min CVaR, return >= {target * TRADING_DAYS:.1%}/yr"] = w
    
    # Simple benchmarks for reference.
    portfolios["Equal weight (uncapped reference)"] = np.full(n_assets, 1.0 / n_assets)
    if {"Equity", "Bond"} <= set(assets):
        w = np.zeros(n_assets)
        w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
        portfolios["60/40 equity/bond"] = w
    
    
    
    # STEP 4: EVALUATE ON TRAIN AND VALIDATION
    
    
    rows = []
    for name, w in portfolios.items():
        var_tr, cvar_tr = var_cvar(w, R_train)
        var_va, cvar_va = var_cvar(w, R_val)
        sharpe_tr = annualized_sharpe(w, R_train, risk_free_annual)
        sharpe_va = annualized_sharpe(w, R_val, risk_free_annual)
        row = {"Portfolio": name}
        row.update({a: w[i] for i, a in enumerate(assets)})
        row.update({
            "Train Sharpe (annualized)": sharpe_tr,
            "Val Sharpe (annualized)": sharpe_va,
            "BL Expected Ann. Return": float(mu_bl @ w) * TRADING_DAYS,
            "Train Ann. Return": (R_train @ w).mean() * TRADING_DAYS,
            "Train 95% VaR (daily)": var_tr,
            "Train 95% CVaR (daily)": cvar_tr,
            "Val Ann. Return": (R_val @ w).mean() * TRADING_DAYS,
            "Val 95% VaR (daily)": var_va,
            "Val 95% CVaR (daily)": cvar_va,
        })
        rows.append(row)
    
    results = pd.DataFrame(rows).set_index("Portfolio")
    # Save into a separate folder, leaving original experiment outputs intact.
    destination = output / "bl_outputs"
    save_black_litterman(bl, destination)
    results.to_csv(destination / "cvar_bl_frontier.csv")
    
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"Train: {len(R_train)} days | Validation: {len(R_val)} days | "
          f"BTC cap: {BTC_CAP}")
    print("\nWEIGHTS:")
    print(results[assets].round(3).to_string())
    print("\nRISK AND RETURN (returns/VaR/CVaR in %; Sharpe ratios are unitless):")
    risk = results.drop(columns=assets).copy()
    for col in risk.columns:
        if "Sharpe" in col:
            risk[col] = risk[col].round(3)
        else:
            risk[col] = (risk[col] * 100).round(2)
    print(risk.to_string())
    print("\nSaved: bl_outputs/cvar_bl_frontier.csv")


if __name__ == "__main__":
    main()
