
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# SETTINGS


ALPHA = 0.95          # CVaR confidence level
TRADING_DAYS = 252
RISK_FREE = 0.0       # annual risk-free rate used in the Sharpe ratio
BTC_CAP = 0.20        # keep the same as cvar_optimization.py; None for no cap

output = Path(__file__).resolve().parent



# STEP 1: LOAD DATA


train = pd.read_csv(
    output / "portfolio_train.csv", index_col=0, parse_dates=True
).sort_index()
validation = pd.read_csv(
    output / "portfolio_validation.csv", index_col=0, parse_dates=True
).sort_index()

assets = list(train.columns)
n = len(assets)
R_train = train.values
R_val = validation[assets].values

# Sample statistics from the training set, annualized.
mu = R_train.mean(axis=0) * TRADING_DAYS
cov = np.cov(R_train, rowvar=False) * TRADING_DAYS

# Weight bounds (long-only, optional BTC cap).
upper = [1.0] * n
if BTC_CAP is not None and "BTC" in assets:
    upper[assets.index("BTC")] = BTC_CAP
bounds = [(0.0, u) for u in upper]
sum_to_one = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
w0 = np.full(n, 1.0 / n)


# STEP 2: MODELS

def min_variance():
    """Smallest possible portfolio variance."""
    res = minimize(lambda w: w @ cov @ w, w0, bounds=bounds,
                   constraints=[sum_to_one], method="SLSQP")
    return res.x


def max_sharpe():
    """Highest (return - risk-free) / volatility."""
    def neg_sharpe(w):
        return -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w)
    res = minimize(neg_sharpe, w0, bounds=bounds,
                   constraints=[sum_to_one], method="SLSQP")
    return res.x


def risk_parity():
    """
    Equal risk contribution. Solves the convex problem
        minimize  0.5 * y' cov y  -  (1/n) * sum(log(y_i)),   y > 0
    and rescales y to sum to 1. At the optimum every asset contributes
    the same share of portfolio volatility.
    """
    def objective(y):
        return 0.5 * y @ cov @ y - np.log(y).sum() / n
    res = minimize(objective, 1.0 / np.sqrt(np.diag(cov)),
                   bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()


def risk_contributions(w):
    """Each asset's share of total portfolio variance (sums to 1)."""
    return w * (cov @ w) / (w @ cov @ w)



# STEP 3: COLLECT PORTFOLIOS

portfolios = {
    "MVO (max Sharpe)": max_sharpe(),
    "MVO (min variance)": min_variance(),
    "Risk parity": risk_parity(),
}

# Min-CVaR weights come from cvar_optimization.py so the CVaR code lives
# in one place.
cvar_file = output / "cvar_frontier.csv"
if not cvar_file.exists():
    raise FileNotFoundError("Run cvar_optimization.py first (it creates cvar_frontier.csv).")
portfolios["Min CVaR"] = pd.read_csv(cvar_file, index_col=0).loc["Min CVaR", assets].values

portfolios["Equal weight"] = np.full(n, 1.0 / n)
if {"Equity", "Bond"} <= set(assets):
    w = np.zeros(n)
    w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
    portfolios["60/40 equity/bond"] = w

# Sanity check: weights sum to 1 and are non-negative.
for name, w in portfolios.items():
    assert abs(w.sum() - 1) < 1e-6 and (w > -1e-8).all(), f"Bad weights for {name}"



# STEP 4: PERFORMANCE METRICS


def performance(w, returns):
    """Daily-rebalanced portfolio metrics on a return matrix."""
    port = returns @ w
    ann_ret = port.mean() * TRADING_DAYS
    ann_vol = port.std(ddof=1) * np.sqrt(TRADING_DAYS)
    cutoff = np.quantile(port, 1 - ALPHA)
    wealth = np.cumprod(1 + port)
    return {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe": (ann_ret - RISK_FREE) / ann_vol,
        "95% VaR (daily)": -cutoff,
        "95% CVaR (daily)": -port[port <= cutoff].mean(),
        "Max Drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
    }


rows = []
for name, w in portfolios.items():
    for label, data in (("Train", R_train), ("Validation", R_val)):
        row = {"Portfolio": name, "Period": label}
        row.update({a: w[i] for i, a in enumerate(assets)})
        row.update(performance(w, data))
        rows.append(row)

results = pd.DataFrame(rows).set_index(["Portfolio", "Period"])
results.to_csv(output / "baseline_comparison.csv")


# STEP 5: PRINT


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(f"Train: {len(R_train)} days | Validation: {len(R_val)} days | BTC cap (MVO, CVaR): {BTC_CAP}")

print("\nSAMPLE STATISTICS USED (training, annualized):")
print(pd.DataFrame({"Mean return": mu, "Volatility": np.sqrt(np.diag(cov))}, index=assets).round(3).to_string())

print("\nRISK PARITY CHECK (share of risk per asset; should be ~equal):")
print(pd.Series(risk_contributions(portfolios["Risk parity"]), index=assets).round(3).to_string())

show = results.copy()
for col in show.columns:
    if col != "Sharpe":
        show[col] = (show[col] * 100).round(2)      # shown in %
    else:
        show[col] = show[col].round(2)
print("\nCOMPARISON (everything in %, except Sharpe; VaR/CVaR are daily losses):")
print(show.to_string())
print("\nSaved: baseline_comparison.csv")
