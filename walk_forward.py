# ============================================================
# CAPSTONE PROJECT: WALK-FORWARD TEST OF THE BASELINE MODELS
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Re-runs each baseline model month by month, the way it would be used
#   in practice. At the start of every month it:
#     1. fits the model on ALL data before that month (expanding window),
#     2. holds those weights for the month (no trading inside the month),
#     3. records the actual daily returns of that month.
#   The held-out months are stitched into one out-of-sample return series
#   per model, and the models are compared on that series.
#
# Models (all use sample statistics: historical mean and covariance):
#   - MVO (max Sharpe) and MVO (min variance)
#   - Risk parity
#   - Min CVaR (no return target)
#   - CVaR (matched return): minimum CVaR subject to earning at least the
#     same expected return as the max-Sharpe MVO portfolio in that month.
#     This is the like-for-like comparison against MVO.
#   - Equal weight and 60/40 benchmarks, also rebalanced monthly
#
# Inputs (from "Splitting data.py"):
#   portfolio_train.csv, portfolio_validation.csv
#
# Outputs:
#   walk_forward_results.csv   performance of each model
#   walk_forward_weights.csv   weights chosen at the start of each month
#   walk_forward_returns.csv   stitched daily out-of-sample returns
#
# Notes:
#   - The first month fitted is the first month of the validation set, so
#     the initial fit uses the training data only.
#   - The test set is NOT used. Keep INCLUDE_TEST = False until the models
#     are finalized.
#   - Weights drift with prices during a month and are reset at the next
#     rebalance. Trading costs are set in COST_BPS below (0 = no costs).
#     A cost is charged on every dollar bought or sold at each rebalance,
#     including the initial purchase. The models do not plan around costs:
#     the weights are the same as without costs, only the returns are lower.
#   - The model functions are copies of those in baseline_models.py and
#     cvar_optimization.py so that this file runs on its own. If you change
#     a model there, change it here too.
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
RISK_FREE = 0.0       # annual risk-free rate for the Sharpe ratio
BTC_CAP = 0.20        # keep the same as the other scripts; None for no cap
INCLUDE_TEST = False  # keep False until the models are finalized
PRINT_WEIGHTS = True  # print the weights chosen at every monthly rebalance

# Trading cost per asset, in basis points (1 bp = 0.01%) of the amount traded,
# charged on both buys and sells. 0 means no cost. Use your own estimates; the
# values here are placeholders and should be agreed on by the team. Assets not
# listed are treated as free to trade.
COST_BPS = {"Equity": 0.0, "Bond": 0.0, "BTC": 0.0}

output = Path(__file__).resolve().parent



# STEP 1: LOAD DATA


def load(name):
    return pd.read_csv(output / f"portfolio_{name}.csv",
                       index_col=0, parse_dates=True).sort_index()


train, validation = load("train"), load("validation")
frames = [train, validation]
if INCLUDE_TEST:
    frames.append(load("test"))

assets = list(train.columns)
n = len(assets)
returns = pd.concat(frames)[assets].sort_index()

if returns.index.hasnans or not returns.index.is_unique:
    raise ValueError("Return dates must be valid and unique.")
if returns.isna().any().any():
    raise ValueError("Returns must not contain missing values.")

# Cost rate per asset as a fraction of the amount traded.
cost_rate = np.array([COST_BPS.get(a, 0.0) for a in assets]) / 10_000

# Weight bounds: long-only, optional BTC cap (risk parity is not capped).
upper = np.ones(n)
if BTC_CAP is not None and "BTC" in assets:
    upper[assets.index("BTC")] = BTC_CAP
bounds = [(0.0, u) for u in upper]
sum_to_one = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
w_equal = np.full(n, 1.0 / n)

failures = {"SLSQP": 0, "LP": []}


def clean(w):
    """Remove tiny solver noise: clip to the bounds and rescale to sum to 1."""
    w = np.clip(w, 0.0, upper)
    return w / w.sum()


# STEP 2: MODELS (each takes only past data)


def min_variance(cov):
    res = minimize(lambda w: w @ cov @ w, w_equal, bounds=bounds,
                   constraints=[sum_to_one], method="SLSQP")
    failures["SLSQP"] += not res.success
    return clean(res.x)


def max_sharpe(mu, cov):
    def neg_sharpe(w):
        return -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w)
    res = minimize(neg_sharpe, w_equal, bounds=bounds,
                   constraints=[sum_to_one], method="SLSQP")
    failures["SLSQP"] += not res.success
    return clean(res.x)


def risk_parity(cov):
    """Equal risk contribution (convex formulation, rescaled to sum to 1)."""
    def objective(y):
        return 0.5 * y @ cov @ y - np.log(y).sum() / n
    res = minimize(objective, 1.0 / np.sqrt(np.diag(cov)),
                   bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()


def min_cvar(R, min_return=None):
    """
    Rockafellar-Uryasev LP on historical scenarios R (days x assets).

    Returns are scaled to percent inside the LP. Daily returns are around
    0.0001, and the solver is much more reliable with numbers near 1. The
    weights are not affected by the scaling.
    """
    N = R.shape[0]
    Rp = R * 100.0
    c = np.concatenate([np.zeros(n), [1.0],
                        np.full(N, 1.0 / ((1 - ALPHA) * N))])
    A_ub = hstack([csr_matrix(-Rp), csr_matrix(-np.ones((N, 1))),
                   -identity(N, format="csr")], format="csr")
    b_ub = np.zeros(N)
    if min_return is not None:
        row = np.concatenate([-Rp.mean(axis=0), [0.0], np.zeros(N)])
        A_ub = vstack([A_ub, csr_matrix(row)], format="csr")
        b_ub = np.append(b_ub, -min_return * 100.0)
    A_eq = csr_matrix(np.concatenate([np.ones(n), [0.0], np.zeros(N)]))

    message = ""
    for method in ("highs", "highs-ds", "highs-ipm"):   # try other HiGHS algorithms if one fails
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=bounds + [(None, None)] + [(0.0, None)] * N,
                      method=method)
        if res.success:
            return clean(res.x[:n])
        message = res.message
    raise RuntimeError(f"CVaR optimization failed: {message}")


def safe_cvar(name, month, previous, R, min_return=None):
    """
    Run min_cvar. If the solver still fails, keep last month's weights for
    this model (equal weight in the first month) and record the failure so
    it is reported at the end instead of crashing the whole run.
    """
    try:
        return min_cvar(R, min_return=min_return)
    except RuntimeError as error:
        failures["LP"].append((month, name, str(error)))
        return previous.get(name, w_equal)


def fit_all(R, month, previous):
    """Fit every model on the history R and return {name: weights}."""
    mu = R.mean(axis=0) * TRADING_DAYS
    cov = np.cov(R, rowvar=False) * TRADING_DAYS
    w_ms = max_sharpe(mu, cov)
    # A little slack (0.0001% a day) keeps the target strictly feasible.
    target = float(R.mean(axis=0) @ w_ms) - 1e-6
    models = {
        "MVO (max Sharpe)": w_ms,
        "MVO (min variance)": min_variance(cov),
        "Risk parity": risk_parity(cov),
        "Min CVaR": safe_cvar("Min CVaR", month, previous, R),
        # Same expected return as max-Sharpe MVO, but minimizing CVaR.
        "CVaR (matched return)": safe_cvar("CVaR (matched return)", month,
                                           previous, R, min_return=target),
        "Equal weight": w_equal,
    }
    if {"Equity", "Bond"} <= set(assets):
        w = np.zeros(n)
        w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
        models["60/40 equity/bond"] = w
    return models



# STEP 3: WALK FORWARD (expanding window, monthly rebalancing)


oos_start = validation.index.min()
months = returns.index.to_period("M")
oos_months = sorted(set(months[returns.index >= oos_start]))

first_day = returns.index[months == oos_months[0]][0]
if first_day < oos_start:
    raise ValueError("The validation set must start at the beginning of a month.")

daily = {}          # model -> list of daily return Series
turnover = {}       # model -> list of one-way turnover at each rebalance
costs = {}          # model -> list of trading costs paid (fraction of portfolio value)
drifted = {}        # model -> weights at the end of the previous month
weight_rows = []
last_weights = {}   # each model's weights from the previous month

for m in oos_months:
    days = returns[months == m]

    # Expanding window: everything strictly before this month.
    history = returns.loc[returns.index < days.index[0]]
    assert history.index.max() < days.index.min(), "Look-ahead detected."

    weights = fit_all(history.values, str(m), last_weights)
    last_weights = weights

    # Growth of each asset over the month, starting from 1.
    growth = (1 + days.values).cumprod(axis=0)

    for name, w in weights.items():
        if name in drifted:
            turnover.setdefault(name, []).append(0.5 * np.abs(w - drifted[name]).sum())

        # Trading cost: charged on everything bought or sold to get from last
        # month's drifted weights to the new weights (from cash in the first month).
        trades = np.abs(w - drifted.get(name, np.zeros(n)))
        cost = float(cost_rate @ trades)
        costs.setdefault(name, []).append(cost)

        # Buy and hold within the month: weights drift with prices. The cost is
        # paid up front, so it reduces the value of the portfolio all month.
        wealth = (growth @ w) * (1.0 - cost)
        previous = np.concatenate([[1.0], wealth[:-1]])
        daily.setdefault(name, []).append(
            pd.Series(wealth / previous - 1.0, index=days.index))

        end_value = w * growth[-1]
        drifted[name] = end_value / end_value.sum()

        weight_rows.append({"Month": str(m), "Portfolio": name,
                            **{a: w[i] for i, a in enumerate(assets)}})

oos_returns = pd.DataFrame({name: pd.concat(parts) for name, parts in daily.items()})
weights_df = pd.DataFrame(weight_rows)



# STEP 4: PERFORMANCE ON THE STITCHED OUT-OF-SAMPLE RETURNS


def performance(r):
    r = r.values
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


results = pd.DataFrame({name: performance(oos_returns[name])
                        for name in oos_returns.columns}).T
results["Avg Monthly Turnover"] = pd.Series(
    {name: np.mean(t) for name, t in turnover.items()})
results["Total Costs Paid"] = pd.Series(
    {name: np.sum(c) for name, c in costs.items()})
if "BTC" in assets:
    results["Avg BTC Weight"] = weights_df.groupby("Portfolio")["BTC"].mean()

results.to_csv(output / "walk_forward_results.csv")
weights_df.to_csv(output / "walk_forward_weights.csv", index=False)
oos_returns.to_csv(output / "walk_forward_returns.csv")


# ------------------------------------------------------------
# STEP 5: PRINT
# ------------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Out-of-sample period: {oos_returns.index.min().date()} to "
      f"{oos_returns.index.max().date()} "
      f"({len(oos_returns)} days, {len(oos_months)} monthly rebalances)")
print("Window: expanding (all data before each month) | Rebalancing: monthly")
print(f"BTC cap (MVO and CVaR models): {BTC_CAP}")
print("Trading costs (bps per amount traded): " +
      ", ".join(f"{a} {COST_BPS.get(a, 0.0):g}" for a in assets) +
      ("  [no costs charged]" if not cost_rate.any() else ""))
if failures["SLSQP"]:
    print(f"WARNING: the MVO solver reported {failures['SLSQP']} non-converged fits.")
if failures["LP"]:
    print(f"WARNING: the CVaR solver failed {len(failures['LP'])} time(s); the previous "
          "month's weights were kept for those months:")
    for month, name, message in failures["LP"]:
        print(f"  {month} | {name} | {message}")

show = results.copy()
for col in show.columns:
    show[col] = show[col].round(2) if col == "Sharpe" else (show[col] * 100).round(2)
print("\nWALK-FORWARD RESULTS (% except Sharpe; VaR/CVaR are daily losses):")
print(show.to_string())

print("\nAVERAGE WEIGHTS CHOSEN (%):")
print((weights_df.groupby("Portfolio")[assets].mean() * 100).round(1).to_string())

print("\nSaved: walk_forward_results.csv, walk_forward_weights.csv, walk_forward_returns.csv")


# STEP 6: WEIGHTS CHOSEN AT EACH MONTHLY REBALANCE

# Each row is the first trading day of a month and the target weights (%)
# set that day. The same table is saved in walk_forward_weights.csv.

if PRINT_WEIGHTS:
    rebalance_date = {str(m): returns.index[months == m][0].date() for m in oos_months}
    for name in weights_df["Portfolio"].unique():
        table = (weights_df[weights_df["Portfolio"] == name]
                 .set_index("Month")[assets] * 100).round(1)
        table.index = [rebalance_date[m] for m in table.index]
        table.index.name = "Rebalance date"
        if len(table.drop_duplicates()) == 1:      # fixed-weight benchmarks
            fixed = ", ".join(f"{a} {table[a].iloc[0]:.1f}%" for a in assets)
            print(f"\n{name}: same target weights every month ({fixed})")
            continue
        print(f"\nTARGET WEIGHTS AT EACH REBALANCE (%): {name}")
        print(table.to_string())
