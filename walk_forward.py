# ============================================================
# CAPSTONE PROJECT: WALK-FORWARD TEST OF THE BASELINE MODELS
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Re-runs each baseline model month by month, the way it would be used
#   in practice. At the start of every month it:
#     1. fits on month-length historical scenarios from the prior five years,
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
#     max-Sharpe expected return minus 1e-6 per holding day.
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
#   - Historical scenarios, max Sharpe, and CVaR use the shared comparison
#     helpers to keep the historical strategies consistent.
#   - Holding-day counts are retrospective; future return values are excluded.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio_optimization import checked_solution, validate_weights
from cvar_optimization import min_cvar as optimize_cvar
from cvar_scenario_comparison import historical_scenarios, matched_return_portfolios

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

ALPHA = 0.95          # CVaR confidence level
TRADING_DAYS = 252
RISK_FREE = 0.0       # annual risk-free rate for the Sharpe ratio
BTC_CAP = 0.20        # keep the same as the other scripts; None for no cap
INCLUDE_TEST = False  # keep False until the models are finalized
ROLLING_YEARS = 5
PRINT_WEIGHTS = True  # print the weights chosen at every monthly rebalance

# Trading cost per asset, in basis points (1 bp = 0.01%) of the amount traded,
# charged on both buys and sells. 0 means no cost. Use your own estimates; the
# values here are placeholders and should be agreed on by the team. Assets not
# listed are treated as free to trade.
COST_BPS = {"Equity": 0.0, "Bond": 0.0, "BTC": 0.0}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "datasets"
output = Path(__file__).resolve().parent



# STEP 1: LOAD DATA


def load(name):
    return pd.read_csv(DATA_DIR / f"portfolio_{name}.csv",
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

# STEP 2: MODELS (each takes only past data)


def min_variance(cov):
    res = minimize(lambda w: w @ cov @ w, w_equal, bounds=bounds,
                   constraints=[sum_to_one], method="SLSQP")
    return checked_solution(res, upper, "MVO (min variance)")


def risk_parity(cov):
    """Equal risk contribution (convex formulation, rescaled to sum to 1)."""
    def objective(y):
        return 0.5 * y @ cov @ y - np.log(y).sum() / n
    res = minimize(objective, 1.0 / np.sqrt(np.diag(cov)),
                   bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return checked_solution(res, np.ones(n), "Risk parity", normalize=True)


def min_cvar(R, min_return=None):
    """Shared CVaR LP on month-length historical return scenarios."""
    return optimize_cvar(R, alpha=ALPHA, upper=upper.tolist(), min_return=min_return)


def fit_all(R, holding_days):
    """Fit every model on prior-five-year, month-length scenarios."""
    cov = np.cov(R, rowvar=False) * TRADING_DAYS/holding_days
    target, w_ms, w_matched = matched_return_portfolios(
        R, upper.tolist(), holding_days=holding_days,
        risk_free=RISK_FREE*holding_days/TRADING_DAYS)
    models = {
        "MVO (max Sharpe)": w_ms,
        "MVO (min variance)": min_variance(cov),
        "Risk parity": risk_parity(cov),
        "Min CVaR": min_cvar(R),
        # Approximately match max-Sharpe expected return with a minimum target.
        "CVaR (matched return)": w_matched,
        "Equal weight": w_equal,
    }
    if {"Equity", "Bond"} <= set(assets):
        w = np.zeros(n)
        w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
        models["60/40 equity/bond"] = w
    for name, weights in models.items():
        caps = np.ones(n) if name in ("Risk parity", "Equal weight", "60/40 equity/bond") else upper
        validate_weights(weights, caps, name, mean=R.mean(axis=0),
                         min_return=target if name == "CVaR (matched return)" else None)
    return models



# STEP 3: WALK FORWARD (rolling five-year window, monthly rebalancing)


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

for m in oos_months:
    days = returns[months == m]

    # Use only observations in the five years strictly before rebalance.
    date = days.index[0]
    history = returns.loc[(returns.index < date) &
                          (returns.index >= date-pd.DateOffset(years=ROLLING_YEARS))]
    assert history.index.max() < days.index.min(), "Look-ahead detected."

    scenarios = historical_scenarios(history, len(days))
    try:
        weights = fit_all(scenarios, len(days))
    except (RuntimeError, ValueError, AssertionError) as error:
        raise RuntimeError(f"Rebalance {date.date()}: {error}") from error

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


# STEP 5: PRINT


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Out-of-sample period: {oos_returns.index.min().date()} to "
      f"{oos_returns.index.max().date()} "
      f"({len(oos_returns)} days, {len(oos_months)} monthly rebalances)")
print("Window: rolling five years | Scenarios: month-length historical returns | Rebalancing: monthly")
print(f"BTC cap (MVO and CVaR models): {BTC_CAP}")
print("Trading costs (bps per amount traded): " +
      ", ".join(f"{a} {COST_BPS.get(a, 0.0):g}" for a in assets) +
      ("  [no costs charged]" if not cost_rate.any() else ""))

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
