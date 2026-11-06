# ============================================================
# CAPSTONE PROJECT: MULTI-PERIOD OPTIMIZATION (MPO) WALK-FORWARD
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   The same monthly walk-forward as walk_forward.py (rolling five-year
#   window, month-length historical scenarios, weights drift within the
#   month), but each rebalance is chosen by multi-period optimization
#   (Boyd et al., 2017, Section 5) instead of a one-month optimization.
#
#   At the start of every month the optimizer:
#     1. takes the current (drifted) weights w0 as given,
#     2. plans weights w1, ..., wH for the next H months, trading off
#        expected return, risk and trading cost in every planned month,
#     3. trades only to w1. Next month the whole plan is rebuilt.
#   With H = 1 this is single-period optimization with trading costs.
#
# How each baseline model becomes an MPO model:
#   MVO (mean-variance)   maximize  sum_k [ mu'w_k - (lambda/2) w_k' S_k w_k
#                                           - cost(w_k - w_{k-1}) ]
#                         Max Sharpe is a ratio, so it has no per-month
#                         penalty form; the mean-variance utility is the
#                         standard MPO version (Boyd et al., eq. 5.2).
#   MVO (min variance)    the same with mu = 0.
#   Risk parity, equal    these produce a target portfolio rather than an
#   weight, 60/40,        objective. Each month the target is recomputed
#   max Sharpe            and MPO decides how far to trade toward it:
#   ("tracking")          minimize  sum_k [ (lambda_track/2) (w_k - t)' S_k (w_k - t)
#                                           + cost(w_k - w_{k-1}) ]
#                         i.e. active risk against the target (Boyd et al.,
#                         Section 4.2) traded off against the cost of moving.
#   cost(z) = GAMMA_TRADE * sum_i cost_rate_i * |z_i|  (COST_BPS below).
#
# Covariance forecasts over the horizon (COV_FORECAST):
#   "constant"  S_k = five-year scenario covariance for every k. With
#               constant forecasts, MPO with horizon H behaves like H = 1
#               with the trading cost spread over H months, so H mostly
#               acts as a trade-aversion knob.
#   "decay"     S_k = d^k * S_short + (1 - d^k) * S_long: today's risk is
#               closer to the recent (SHORT_WINDOW-day) covariance and
#               reverts to the five-year level over the horizon. This is
#               the case where planning ahead can change the trade.
#               A GARCH/DCC forecast per month can replace this later.
#
# Reference rows ("SPO, cost-unaware"): H = 1 with GAMMA_TRADE = 0, i.e.
#   each model trades fully to its one-month optimum and costs are only
#   charged afterwards. For risk parity, equal weight, 60/40 and max Sharpe
#   these are exactly the models in walk_forward.py.
#
# Inputs:  portfolio_train.csv, portfolio_validation.csv
# Outputs: mpo_results.csv, mpo_weights.csv, mpo_returns.csv
#
# Notes:
#   - The test set is NOT used. Keep INCLUDE_TEST = False until the
#     settings (H, GAMMA_TRADE, RISK_AVERSION, ...) are chosen on validation.
#   - Planning uses the simplified dynamics w_{k+1} = w_k + z_k (returns
#     within the plan are ignored). Only w1 is traded, and the simulation
#     itself uses the true drift, as in walk_forward.py.
#   - Every future month in the plan is assumed to have the same number of
#     trading days as the current month.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TRADING_DAYS = 252
RISK_FREE = 0.0
ALPHA = 0.95          # only used for the VaR/CVaR performance metrics
BTC_CAP = 0.20        # same as the other scripts (MVO models); None for no cap
INCLUDE_TEST = False
ROLLING_YEARS = 5

HORIZONS = [1, 2, 3, 6]   # planning horizons H to test, in months

# Trading cost per asset in bps of the amount traded (buys and sells).
# MPO needs a non-zero cost: with no cost it reduces to repeated SPO.
COST_BPS = {"Equity": 10.0, "Bond": 10.0, "BTC": 10.0}
GAMMA_TRADE = 1.0     # scales the cost inside the optimizer (1 = true cost)

RISK_AVERSION = 2.5   # lambda for the mean-variance models (same as BL)
TRACKING_AVERSION = 2.5   # lambda_track for the tracking models

COV_FORECAST = "decay"    # "constant" or "decay" (see header)
SHORT_WINDOW = 63         # trading days in the recent covariance
COV_DECAY = 0.5           # d: share of the recent covariance left after one month

EXPECTED_RETURNS = "historical"   # "historical" or "bl" (uses bl_core.py)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
output = Path(__file__).resolve().parent


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------

def find_data_dir():
    """Data/datasets under the project root (as walk_forward.py), else beside
    this file or in a parent folder (as bl_core.py)."""
    candidates = [PROJECT_ROOT / "Data" / "datasets", output, *output.parents]
    for folder in candidates:
        if (folder / "portfolio_train.csv").exists():
            return folder
    raise FileNotFoundError("portfolio_train.csv not found.")


DATA_DIR = find_data_dir()


def load(name):
    return pd.read_csv(DATA_DIR / f"portfolio_{name}.csv",
                       index_col=0, parse_dates=True).sort_index()


train, validation = load("train"), load("validation")
frames = [train, validation] + ([load("test")] if INCLUDE_TEST else [])
assets = list(train.columns)
n = len(assets)
returns = pd.concat(frames)[assets].sort_index()
if returns.index.hasnans or not returns.index.is_unique or returns.isna().any().any():
    raise ValueError("Returns must have unique dates and no missing values.")

cost_rate = np.array([COST_BPS.get(a, 0.0) for a in assets]) / 10_000

upper_capped = np.ones(n)
if BTC_CAP is not None and "BTC" in assets:
    upper_capped[assets.index("BTC")] = BTC_CAP
upper_free = np.ones(n)

if EXPECTED_RETURNS == "bl":
    import bl_core as bl


# ------------------------------------------------------------
# STEP 2: FORECASTS FOR EACH MONTH OF THE HORIZON
# ------------------------------------------------------------

def historical_scenarios(history, holding_days):
    """Overlapping month-length compounded returns (as in bl_core.py)."""
    growth = (1 + history).rolling(holding_days).apply(np.prod, raw=True)
    return (growth - 1).dropna().to_numpy()


def forecasts(history, holding_days, H):
    """
    Month-length expected returns and covariances for months 1..H ahead.
    Returns (mu, [S_1, ..., S_H], S_long), all on the holding-period scale.
    """
    R = historical_scenarios(history, holding_days)
    S_long = np.cov(R, rowvar=False)
    if EXPECTED_RETURNS == "bl":
        mu = bl.black_litterman(R, assets)["mu_bl"]
    else:
        mu = R.mean(axis=0)

    if COV_FORECAST == "constant":
        covs = [S_long] * H
    elif COV_FORECAST == "decay":
        S_short = np.cov(history.values[-SHORT_WINDOW:], rowvar=False) * holding_days
        covs = [COV_DECAY ** k * S_short + (1 - COV_DECAY ** k) * S_long
                for k in range(1, H + 1)]
    else:
        raise ValueError("COV_FORECAST must be 'constant' or 'decay'.")
    return mu, covs, S_long


# ------------------------------------------------------------
# STEP 3: TARGET PORTFOLIOS (the single-period baseline models)
# ------------------------------------------------------------

def clean(w, upper):
    w = np.clip(w, 0.0, upper)
    return w / w.sum()


def max_sharpe(mu, cov, upper):
    w0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: -(w @ mu - RISK_FREE) / np.sqrt(w @ cov @ w), w0,
                   bounds=[(0.0, u) for u in upper],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Max Sharpe did not converge: {res.message}")
    return clean(res.x, upper)


def risk_parity(cov):
    res = minimize(lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / n,
                   1.0 / np.sqrt(np.diag(cov)), bounds=[(1e-9, None)] * n,
                   method="L-BFGS-B")
    return res.x / res.x.sum()


def targets(mu, S_long, holding_days):
    """Target weights of the models that MPO tracks."""
    scale = TRADING_DAYS / holding_days          # annualize, as fit_models does
    out = {
        "Max Sharpe (tracking)": max_sharpe(mu * scale, S_long * scale, upper_capped),
        "Risk parity (tracking)": risk_parity(S_long * scale),
        "Equal weight (tracking)": np.full(n, 1.0 / n),
    }
    if {"Equity", "Bond"} <= set(assets):
        w = np.zeros(n)
        w[assets.index("Equity")], w[assets.index("Bond")] = 0.6, 0.4
        out["60/40 (tracking)"] = w
    return out


# ------------------------------------------------------------
# STEP 4: THE MPO PROBLEM
# ------------------------------------------------------------

def solve_mpo(w0, covs, upper, mu=None, target=None, lam=RISK_AVERSION,
              gamma_trade=GAMMA_TRADE):
    """
    Plan weights for len(covs) months and return the first month's weights.

    w0      current (drifted) weights; zeros before the first purchase
    mu      expected holding-period returns (mean-variance models), or None
    target  target weights (tracking models), or None
    Exactly one of mu / target is used: with target, risk is active risk
    (w - target)' S (w - target); otherwise it is total risk w' S w.
    """
    H = len(covs)
    W = cp.Variable((H, n))
    objective, previous = 0, w0
    for k in range(H):
        S = cp.psd_wrap((covs[k] + covs[k].T) / 2)
        wk = W[k]
        if target is not None:
            objective -= lam / 2 * cp.quad_form(wk - target, S)
        else:
            objective += mu @ wk - lam / 2 * cp.quad_form(wk, S)
        objective -= gamma_trade * cost_rate @ cp.abs(wk - previous)
        previous = wk
    constraints = [cp.sum(W, axis=1) == 1, W >= 0, W <= np.tile(upper, (H, 1))]
    problem = cp.Problem(cp.Maximize(100 * objective), constraints)  # x100: scaling only
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MPO problem status: {problem.status}")
    return clean(W.value[0], upper), W.value


def fit_all(history, holding_days, H, drifted, gamma_trade):
    """Weights to trade to this month for every model, at horizon H."""
    mu, covs, S_long = forecasts(history, holding_days, H)
    zero = np.zeros(n)
    out = {}
    out["MVO (mean-variance)"] = solve_mpo(
        drifted.get("MVO (mean-variance)", zero), covs, upper_capped, mu=mu,
        gamma_trade=gamma_trade)[0]
    out["MVO (min variance)"] = solve_mpo(
        drifted.get("MVO (min variance)", zero), covs, upper_capped, mu=np.zeros(n),
        gamma_trade=gamma_trade)[0]
    for name, t in targets(mu, S_long, holding_days).items():
        caps = upper_capped if name.startswith("Max Sharpe") else upper_free
        out[name] = solve_mpo(drifted.get(name, zero), covs, caps, target=t,
                              lam=TRACKING_AVERSION, gamma_trade=gamma_trade)[0]
    return out


# ------------------------------------------------------------
# STEP 5: WALK FORWARD (one run per horizon, plus the cost-unaware reference)
# ------------------------------------------------------------

oos_start = validation.index.min()
months = returns.index.to_period("M")
oos_months = sorted(set(months[returns.index >= oos_start]))
if returns.index[months == oos_months[0]][0] < oos_start:
    raise ValueError("The validation set must start at the beginning of a month.")

runs = [("SPO, cost-unaware", 1, 0.0)] + [(f"MPO H={H}", H, GAMMA_TRADE) for H in HORIZONS]

daily, turnover, costs, weight_rows = {}, {}, {}, []

for run_label, H, gamma_trade in runs:
    drifted = {}
    for m in oos_months:
        days = returns[months == m]
        date = days.index[0]
        history = returns.loc[(returns.index < date) &
                              (returns.index >= date - pd.DateOffset(years=ROLLING_YEARS))]
        assert history.index.max() < days.index.min(), "Look-ahead detected."

        try:
            weights = fit_all(history, len(days), H, drifted, gamma_trade)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"{run_label}, rebalance {date.date()}: {error}") from error

        growth = (1 + days.values).cumprod(axis=0)
        for name, w in weights.items():
            key = (name, run_label)
            if name in drifted:
                turnover.setdefault(key, []).append(0.5 * np.abs(w - drifted[name]).sum())
            cost = float(cost_rate @ np.abs(w - drifted.get(name, np.zeros(n))))
            costs.setdefault(key, []).append(cost)
            wealth = (growth @ w) * (1.0 - cost)
            previous = np.concatenate([[1.0], wealth[:-1]])
            daily.setdefault(key, []).append(pd.Series(wealth / previous - 1.0, index=days.index))
            end_value = w * growth[-1]
            drifted[name] = end_value / end_value.sum()
            weight_rows.append({"Run": run_label, "Month": str(m), "Portfolio": name,
                                **{a: w[i] for i, a in enumerate(assets)}})

oos_returns = pd.DataFrame({f"{name} | {run}": pd.concat(parts)
                            for (name, run), parts in daily.items()})
weights_df = pd.DataFrame(weight_rows)


# ------------------------------------------------------------
# STEP 6: PERFORMANCE (net of trading costs)
# ------------------------------------------------------------

def performance(r):
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


rows = []
for (name, run), parts in daily.items():
    row = {"Portfolio": name, "Run": run,
           **performance(pd.concat(parts).values),
           "Avg Monthly Turnover": np.mean(turnover.get((name, run), [np.nan])),
           "Total Costs Paid": np.sum(costs[(name, run)])}
    sub = weights_df[(weights_df["Portfolio"] == name) & (weights_df["Run"] == run)]
    if "BTC" in assets:
        row["Avg BTC Weight"] = sub["BTC"].mean()
    rows.append(row)
results = pd.DataFrame(rows).set_index(["Portfolio", "Run"])

results.to_csv(output / "mpo_results.csv")
weights_df.to_csv(output / "mpo_weights.csv", index=False)
oos_returns.to_csv(output / "mpo_returns.csv")


# ------------------------------------------------------------
# STEP 7: PRINT
# ------------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Out-of-sample: {returns.index[returns.index >= oos_start].min().date()} to "
      f"{returns.index.max().date()} ({len(oos_months)} monthly rebalances)")
print(f"Costs (bps): " + ", ".join(f"{a} {COST_BPS.get(a, 0):g}" for a in assets) +
      f" | gamma_trade {GAMMA_TRADE} | lambda {RISK_AVERSION} | lambda_track {TRACKING_AVERSION}")
print(f"Covariance forecast: {COV_FORECAST}"
      + (f" (recent {SHORT_WINDOW} days, decay {COV_DECAY}/month)" if COV_FORECAST == "decay" else "")
      + f" | expected returns: {EXPECTED_RETURNS} | BTC cap {BTC_CAP}")

run_order = [r[0] for r in runs]
for metric, pct in (("Sharpe", False), ("Ann. Return", True), ("Max Drawdown", True),
                    ("Avg Monthly Turnover", True), ("Total Costs Paid", True)):
    table = results[metric].unstack("Run")[run_order]
    table = (table * 100).round(2) if pct else table.round(3)
    print(f"\n{metric.upper()}" + (" (%)" if pct else "") + " by model and run:")
    print(table.to_string())

print("\nSaved: mpo_results.csv, mpo_weights.csv, mpo_returns.csv")
