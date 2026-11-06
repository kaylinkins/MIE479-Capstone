# ============================================================
# CAPSTONE PROJECT: MPO WALK-FORWARD FOR THE BASELINE MODELS
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# What this file does:
#   Runs the monthly walk-forward (as walk_forward.py) with every rebalance
#   chosen by the multi-period planners:
#     Multi_Period_Baselines.py   MVO, min variance, max Sharpe, risk parity,
#                                 equal weight, 60/40
#     multiperiod_cvar.py        CVaR (teammate's planner), for comparison
#   Each month it builds a forecast bundle from the five years before the
#   month, plans H months ahead, trades to the first planned weights, and
#   lets them drift through the month (execute_first_period).
#
# Run order:  Data_Collection_and_Cleaning.py -> Splitting data.py -> this file
# Outputs:    mpo_baseline_results.csv, mpo_baseline_weights.csv,
#             mpo_baseline_returns.csv
#
# Forecasts: the bundle here uses month-length historical scenarios (same
#   distribution for every planned month), so results line up with
#   walk_forward.py / Table 5. To use GARCH/DCC scenarios, replace
#   make_bundle() with the shared forecast bundle (same fields).
# Keep INCLUDE_TEST = False until settings are chosen on validation.
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import Multi_Period_Baselines as mb
import multiperiod_cvar as mc

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

INCLUDE_TEST = False
ROLLING_YEARS = 5
TRADING_DAYS = 252
HORIZONS = [1, 2, 3, 6]            # planning horizons to test (months)
COST_RATE = 0.001                  # 10 bps per dollar traded, all assets
BTC_CAP = 0.20
MV_RISK_AVERSION = mb.DEFAULT_RISK_AVERSION   # 1.25 = lambda/2, lambda = 2.5
CVAR_RISK_AVERSION = 1.0
INCLUDE_CVAR = True                # also run the teammate's CVaR planner

MODELS = list(mb.BASELINES) + (["CVaR (teammate)"] if INCLUDE_CVAR else [])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
output = Path(__file__).resolve().parent


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------

def find_data_dir():
    for folder in [PROJECT_ROOT / "Data" / "datasets", output, *output.parents]:
        if (folder / "portfolio_train.csv").exists():
            return folder
    raise FileNotFoundError("portfolio_train.csv not found; run Splitting data.py first.")


DATA_DIR = find_data_dir()


def load(name):
    return pd.read_csv(DATA_DIR / f"portfolio_{name}.csv",
                       index_col=0, parse_dates=True).sort_index()


train, validation = load("train"), load("validation")
frames = [train, validation] + ([load("test")] if INCLUDE_TEST else [])
returns = pd.concat(frames)[list(mb.ASSETS)].sort_index()


# ------------------------------------------------------------
# STEP 2: FORECAST BUNDLE (historical scenarios)
# ------------------------------------------------------------

@dataclass
class Bundle:
    assets: tuple
    returns: dict
    expected_returns: np.ndarray
    periods: list
    origin: pd.Timestamp
    metadata: dict = field(default_factory=dict)


def make_bundle(history, days, horizon):
    """Month-length overlapping historical scenarios, repeated for each planned month."""
    growth = (1 + history).rolling(len(days)).apply(np.prod, raw=True)
    R = (growth - 1).dropna().to_numpy()
    return Bundle(mb.ASSETS, {"historical": np.repeat(R[:, None, :], horizon, axis=1)},
                  np.repeat(R.mean(axis=0)[None], horizon, axis=0),
                  [days.index] * horizon, days.index[0])


def plan_for(model, bundle, w_prev, horizon):
    if model == "CVaR (teammate)":
        return mc.optimize_forecast(bundle, "historical", w_prev, horizon=horizon,
                                    upper=(1., 1., BTC_CAP), cost_rate=COST_RATE,
                                    risk_aversion=CVAR_RISK_AVERSION)
    return mb.optimize_forecast(bundle, "historical", w_prev, baseline=model, horizon=horizon,
                                btc_cap=BTC_CAP, cost_rate=COST_RATE,
                                risk_aversion=MV_RISK_AVERSION)


# ------------------------------------------------------------
# STEP 3: WALK FORWARD
# ------------------------------------------------------------

months = returns.index.to_period("M")
oos_months = sorted(set(months[returns.index >= validation.index.min()]))
daily, stats, weight_rows = {}, {}, []

for H in HORIZONS:
    for model in MODELS:
        w_prev, parts, turnover, fees = np.zeros(3), [], [], []
        for m in oos_months:
            days = returns[months == m]
            date = days.index[0]
            history = returns[(returns.index < date) &
                              (returns.index >= date - pd.DateOffset(years=ROLLING_YEARS))]
            assert history.index.max() < date, "Look-ahead detected."
            try:
                plan = plan_for(model, make_bundle(history, days, H), w_prev, H)
            except (RuntimeError, ValueError, AssertionError) as error:
                raise RuntimeError(f"{model}, H={H}, {date.date()}: {error}") from error
            executed = mb.execute_first_period(plan, days.values)
            if w_prev.sum() > 0:
                turnover.append(0.5 * np.abs(plan.first_weights - w_prev).sum())
            fees.append(executed.fee_fraction)
            parts.append(pd.Series(executed.net_daily_returns, index=days.index))
            weight_rows.append({"Model": model, "H": H, "Month": str(m),
                                **dict(zip(mb.ASSETS, plan.first_weights))})
            w_prev = executed.ending_weights
        daily[(model, H)] = pd.concat(parts)
        stats[(model, H)] = (np.mean(turnover), np.sum(fees))


# ------------------------------------------------------------
# STEP 4: PERFORMANCE (net of trading costs)
# ------------------------------------------------------------

def performance(r):
    r = r.to_numpy()
    ann_ret, ann_vol = r.mean() * TRADING_DAYS, r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    var, cvar = mc.loss_var_cvar(r, 0.95)
    wealth = np.cumprod(1 + r)
    return {"Total Return": wealth[-1] - 1, "Ann. Return": ann_ret, "Ann. Volatility": ann_vol,
            "Sharpe": ann_ret / ann_vol, "95% VaR (daily)": var, "95% CVaR (daily)": cvar,
            "Max Drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min()}


results = pd.DataFrame({key: {**performance(r), "Avg Monthly Turnover": stats[key][0],
                              "Total Costs Paid": stats[key][1]}
                        for key, r in daily.items()}).T
results.index.names = ["Model", "H"]
weights_df = pd.DataFrame(weight_rows)

results.to_csv(output / "mpo_baseline_results.csv")
weights_df.to_csv(output / "mpo_baseline_weights.csv", index=False)
pd.DataFrame({f"{m} | H={h}": r for (m, h), r in daily.items()}).to_csv(
    output / "mpo_baseline_returns.csv")


# ------------------------------------------------------------
# STEP 5: PRINT
# ------------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print(f"Out-of-sample: {daily[(MODELS[0], HORIZONS[0])].index.min().date()} to "
      f"{daily[(MODELS[0], HORIZONS[0])].index.max().date()} ({len(oos_months)} rebalances) | "
      f"cost {COST_RATE * 1e4:g} bps | BTC cap {BTC_CAP} | "
      f"risk aversion MV {MV_RISK_AVERSION}, CVaR {CVAR_RISK_AVERSION}")
for metric, pct in (("Sharpe", False), ("Ann. Return", True), ("Max Drawdown", True),
                    ("Avg Monthly Turnover", True), ("Total Costs Paid", True)):
    table = results[metric].astype(float).unstack("H")
    table.columns = [f"H={h}" for h in table.columns]
    print(f"\n{metric.upper()}" + (" (%)" if pct else "") + ":")
    print(((table * 100) if pct else table).round(2 if pct else 3).to_string())
print("\nSaved: mpo_baseline_results.csv, mpo_baseline_weights.csv, mpo_baseline_returns.csv")
