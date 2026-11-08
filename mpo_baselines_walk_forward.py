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
#     CVaR_MPO/cvar_mpo.py       Matched-return minimum CVaR plus trading costs
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
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CVAR_DIRECTORY = str(PROJECT_ROOT / "CVaR_MPO")
if _CVAR_DIRECTORY not in sys.path:
    sys.path.insert(0, _CVAR_DIRECTORY)

import Multi_Period_Baselines as mb
import cvar_mpo as mc

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
CVAR_RISK_AVERSION = 0.5            # same risk/cost trade-off as matched Hybrid DCC study
INCLUDE_CVAR = True                # also run the teammate's CVaR planner
CVAR_MODEL = "CVaR (matched return)"

MODELS = list(mb.BASELINES) + ([CVAR_MODEL] if INCLUDE_CVAR else [])

output = Path(__file__).resolve().parent


# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------

def find_data_dir():
    for folder in [PROJECT_ROOT / "Data" / "datasets", output, *output.parents]:
        if (folder / "portfolio_train.csv").exists():
            return folder
    raise FileNotFoundError("portfolio_train.csv not found; run Splitting data.py first.")


def load(name, data_dir=None):
    data_dir = find_data_dir() if data_dir is None else Path(data_dir)
    return pd.read_csv(data_dir / f"portfolio_{name}.csv",
                       index_col=0, parse_dates=True).sort_index()


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
    if model == CVAR_MODEL:
        # Preserve the single-period reference: native scenario means and the
        # original maximum-Sharpe target minus 1e-6 per holding day. Keep the
        # existing CVaR-plus-cost planner and exact monthly execution unchanged.
        plan = mc.optimize_matched_cvar(
            bundle.returns['historical'][:, :horizon], w_prev,
            [len(period) for period in bundle.periods[:horizon]],
            native_layout='F',
            upper=(1., 1., BTC_CAP), cost_rate=COST_RATE,
            risk_aversion=CVAR_RISK_AVERSION, alpha=.95, risk_free=0.)
        plan.metadata['forecast_model'] = 'historical'
        return plan
    return mb.optimize_forecast(bundle, "historical", w_prev, baseline=model, horizon=horizon,
                                btc_cap=BTC_CAP, cost_rate=COST_RATE,
                                risk_aversion=MV_RISK_AVERSION)


# ------------------------------------------------------------
# STEP 3: WALK FORWARD
# ------------------------------------------------------------

def run_walk_forward(returns, oos_start, *, horizons=None, models=None, progress=True):
    """Plan from strictly prior returns and execute only the next actual month.

    Kept separate from main so importing this module never loads a dataset,
    runs a backtest, or overwrites results.
    """
    horizons = HORIZONS if horizons is None else horizons
    models = MODELS if models is None else models
    returns = returns.loc[:, list(mb.ASSETS)].copy()
    if (returns.empty or not returns.index.is_monotonic_increasing or
            returns.index.has_duplicates or not np.isfinite(returns.to_numpy()).all() or
            (returns <= -1).any().any()):
        raise ValueError("Require ordered, unique dates and finite daily simple returns above -1")
    months = returns.index.to_period("M")
    oos_months = sorted(set(months[returns.index >= pd.Timestamp(oos_start)]))
    if not oos_months or not horizons or not models:
        raise ValueError("Supply at least one evaluation month, horizon and model")
    daily, stats, weight_rows = {}, {}, []
    if progress:
        print(f"Starting validation walk-forward: {len(oos_months)} months, "
              f"{len(models) * len(horizons)} strategy/horizon runs. "
              "Progress is printed below; results are saved when all runs finish.", flush=True)
    for h_index, H in enumerate(horizons):
        for model_index, model in enumerate(models):
            case_number = h_index * len(models) + model_index + 1
            case_started = time.perf_counter()
            if progress:
                print(f"[{case_number:02d}/{len(models) * len(horizons)}] H={H}: {model}", flush=True)
            w_prev, parts, turnover, fees = np.zeros(3), [], [], []
            for month_index, m in enumerate(oos_months):
                days = returns[months == m]
                date = days.index[0]
                history = returns[(returns.index < date) &
                                  (returns.index >= date-pd.DateOffset(years=ROLLING_YEARS))]
                if history.empty or history.index.max() >= date:
                    raise ValueError("Require nonempty history strictly before the rebalance")
                try:
                    plan = plan_for(model, make_bundle(history, days, H), w_prev, H)
                except (RuntimeError, ValueError, AssertionError) as error:
                    raise RuntimeError(f"{model}, H={H}, {date.date()}: {error}") from error
                executed = mb.execute_first_period(plan, days.values)
                if w_prev.sum() > 0:
                    turnover.append(0.5*np.abs(plan.first_weights-w_prev).sum())
                fees.append(executed.fee_fraction)
                parts.append(pd.Series(executed.net_daily_returns, index=days.index))
                floors = plan.metadata.get("gross_return_targets")
                weight_rows.append({"Model": model, "H": H, "Month": str(m),
                    **dict(zip(mb.ASSETS, plan.first_weights)),
                    "Expected Return": plan.expected_returns[0],
                    "Return Floor": None if floors is None else floors[0],
                    "Return Reward": plan.metadata["return_reward"],
                    "Risk Aversion": plan.metadata["risk_aversion"],
                    "Forecast Model": plan.metadata["forecast_model"]})
                w_prev = executed.ending_weights
                if progress and ((month_index+1) % 7 == 0 or month_index+1 == len(oos_months)):
                    print(f"  {month_index+1}/{len(oos_months)} months complete "
                          f"({time.perf_counter()-case_started:.1f}s)", flush=True)
            daily[(model, H)] = pd.concat(parts)
            stats[(model, H)] = (float(np.mean(turnover)) if turnover else 0., float(np.sum(fees)))
    return daily, stats, pd.DataFrame(weight_rows)


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


def main():
    print(f"Python interpreter: {sys.executable}", flush=True)
    data_dir = find_data_dir()
    train, validation = load("train", data_dir), load("validation", data_dir)
    frames = [train, validation]+([load("test", data_dir)] if INCLUDE_TEST else [])
    returns = pd.concat(frames)[list(mb.ASSETS)].sort_index()
    daily, stats, weights_df = run_walk_forward(returns, validation.index.min())
    results = pd.DataFrame({key: {**performance(r), "Avg Monthly Turnover": stats[key][0],
                                  "Total Costs Paid": stats[key][1]}
                           for key, r in daily.items()}).T
    results.index.names = ["Model", "H"]
    results.to_csv(output/"mpo_baseline_results.csv")
    weights_df.to_csv(output/"mpo_baseline_weights.csv", index=False)
    pd.DataFrame({f"{m} | H={h}": r for (m, h), r in daily.items()}).to_csv(output/"mpo_baseline_returns.csv")
    result_names = ("mpo_baseline_results.csv", "mpo_baseline_weights.csv", "mpo_baseline_returns.csv")
    metadata = {
        "forecast_distribution": "overlapping historical monthly returns repeated across stages",
        "expected_returns": "native historical monthly scenario means; no Black-Litterman or Hybrid DCC inputs",
        "cvar_model": CVAR_MODEL, "cvar_function": "cvar_mpo.optimize_matched_cvar",
        "cvar_return_reward": 0., "cvar_risk_aversion": CVAR_RISK_AVERSION,
        "cvar_return_floors": "original maximum-Sharpe expected return from native scenario means minus 1e-6 per holding day; same caps",
        "alpha": .95, "btc_cap": BTC_CAP, "cost_rate": COST_RATE,
        "horizons": HORIZONS, "include_test": INCLUDE_TEST,
        "output_sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in result_names}}
    (output/"mpo_baseline_run_metadata.json").write_text(json.dumps(metadata, indent=2)+'\n', encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    first = daily[(MODELS[0], HORIZONS[0])]
    print(f"Out-of-sample: {first.index.min().date()} to {first.index.max().date()} "
          f"({first.index.to_period('M').nunique()} rebalances) | cost {COST_RATE*1e4:g} bps | "
          f"BTC cap {BTC_CAP} | risk aversion MV {MV_RISK_AVERSION}, CVaR {CVAR_RISK_AVERSION}")
    for metric, pct in (("Sharpe", False), ("Ann. Return", True), ("Max Drawdown", True),
                        ("Avg Monthly Turnover", True), ("Total Costs Paid", True)):
        table = results[metric].astype(float).unstack("H")
        table.columns = [f"H={h}" for h in table.columns]
        print(f"\n{metric.upper()}"+(" (%)" if pct else "")+":")
        print(((table*100) if pct else table).round(2 if pct else 3).to_string())
    print("\nSaved: "+", ".join(result_names)+", mpo_baseline_run_metadata.json")
    return results


if __name__ == "__main__":
    main()
