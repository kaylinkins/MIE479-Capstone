"""Backtest: walk-forward evaluation of the regime MVO decision models.

Run from the repository root:
    python Regime_MVO_MPO/regime_mvo_backtest.py
    python Regime_MVO_MPO/regime_mvo_backtest.py --model aware       # Model 1 only
    python Regime_MVO_MPO/regime_mvo_backtest.py --model switching   # Model 2 only
    python Regime_MVO_MPO/regime_mvo_backtest.py --horizons 1 3      # only some horizons (default 1 2 3)

The two decision models live in their own files (keep all three together):
  regime_aware_mvo.py      Model 1: regime-aware MVO MPO (Max Sharpe MVO and
                           Min Var MVO fed with regime-dependent forecasts)
  regime_switching_mvo.py  Model 2: regime-switching MVO MPO, bull -> Max
                           Sharpe and bear -> Min Var, switching at each stage
This file only (1) loads the forecasts and realized returns, (2) asks a model
for weights each month, (3) holds those weights through the real month and
(4) scores the result. It never reads portfolio_test.csv.

Consistency with Kaylin's regime-switching CVaR (CVaR Regime Switching/
run_switching_all_dcc.py and regime_switching_cvar.py)
----------------------------------------------------------------------
Same as hers:
  * Forecasts: native HMM, native Jump Model (training-tuned version,
    jump_tuned_outputs), and Regime regression
    (hmm_regime_conditional_returns.csv, hard-selecting the bull or bear
    conditional mean of each stage's forecast regime).
  * Regime signal: the forecast MONTH-END bear probability of each stage
    (p_bear_month{k}_end for HMM / Jump; p_bear_month_end for Regime
    regression); bull if < 0.5, bear if > 0.5, an exact tie is an error.
  * Risk model: DCC-GARCH scenarios for every forecast source (RISK_MODEL),
    regenerated with the team's scenario_generation.py (10,000 scenarios,
    seed 42, prior five years, frozen 2022-2024 NYSE calendar). The MVO
    covariance of stage k is the covariance of those month-k scenarios, and
    the max-Sharpe return floor uses the same covariance.
  * Bull stages: min risk + cost with the max-Sharpe return floor; bear
    stages: min risk + cost with no floor; the full plan is solved jointly and
    only stage one is executed.
  * Return floor = stage mean . max-Sharpe weights - 1e-6 per trading day of
    that month (cvar_mpo.matched_return_targets), with trading days from the
    frozen 2022-2024 NYSE calendar (forecast_trading_days.csv).
  * H = 1 and 3 as in her run, plus H = 2 (HORIZONS); 10 bps cost; long-only, fully invested; BTC cap 20%; start
    in cash; exact self-financing fee and daily drift; no terminal liquidation.
  * Metrics: cumulative net return; Sharpe annualized from net daily returns
    with sample standard deviation and zero risk-free rate; max drawdown on
    daily net wealth; average executed weights; one-way turnover with the
    first (from-cash) month counted as 0; conditional Sharpe in bull and bear
    months using the COMMON HMM forecast month-1-end labels for every model.
Regime-aware MVO (Model 1) uses exactly the same inputs, risk model, floor,
costs, caps, horizons, execution and metrics as the switching model. It is
ONE model: Max Sharpe MVO with the floor in every month (the MVO counterpart
of matched-return CVaR). Min Var MVO (floor in no month, the counterpart of
min CVaR) is also reported as a reference (AWARE_POLICIES): it ignores return
forecasts, so its rows are the same for every source. Its executed
regimes are reported with the same rule, for comparison only (they do not
change its decisions).
Different by design:
  * Risk measure: variance (MVO) instead of CVaR95, weight gamma = 1.25
    instead of 0.5; the two risk scales are not comparable.
  * The static (no-regime) control is off by default, as in her run
    (INCLUDE_STATIC). Equal weight is reported as a baseline.
Not verifiable here: her scenario cache (forecast_comparison_results) and
helper modules are not in the repository, so the regenerated DCC scenarios
are assumed, not checked, to equal hers.

Outputs (Regime_MVO_MPO/outputs/)
    regime_mvo_results.csv           one row per model x source x H x BTC cap
    regime_mvo_weights.csv           executed weights, stage regimes and rules
    regime_mvo_daily_returns.csv     net daily returns of every strategy
    dcc_garch_stage_covariances.csv  cached DCC-GARCH stage covariances

Limits: 28 validation months, already inspected repeatedly; treat results as
validation, not a fresh holdout.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import regime_aware_mvo as aware
import regime_switching_mvo as switching

# ------------------------------------------------------------
# BACKTEST SETTINGS
# ------------------------------------------------------------

HORIZONS = aware.HORIZONS       # (1, 2, 3); the CVaR switching run used (1, 3). Override: --horizons 1 3
BTC_CAPS = (0.20,)              # add None for an unrestricted long-only run
COST_RATE = aware.COST_RATE     # 10 bps realized fee; same rate the models plan with
RISK_MODEL = "DCC GARCH"        # "DCC GARCH" (as in the CVaR run) or "native" (each source's own covariance)
INCLUDE_STATIC = False          # static no-regime control; excluded in the CVaR run
AWARE_POLICIES = ("Max Sharpe MVO", "Min Var MVO")   # Max Sharpe MVO = the regime-aware model (MVO twin of
                                       # matched-return CVaR); Min Var MVO = reference row (twin of min CVaR). Min Var
                                       # ignores return forecasts, so it is identical for every forecast source.
N_SCENARIOS = 10000             # DCC-GARCH scenario settings, as in the CVaR run
SEED = 42
TRADING_DAYS = 252

# (model name, policy label, decision function(current_weights, forecast, horizon, caps))
MODELS = {
    "aware": [
        ("Regime-aware MVO MPO", policy,
         lambda w, f, h, c, policy=policy: aware.solve_plan(w, f, h, policy, c, cost_rate=COST_RATE))
        for policy in AWARE_POLICIES],
    "switching": [
        ("Regime-switching MVO MPO", switching.POLICY,
         lambda w, f, h, c: switching.solve_plan(w, f, h, c, cost_rate=COST_RATE))],
}

HERE = Path(__file__).resolve().parent


def find_repo_root():
    """The repository root: the first folder at or above this file that holds
    the teammates' forecast folders. Works whether these files sit in the
    repo root or in Regime_MVO_MPO/."""
    for folder in (HERE, *HERE.parents):
        if (folder / "HMM_regime_forecasts").is_dir() and (folder / "Return_Forecasting").is_dir():
            return folder
    raise FileNotFoundError("Could not find the repository root (a folder containing "
                            "HMM_regime_forecasts/ and Return_Forecasting/) above " + str(HERE))


def first_existing(*paths):
    """First path that exists, so moved output files are still found."""
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


ROOT = find_repo_root()
OUTPUT_DIR = ROOT / "Regime_MVO_MPO" / "outputs"
ASSETS = ("Equity", "Bond", "BTC")

HMM_DIR = ROOT / "HMM_regime_forecasts" / "hmm_outputs"
JUMP_DIR = ROOT / "HMM_regime_forecasts" / "jump_tuned_outputs"   # training-tuned jump model (as in the CVaR run); original: jump_outputs
REGRESSION_FILE = first_existing(                             # moved from outputs/misc/ on 2026-11-16
    ROOT / "Return_Forecasting" / "outputs" / "hmm_regime_conditional_returns.csv",
    ROOT / "Return_Forecasting" / "outputs" / "misc" / "hmm_regime_conditional_returns.csv")
REGIME_EWMA_FILE = (ROOT / "Covariance_Estimation" / "outputs"
                    / "covariances_regime_ewma_training_selected_walk_forward_validation.csv")
STATIC_RETURNS_FILE = ROOT / "Return_Forecasting" / "outputs" / "walk_forward_validation_expected_returns_h1_h3.csv"
STATIC_COV_FILE = ROOT / "Covariance_Estimation" / "outputs" / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"
SCENARIO_DIR = ROOT / "Distribution Modelling (Scenario Generation)"
DCC_CACHE = OUTPUT_DIR / "dcc_garch_stage_covariances.csv"
TRADING_DAYS_FILE = HMM_DIR / "forecast_trading_days.csv"     # frozen NYSE calendar, same for every source


# ------------------------------------------------------------
# STEP 1: LOAD REALIZED RETURNS
# ------------------------------------------------------------

def find_returns(name):
    for folder in (ROOT / "Data Work", ROOT / "Data", ROOT / "HMM_regime_forecasts", ROOT / "Data" / "datasets"):
        path = folder / f"portfolio_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
            frame = frame.loc[:, list(ASSETS)].astype(float)
            if frame.index.has_duplicates or frame.isna().any().any() or (frame <= -1).any().any():
                raise ValueError(f"{path}: invalid dates or simple returns")
            return frame
    raise FileNotFoundError(f"portfolio_{name}.csv not found")


def month_slices(returns, dates):
    """Realized daily returns from each rebalance date to the end of its month."""
    out = {}
    for date in dates:
        start = pd.Timestamp(date)
        month = returns[(returns.index >= start) & (returns.index.to_period("M") == start.to_period("M"))]
        if month.empty or month.index[0] != start:
            raise ValueError(f"Rebalance date {date} is not the first return day of its month")
        out[date] = month
    return out


# ------------------------------------------------------------
# STEP 2: LOAD FORECASTS (keyed by rebalance date, months 1-3)
# ------------------------------------------------------------

def read_wide_returns(path):
    frame = pd.read_csv(path)
    if "origin_type" in frame.columns:
        frame = frame[frame["origin_type"] == "validation"]
    out = {}
    for date, group in frame.groupby("rebalance_date"):
        group = group.sort_values("month_ahead")
        if list(group["month_ahead"]) != [1, 2, 3]:
            raise ValueError(f"{path.name}: {date} lacks horizons 1-3")
        out[date] = group
    return out


def read_long_covariances(path):
    frame = pd.read_csv(path)
    out = {}
    for date, group in frame.groupby("rebalance_date"):
        mats = []
        for k in (1, 2, 3):
            block = group[group["month_ahead"] == k].set_index("asset").loc[list(ASSETS), list(ASSETS)]
            matrix = block.to_numpy(float)
            mats.append((matrix + matrix.T) / 2)
        out[date] = np.asarray(mats)
    return out


def load_file_source(folder):
    """HMM / Jump Model: native means and covariances; regime signal is the
    forecast month-end bear probability of each stage (as in the CVaR run)."""
    returns = read_wide_returns(folder / "expected_returns.csv")
    covs = read_long_covariances(folder / "covariances.csv")
    probs = pd.read_csv(folder / "regime_probabilities.csv").set_index("rebalance_date")
    out = {}
    for date, group in returns.items():
        p = probs.loc[date, [f"p_bear_month{k}_end" for k in (1, 2, 3)]].to_numpy(float)
        out[date] = aware.Forecast(group[list(ASSETS)].to_numpy(float), covs[date], p)
    return out


def load_regression_source():
    """Regime regression estimates (hmm_regime_conditional_returns.csv): select
    each stage's regime from its month-end bear probability, then use that
    regime's bull or bear conditional estimate. Native covariance: regime EWMA."""
    returns = read_wide_returns(REGRESSION_FILE)
    covs = read_long_covariances(REGIME_EWMA_FILE)
    out = {}
    for date, group in returns.items():
        p = group["p_bear_month_end"].to_numpy(float)
        regimes = switching.regime_path(p, 3)
        mu = np.array([[row[f"{a}_conditional_on_{regime}_end"] for a in ASSETS]
                       for regime, (_, row) in zip(regimes, group.iterrows())])
        out[date] = aware.Forecast(mu, covs[date], p)
    return out


def load_static_source():
    returns = read_wide_returns(STATIC_RETURNS_FILE)
    covs = read_long_covariances(STATIC_COV_FILE)
    return {date: aware.Forecast(group[list(ASSETS)].to_numpy(float), covs[date], None)
            for date, group in returns.items()}


# ------------------------------------------------------------
# STEP 2b: DCC-GARCH RISK MODEL (shared by every source)
# ------------------------------------------------------------

def disable_broken_numba():
    """Let `arch` run without numba when numba is installed but broken.

    `arch` speeds up GARCH with numba if it can import it. A numba built for an
    older numpy (common in Anaconda after upgrading numpy) crashes on import
    with a SystemError instead of an ImportError, so `arch` fails to load.
    `arch` supports ARCH_DISABLE_NUMBA=1, which uses its pure-Python code
    instead: same results, slower.
    """
    if "arch" in sys.modules or os.environ.get("ARCH_DISABLE_NUMBA"):
        return
    try:
        import numba  # noqa: F401
    except Exception:
        os.environ["ARCH_DISABLE_NUMBA"] = "1"
        print("  numba could not be loaded; running arch without it (same results, slower).",
              flush=True)


def build_dcc_covariances(train, validation, dates):
    """Stage covariances of DCC-GARCH month-length scenarios at each origin.

    Uses the team's scenario_generation.py exactly as run_cvar_mpo.py does:
    history = training + validation returns before the origin (the generator
    keeps the prior five years), frozen 2022-2024 NYSE calendar, origin index =
    chronological validation month, N_SCENARIOS scenarios, SEED.
    """
    if str(SCENARIO_DIR) not in sys.path:
        sys.path.insert(0, str(SCENARIO_DIR))
    disable_broken_numba()
    try:
        import scenario_generation as sg
        import run_cvar_mpo
    except ImportError as exc:
        raise ImportError("Building DCC-GARCH covariances needs the 'arch' package "
                          "(python -m pip install arch) and the Distribution Modelling folder.") from exc
    calendar = run_cvar_mpo.known_calendar()
    rows = []
    for index, date in enumerate(dates):
        origin = pd.Timestamp(date)
        history = pd.concat([train, validation[validation.index < origin]])
        periods = sg.monthly_periods(origin, calendar, 3)
        forecast = sg.generate_scenarios(history, origin, periods, index,
                                         n_scenarios=N_SCENARIOS, seed=SEED)
        cube = forecast.returns["DCC GARCH"]                  # (scenarios, months, assets)
        for k in range(3):
            matrix = np.cov(cube[:, k, :], rowvar=False)
            for i, asset in enumerate(ASSETS):
                rows.append({"rebalance_date": date, "month_ahead": k + 1,
                             "forecast_month": str(origin.to_period("M") + k), "asset": asset,
                             **dict(zip(ASSETS, matrix[i]))})
        print(f"  DCC-GARCH scenarios {index + 1:02d}/{len(dates)} {date}", flush=True)
    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DCC_CACHE, index=False)
    return frame


def dcc_covariances(train, validation, dates):
    """Cached DCC-GARCH stage covariances; rebuilt if missing or for other dates."""
    if DCC_CACHE.exists():
        cached = pd.read_csv(DCC_CACHE)
        if sorted(cached["rebalance_date"].unique()) == list(dates):
            return read_long_covariances(DCC_CACHE)
    print("Building DCC-GARCH stage covariances (about 1 second per month)...", flush=True)
    build_dcc_covariances(train, validation, dates)
    return read_long_covariances(DCC_CACHE)


def with_covariances(forecasts, covariances):
    return {date: aware.Forecast(f.mu, covariances[date], f.p_bear, f.days) for date, f in forecasts.items()}


def load_trading_days():
    """Trading days of forecast months 1-3 at each rebalance date."""
    frame = pd.read_csv(TRADING_DAYS_FILE)
    out = {}
    for date, group in frame.groupby("rebalance_date"):
        group = group.sort_values("month_ahead")
        if list(group["month_ahead"]) != [1, 2, 3]:
            raise ValueError(f"{TRADING_DAYS_FILE.name}: {date} lacks months 1-3")
        out[date] = group["trading_days"].to_numpy(int)
    return out


def with_trading_days(forecasts, days):
    return {date: aware.Forecast(f.mu, f.cov, f.p_bear, days[date]) for date, f in forecasts.items()}


# ------------------------------------------------------------
# STEP 3: HOLD THE WEIGHTS THROUGH THE REAL MONTH
# ------------------------------------------------------------

def self_financing_fee(target, prior, cost_rate=COST_RATE):
    """Same as cvar_mpo.self_financing_fee."""
    if cost_rate == 0:
        return 0.0
    return float(brentq(lambda f: f - cost_rate * np.abs((1 - f) * target - prior).sum(), 0.0, 0.999))


def execute_month(target, prior, daily_returns, cost_rate=COST_RATE):
    """Same as cvar_mpo.execute_first_period: hold with drift, fee on day one."""
    fee = self_financing_fee(target, prior, cost_rate)
    holdings, gross = target.astype(float).copy(), []
    for row in daily_returns:
        r = float(row @ holdings)
        gross.append(r)
        holdings = holdings * (1 + row) / (1 + r)
    net = np.asarray(gross)
    net[0] = (1 - fee) * (1 + net[0]) - 1
    return net, holdings, fee


# ------------------------------------------------------------
# STEP 4: WALK FORWARD
# ------------------------------------------------------------

def run_strategy(forecasts, months, decide, horizon, caps, label):
    """Each month: ask the model for weights, then live through the real month."""
    w_prev, parts, rows = np.zeros(3), [], []
    for date in sorted(forecasts):
        target, info = decide(w_prev, forecasts[date], horizon, caps)
        net, ending, fee = execute_month(target, w_prev, months[date].to_numpy())
        p = forecasts[date].p_bear
        regimes = [] if p is None else switching.regime_path(p, horizon)
        parts.append(pd.Series(net, index=months[date].index))
        rows.append({**label, "rebalance_date": date, **dict(zip(ASSETS, target)),
                     "executed_regime": regimes[0] if regimes else "",
                     "stage_regimes": "-".join(regimes), "stage_policies": "-".join(info["stages"]),
                     "return_floor_month1": np.nan if info["floors"][0] is None else info["floors"][0],
                     "p_bear_month1_end": np.nan if p is None else p[0],
                     "one_way_turnover": 0.5 * np.abs(target - w_prev).sum() if w_prev.any() else 0.0,
                     "fee": fee, "net_month_return": float(np.prod(1 + net) - 1)})
        w_prev = ending
    return pd.concat(parts), rows


def equal_weight(months, label):
    w_prev, parts, rows = np.zeros(3), [], []
    target = np.full(3, 1 / 3)
    for date in sorted(months):
        net, ending, fee = execute_month(target, w_prev, months[date].to_numpy())
        parts.append(pd.Series(net, index=months[date].index))
        rows.append({**label, "rebalance_date": date, **dict(zip(ASSETS, target)),
                     "one_way_turnover": 0.5 * np.abs(target - w_prev).sum() if w_prev.any() else 0.0,
                     "fee": fee, "net_month_return": float(np.prod(1 + net) - 1)})
        w_prev = ending
    return pd.concat(parts), rows


# ------------------------------------------------------------
# STEP 5: SCORE AND SAVE
# ------------------------------------------------------------

def annualized_sharpe(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)


def performance(r, weights, labels):
    """Same definitions as the CVaR run: cumulative net return, Sharpe from net
    daily returns (sample std, zero risk-free), max drawdown on daily wealth."""
    values = r.to_numpy()
    wealth = np.cumprod(1 + values)
    month = r.index.to_period("M").astype(str)
    out = {"Net total return": wealth[-1] - 1, "Ann. return": values.mean() * TRADING_DAYS,
           "Ann. volatility": values.std(ddof=1) * np.sqrt(TRADING_DAYS),
           "Sharpe": annualized_sharpe(values),
           "Max drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
           "Avg monthly turnover": weights["one_way_turnover"].mean(),
           "Total costs paid": weights["fee"].sum(),
           **{f"Avg {asset}": weights[asset].mean() for asset in ASSETS}}
    for regime in ("bull", "bear"):
        mask = np.asarray([labels.get(m) == regime for m in month])
        out[f"{regime.capitalize()} Sharpe (common HMM label)"] = annualized_sharpe(values[mask])
        out[f"{regime.capitalize()} months"] = len(set(month[mask]))
    return out


def common_hmm_labels():
    """Common HMM forecast month-1-end labels by month, used for every model."""
    probs = pd.read_csv(HMM_DIR / "regime_probabilities.csv")
    return {str(pd.Timestamp(d).to_period("M")): switching.regime_path([p], 1)[0]
            for d, p in zip(probs["rebalance_date"], probs["p_bear_month1_end"])}


def main(selected=("aware", "switching"), horizons=HORIZONS):
    train, validation = find_returns("train"), find_returns("validation")
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Training must precede validation")
    sources = {"HMM": load_file_source(HMM_DIR), "Jump Model": load_file_source(JUMP_DIR),
               "Regime regression": load_regression_source()}
    if INCLUDE_STATIC:
        sources["Static"] = load_static_source()
    dates = sorted(sources["HMM"])
    for name, forecasts in sources.items():
        if sorted(forecasts) != dates:
            raise ValueError(f"{name} forecasts cover different rebalance dates than HMM")
    if RISK_MODEL == "DCC GARCH":
        covariances = dcc_covariances(train, validation, dates)
        sources = {name: with_covariances(f, covariances) for name, f in sources.items()}
    elif RISK_MODEL != "native":
        raise ValueError("RISK_MODEL must be 'DCC GARCH' or 'native'")
    days = load_trading_days()
    if sorted(days) != dates:
        raise ValueError(f"{TRADING_DAYS_FILE.name} covers different rebalance dates")
    sources = {name: with_trading_days(f, days) for name, f in sources.items()}
    months = month_slices(validation, dates)
    for i, date in enumerate(dates):        # realized month must match the forecast calendar
        if len(months[date]) != days[date][0] and not (i == len(dates) - 1 and len(months[date]) < days[date][0]):
            raise ValueError(f"{date}: realized month has {len(months[date])} days, calendar {days[date][0]}")
    labels = common_hmm_labels()

    strategies = [entry for key in selected for entry in MODELS[key]]
    results, weight_rows, daily = [], [], {}
    for cap, source, (model_name, policy, decide), horizon in itertools.product(
            BTC_CAPS, sources, strategies, horizons):
        if source == "Static" and model_name.startswith("Regime-switching"):
            continue                         # no regime forecasts in the control
        caps = np.array([1.0, 1.0, 1.0 if cap is None else cap])
        cap_label = "none" if cap is None else f"{cap:.0%}"
        label = {"Model": model_name, "Forecasts": source, "Policy": policy, "H": horizon,
                 "BTC cap": cap_label, "Risk model": RISK_MODEL}
        r, rows = run_strategy(sources[source], months, decide, horizon, caps, label)
        weights = pd.DataFrame(rows)
        result = {**label, **performance(r, weights, labels)}
        if (weights["executed_regime"] != "").any():
            result["Executed bull months"] = int((weights["executed_regime"] == "bull").sum())
            result["Executed bear months"] = int((weights["executed_regime"] == "bear").sum())
        results.append(result)
        weight_rows.extend(rows)
        daily[f"{model_name} | {source} | {policy} | H={horizon} | cap {cap_label}"] = r

    label = {"Model": "Baseline", "Forecasts": "-", "Policy": "Equal weight (1/N monthly)", "H": 1,
             "BTC cap": "none", "Risk model": "-"}
    r, rows = equal_weight(months, label)
    results.append({**label, **performance(r, pd.DataFrame(rows), labels)})
    weight_rows.extend(rows)
    daily["Equal weight (1/N monthly)"] = r

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(OUTPUT_DIR / "regime_mvo_results.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUTPUT_DIR / "regime_mvo_weights.csv", index=False)
    pd.DataFrame(daily).to_csv(OUTPUT_DIR / "regime_mvo_daily_returns.csv")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print(f"\nValidation: {dates[0]} to {validation.index.max().date()} ({len(dates)} rebalances) | "
          f"risk model {RISK_MODEL} | cost {COST_RATE * 1e4:g} bps | gamma {aware.RISK_AVERSION} | "
          f"regime: month-end P(bear) per stage, bull < 0.5 < bear")
    show = table.copy()
    pct = ["Net total return", "Max drawdown", "Avg monthly turnover", "Avg BTC", "Avg Equity", "Avg Bond"]
    for column in pct:
        show[column] = (show[column] * 100).round(2)
    cols = ["Model", "Forecasts", "Policy", "H", "Net total return", "Sharpe", "Max drawdown",
            "Avg BTC", "Avg Equity", "Avg Bond", "Avg monthly turnover",
            "Bull Sharpe (common HMM label)", "Bear Sharpe (common HMM label)"]
    print("\n(% for returns, drawdown, weights and turnover)")
    print(show[cols].round(3).to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR.relative_to(ROOT)}/: regime_mvo_results.csv, "
          "regime_mvo_weights.csv, regime_mvo_daily_returns.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the regime MVO MPO models.")
    parser.add_argument("--model", choices=("aware", "switching", "both"), default="both")
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS),
                        help="planning horizons in months (default: 1 2 3)")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(("aware", "switching") if args.model == "both" else (args.model,),
             tuple(sorted(set(args.horizons))))