"""Backtest: walk-forward evaluation of the regime MVO decision models.

Run from the repository root:
    python "MVO Regime Models/regime_mvo_backtest.py"
    python "MVO Regime Models/regime_mvo_backtest.py" --model aware
    python "MVO Regime Models/regime_mvo_backtest.py" --model switching
    python "MVO Regime Models/regime_mvo_backtest.py" --horizons 1 3

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
  * Risk model: the exact DCC-GARCH scenarios from the CVaR forecast cache.
    The MVO covariance for each stage is the sample covariance of those draws.
  * Bull stages: min risk + cost with the max-Sharpe return floor; bear
    stages: min risk + cost with no floor; the full plan is solved jointly and
    only stage one is executed.
  * Return floors are read from the paired CVaR decision artifacts. They use
    the stage mean times max-Sharpe weights minus 1e-6 per holding day.
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
DCC-GARCH requires the completed CVaR forecast, fixed-policy all-DCC and
switching all-DCC runs. Override their directories using the CLI if needed.

Outputs (Regime_MVO_MPO/outputs/)
    regime_mvo_results.csv           one row per model x source x H x BTC cap
    regime_mvo_weights.csv           executed weights, stage regimes and rules
    regime_mvo_daily_returns.csv     net daily returns of every strategy

Limits: 28 validation months, already inspected repeatedly; treat results as
validation, not a fresh holdout.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import itertools
import json
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
    return HERE.parent


def first_existing(*paths):
    """First path that exists, so moved output files are still found."""
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


ROOT = find_repo_root()
OUTPUT_DIR = ROOT / "Regime_MVO_MPO" / "outputs"
ASSETS = ("Equity", "Bond", "BTC")
CVAR_DIR = ROOT / "CVaR Regime Models"
FORECAST_RESULTS = CVAR_DIR / "forecast_comparison_results"
CVAR_AWARE_RESULTS = CVAR_DIR / "all_dcc_comparison_results"
CVAR_SWITCHING_RESULTS = CVAR_DIR / "regime_switching_all_dcc_results"

HMM_DIR = ROOT / "HMM_regime_forecasts" / "hmm_outputs"
JUMP_DIR = ROOT / "HMM_regime_forecasts" / "jump_tuned_outputs"   # training-tuned jump model (as in the CVaR run); original: jump_outputs
REGRESSION_FILES = (
    ROOT / "Return_Forecasting" / "outputs" / "hmm_regime_conditional_returns.csv",
    ROOT / "Return_Forecasting" / "outputs" / "misc" / "hmm_regime_conditional_returns.csv")
REGIME_EWMA_FILE = (ROOT / "Covariance_Estimation" / "outputs"
                    / "covariances_regime_ewma_training_selected_walk_forward_validation.csv")
STATIC_RETURNS_FILE = ROOT / "Return_Forecasting" / "outputs" / "walk_forward_validation_expected_returns_h1_h3.csv"
STATIC_COV_FILE = ROOT / "Covariance_Estimation" / "outputs" / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"
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
    returns = read_wide_returns(first_existing(*REGRESSION_FILES))
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

def completed_manifest(folder):
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest["completed"]:
        raise ValueError(f"CVaR run is incomplete: {folder}")
    return manifest


def checked_artifact(folder, manifest, relative):
    path = folder / relative
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != manifest["output_sha256"].get(relative):
        raise ValueError(f"CVaR artifact differs from its run manifest: {path}")
    return path


def load_paired_forecasts(forecast_results, aware_results, switching_results):
    """Load the same DCC draws, forecast means, stage signals and floors as CVaR."""
    forecast_run, aware_run, switching_run = map(
        completed_manifest, (forecast_results, aware_results, switching_results))
    for run in (aware_run, switching_run):
        protocol = run["protocol"]
        if protocol["source_forecast_fingerprint"] != forecast_run["fingerprint"]:
            raise ValueError("Paired CVaR runs must use the same forecast cache")
        if protocol["upper"] != [1.0, 1.0, 0.2] or protocol["horizons"] != [1, 3]:
            raise ValueError("Paired CVaR runs must use the 20% BTC cap and horizons 1 and 3")
    if BTC_CAPS != (0.20,):
        raise ValueError("Paired MVO and CVaR require the same 20% BTC cap")

    experiments = {"HMM": "hmm", "Jump Model": "jump", "Regime regression": "regression_dcc"}
    if INCLUDE_STATIC:
        experiments["Static"] = "static_dcc"
    signals = pd.read_csv(
        checked_artifact(switching_results, switching_run, "stage_selections.csv"),
        float_precision="round_trip")
    sources = {name: {} for name in experiments}
    decisions = sorted(relative for relative in aware_run["output_sha256"]
                       if relative.startswith("decisions/") and relative.endswith(".json"))
    switching_decisions = sorted(relative for relative in switching_run["output_sha256"]
                                 if relative.startswith("decisions/") and relative.endswith(".json"))
    if not decisions or decisions != switching_decisions:
        raise ValueError("Paired CVaR runs must cover the same rebalances")
    for relative in decisions:
        date = Path(relative).stem
        fixed, mixed = (
            json.loads(checked_artifact(folder, run, relative).read_text(encoding="utf-8"))
            for folder, run in ((aware_results, aware_run), (switching_results, switching_run)))
        cache_name = f"forecasts/{date}.npz"
        path = checked_artifact(forecast_results, forecast_run, cache_name)
        for decision in (fixed, mixed):
            if (decision["origin"] != date or decision["scenario_key"] != "DCC GARCH"
                    or decision["source_forecast_sha256"] != forecast_run["output_sha256"][cache_name]):
                raise ValueError(f"{date}: paired CVaR decisions must use the same DCC-GARCH draws")
        with np.load(path, allow_pickle=False) as saved:
            cube = np.asarray(saved["DCC GARCH"], float)
            days = saved["holding_days"]
            cov = np.asarray([np.cov(cube[:, k], rowvar=False, ddof=1) for k in range(len(days))])
            for name, experiment in experiments.items():
                means = saved[f"means_{experiment}"]
                matched = fixed["plans"][f"{experiment} | matched_return | H=3"]["metadata"]
                targets = matched["gross_return_targets"]
                assert_context = f"{date}, {name}"
                np.testing.assert_array_equal(matched["holding_days"], days, err_msg=assert_context)
                first = fixed["plans"][f"{experiment} | matched_return | H=1"]["metadata"]
                np.testing.assert_array_equal(first["gross_return_targets"], targets[:1], err_msg=assert_context)
                p_bear = None
                if name != "Static":
                    rows = signals[(signals.origin == date) & (signals.experiment == experiment)].sort_values("stage")
                    np.testing.assert_array_equal(rows.stage, [1, 2, 3], err_msg=assert_context)
                    np.testing.assert_array_equal(rows[list(ASSETS)], means, err_msg=assert_context)
                    p_bear = rows.p_bear_month_end.to_numpy(float)
                    regimes = switching.regime_path(p_bear, 3)
                    switch_targets = [target if regime == "bull" else None
                                      for target, regime in zip(targets, regimes)]
                    for horizon in (1, 3):
                        stage_plan = mixed["plans"][f"{experiment} | H={horizon}"]["metadata"]
                        np.testing.assert_array_equal(stage_plan["stage_regimes"], regimes[:horizon],
                                                      err_msg=assert_context)
                        np.testing.assert_array_equal(stage_plan["gross_return_targets"], switch_targets[:horizon],
                                                      err_msg=assert_context)
                sources[name][date] = aware.Forecast(means, cov, p_bear, days, np.asarray(targets, float))
    return sources


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
    return {date: aware.Forecast(f.mu, f.cov, f.p_bear, days[date], f.return_targets)
            for date, f in forecasts.items()}


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
                     **{f"return_floor_month{k}": floor for k, floor in enumerate(info["floors"], 1)},
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


def main(selected=("aware", "switching"), horizons=HORIZONS, *,
         forecast_results=FORECAST_RESULTS, aware_results=CVAR_AWARE_RESULTS,
         switching_results=CVAR_SWITCHING_RESULTS):
    train, validation = find_returns("train"), find_returns("validation")
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Training must precede validation")
    if RISK_MODEL == "DCC GARCH":
        sources = load_paired_forecasts(Path(forecast_results), Path(aware_results), Path(switching_results))
        days = {date: forecast.days for date, forecast in sources["HMM"].items()}
    elif RISK_MODEL == "native":
        sources = {"HMM": load_file_source(HMM_DIR), "Jump Model": load_file_source(JUMP_DIR),
                   "Regime regression": load_regression_source()}
        if INCLUDE_STATIC:
            sources["Static"] = load_static_source()
        days = load_trading_days()
        if sorted(days) != sorted(sources["HMM"]):
            raise ValueError(f"{TRADING_DAYS_FILE.name} covers different rebalance dates")
        sources = {name: with_trading_days(f, days) for name, f in sources.items()}
    else:
        raise ValueError("RISK_MODEL must be 'DCC GARCH' or 'native'")
    dates = sorted(sources["HMM"])
    for name, forecasts in sources.items():
        if sorted(forecasts) != dates:
            raise ValueError(f"{name} forecasts cover different rebalance dates than HMM")
    months = month_slices(validation, dates)
    for i, date in enumerate(dates):        # realized month must match the forecast calendar
        if len(months[date]) != days[date][0] and not (i == len(dates) - 1 and len(months[date]) < days[date][0]):
            raise ValueError(f"{date}: realized month has {len(months[date])} days, calendar {days[date][0]}")
    labels = {str(pd.Timestamp(date).to_period("M")): switching.regime_path(forecast.p_bear, 1)[0]
              for date, forecast in sources["HMM"].items()}

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
    print(f"\nValidation: {dates[0]} to {months[dates[-1]].index[-1].date()} ({len(dates)} rebalances) | "
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
    parser.add_argument("--forecast-results", type=Path, default=FORECAST_RESULTS,
                        help="completed CVaR forecast cache")
    parser.add_argument("--cvar-aware-results", type=Path, default=CVAR_AWARE_RESULTS,
                        help="completed fixed-policy all-DCC CVaR results")
    parser.add_argument("--cvar-switching-results", type=Path, default=CVAR_SWITCHING_RESULTS,
                        help="completed switching all-DCC CVaR results")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(("aware", "switching") if args.model == "both" else (args.model,),
             tuple(sorted(set(args.horizons))), forecast_results=args.forecast_results,
             aware_results=args.cvar_aware_results, switching_results=args.cvar_switching_results)
