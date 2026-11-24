"""Regime-aware MVO results tables, in the same format as the regime-aware CVaR tables.

Run from anywhere inside the repo:
    python Regime_MVO_MPO/regime_aware_mvo_tables.py

Model: regime-aware Max Sharpe MVO (regime_aware_mvo.py), H = 3 only, HMM and
tuned jump-model forecasts. One table per covariance source in COVARIANCE_SOURCES:
  "EWMA 0.93"         Covariance_Estimation/outputs/
                      covariances_ewma_lambda_0_93_walk_forward_validation.csv
                      (trailing 60 months, lambda chosen on training data; the same
                      matrix for months 1-3 of each rebalance)
  "DCC GARCH"         exactly the inputs the paired CVaR runs used, loaded by
                      regime_mvo_backtest.load_paired_forecasts: covariance of the
                      cached 10,000 DCC-GARCH draws, forecast means, month-end
                      P(bear) and the shared max-Sharpe return floors. Needs the
                      completed CVaR result folders (see regime_mvo_backtest.py);
                      if they are missing this table is skipped with a message.
  "Native scenarios"  exactly the inputs of the CVaR native-scenario run
                      (forecast_comparison_results): covariance of each regime model's
                      own cached 10,000 scenarios, its forecast means and month-end
                      P(bear), and that run's shared max-Sharpe return floors
The expected returns and regime probabilities always come from the HMM or jump model.

Return floors (bull / Max Sharpe stages): mu_k . w_k^MS - 1e-6 x trading days, the
definition used by the CVaR regime-aware and regime-switching runs, where w_k^MS is
the max-Sharpe portfolio over a risk-free rate of regime_aware_mvo.RISK_FREE_ANNUAL
(4% by default, compounded over the month's trading days). The CVaR runs saved their
floors with a 0% rate; those saved floors are used only if RISK_FREE_ANNUAL is set
back to 0, otherwise every table computes its floor with the same definition and
the 4% rate. No CVaR run uses EWMA, so the EWMA table always computes its own.

Other settings match the regime-aware CVaR run: 10 bps cost, caps (1, 1, 0.2),
start in cash.

Metrics (definitions reproduce the regime-aware CVaR tables exactly):
  Net CAGR            prod(1 + r) ** (252 / days) - 1, over the days in that regime
  Annual volatility   daily std (ddof=1) x sqrt(252)
  Annual Sharpe       (mean daily r - daily rf) x 252 / annual vol, rf = 4%/yr
                      compounded: daily rf = 1.04 ** (1/252) - 1
  Daily 95% VaR       95th percentile of daily losses (numpy interpolation)
  Monthly 95% VaR     95th percentile of monthly net losses
  Max drawdown        Overall: whole path. Bull/Bear: worst drawdown within any
                      single episode of consecutive months in that regime
  Annual costs        sum of fees paid in that regime / (days in regime / 252)
Regime = common HMM label (month-1-end P(bear) of the HMM forecast; bull if
< 0.5), the same labels for every row, as in the CVaR tables.
"""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_aware_mvo as aware          # noqa: E402
import regime_mvo_backtest as bt          # noqa: E402
import regime_switching_mvo as switching  # noqa: E402

# ---------------------------------------------------------------- settings
HORIZON = 3
POLICY = "Max Sharpe MVO"                 # the regime-aware model
RISK_FREE_ANNUAL = 0.04                   # for reported Sharpe only
BTC_CAP = 0.20
FORECASTS = {"HMM": bt.HMM_DIR, "Jump Model": bt.JUMP_DIR}   # JUMP_DIR = jump_tuned_outputs
COVARIANCE_SOURCES = ("EWMA 0.93", "DCC GARCH")   # add "Native scenarios" for the CVaR-matching native table
EWMA_FILE = bt.ROOT / "Covariance_Estimation" / "outputs" / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"
TRADING_DAYS = 252
OUTPUT_CSV = bt.OUTPUT_DIR / "regime_aware_mvo_tables.csv"
OUTPUT_MD = bt.OUTPUT_DIR / "regime_aware_mvo_tables.md"

COLUMNS = ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe", "Daily 95% VaR (%)",
           "Monthly 95% VaR (%)", "Max drawdown (%)", "Annual transaction costs (%)"]


# ---------------------------------------------------------------- covariances
def ewma_covariances(dates):
    """EWMA lambda = 0.93 monthly covariance forecasts for months 1-3 of each rebalance."""
    if not EWMA_FILE.exists():
        raise FileNotFoundError(f"{EWMA_FILE.relative_to(bt.ROOT)} not found")
    covs = bt.read_long_covariances(EWMA_FILE)
    missing = [d for d in dates if d not in covs]
    if missing:
        raise ValueError(f"{EWMA_FILE.name} lacks rebalance dates {missing[:3]}")
    for d in dates:
        if covs[d].shape[0] < HORIZON:
            raise ValueError(f"{EWMA_FILE.name}: {d} has fewer than {HORIZON} forecast months")
    return covs


def with_covariances(forecasts, covariances, days):
    """Keep a forecast's means and P(bear); replace its covariance; attach trading days."""
    return {date: aware.Forecast(f.mu, covariances[date], f.p_bear, days[date])
            for date, f in forecasts.items()}


def common_hmm_labels():
    """Month -> 'bull'/'bear' from the HMM's month-1-end P(bear): the same labels for every row."""
    frame = pd.read_csv(bt.HMM_DIR / "regime_probabilities.csv")
    frame = frame[frame["origin_type"] == "validation"] if "origin_type" in frame else frame
    return {str(pd.Timestamp(d).to_period("M")): switching.regime_path([p], 1)[0]
            for d, p in zip(frame["rebalance_date"], frame["p_bear_month1_end"])}


NATIVE_EXPERIMENTS = {"HMM": ("hmm", "HMM"), "Jump Model": ("jump", "Jump Model")}


def load_native_paired(results=None):
    """Inputs of the CVaR native-scenario run: own scenarios, means, P(bear) and floors."""
    results = Path(results or bt.FORECAST_RESULTS)
    run = bt.completed_manifest(results)
    if run["protocol"]["upper"] != [1.0, 1.0, BTC_CAP] or run["protocol"]["risk_free"] != 0.0:
        raise ValueError("The CVaR native run must use the 20% BTC cap and a zero risk-free floor")
    signals = pd.read_csv(bt.checked_artifact(results, run, "regime_inputs.csv"), float_precision="round_trip")
    decisions = sorted(r for r in run["output_sha256"] if r.startswith("decisions/") and r.endswith(".json"))
    out = {name: {} for name in NATIVE_EXPERIMENTS}
    for relative in decisions:
        date = Path(relative).stem
        decision = json.loads(bt.checked_artifact(results, run, relative).read_text(encoding="utf-8"))
        cache_name = f"forecasts/{date}.npz"
        with np.load(bt.checked_artifact(results, run, cache_name), allow_pickle=False) as saved:
            days = np.asarray(saved["holding_days"], int)
            for name, (experiment, scenarios) in NATIVE_EXPERIMENTS.items():
                cube = np.asarray(saved[scenarios], float)
                cov = np.stack([np.cov(cube[:, k], rowvar=False, ddof=1) for k in range(len(days))])
                meta = decision["plans"][f"{experiment} | matched_return | H=3"]["metadata"]
                if meta["scenario_source"] != scenarios or list(meta["holding_days"]) != days.tolist():
                    raise ValueError(f"{date}, {name}: CVaR plan does not match the cached scenarios")
                row = signals[(signals["source"] == name) & (signals["rebalance_date"] == date)]
                if len(row) != 1:
                    raise ValueError(f"{date}, {name}: missing regime probabilities")
                p_bear = row[[f"p_bear_month{k}_end" for k in (1, 2, 3)]].to_numpy(float)[0]
                out[name][date] = aware.Forecast(np.asarray(saved[f"means_{experiment}"], float), cov,
                                                 p_bear, days, np.asarray(meta["gross_return_targets"], float))
    return out


def forecasts_for(source, name, folder, dates, days, cache):
    """Full forecast package (means, covariance, P(bear), days, floors) for one table."""
    if source == "DCC GARCH":
        if "paired" not in cache:
            cache["paired"] = bt.load_paired_forecasts(
                bt.FORECAST_RESULTS, bt.CVAR_AWARE_RESULTS, bt.CVAR_SWITCHING_RESULTS)
        forecasts = cache["paired"][name]
    elif source == "Native scenarios":
        if "native" not in cache:
            cache["native"] = load_native_paired()
        forecasts = cache["native"][name]
    else:
        if source == "EWMA 0.93":
            covariances = cache.setdefault("ewma", ewma_covariances(dates))
        else:
            raise ValueError(f"Unknown covariance source: {source}")
        forecasts = with_covariances(bt.load_file_source(folder), covariances, days)
    if sorted(forecasts) != dates:
        raise ValueError(f"{name} ({source}) covers different rebalance dates")
    return forecasts


# ---------------------------------------------------------------- metrics
def max_drawdown(daily):
    wealth = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]      # peak includes the starting value
    return float(np.min(wealth / peak - 1))


def regime_metrics(months, regime=None):
    """months: list of dicts (in time order) with 'i', 'net' (daily array), 'fee', 'regime'."""
    chosen = [m for m in months if regime is None or m["regime"] == regime]
    if not chosen:
        return {c: np.nan for c in COLUMNS}
    r = np.concatenate([m["net"] for m in chosen])
    days = len(r)
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    daily_rf = (1 + RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1
    monthly = np.array([np.prod(1 + m["net"]) - 1 for m in chosen])
    # episodes: whole path for Overall, runs of consecutive months otherwise
    episodes = []
    for m in chosen:
        if episodes and (regime is None or episodes[-1][-1]["i"] == m["i"] - 1):
            episodes[-1].append(m)
        else:
            episodes.append([m])
    drawdown = min(max_drawdown(np.concatenate([m["net"] for m in e])) for e in episodes)
    return {
        "Net CAGR (%)": (np.prod(1 + r) ** (TRADING_DAYS / days) - 1) * 100,
        "Annual volatility (%)": vol * 100,
        "Annual Sharpe": (r.mean() - daily_rf) * TRADING_DAYS / vol,
        "Daily 95% VaR (%)": np.quantile(-r, 0.95) * 100,
        "Monthly 95% VaR (%)": np.quantile(-monthly, 0.95) * 100,
        "Max drawdown (%)": drawdown * 100,
        "Annual transaction costs (%)": sum(m["fee"] for m in chosen) / (days / TRADING_DAYS) * 100,
    }


# ---------------------------------------------------------------- run
def run_specification(forecasts, months_by_date, labels):
    caps = np.array([1.0, 1.0, BTC_CAP])
    decide = lambda w, f, h, c: aware.solve_plan(w, f, h, POLICY, c, cost_rate=bt.COST_RATE)
    w_prev, months = np.zeros(3), []
    for i, date in enumerate(sorted(forecasts)):
        target, _ = decide(w_prev, forecasts[date], HORIZON, caps)
        net, w_prev, fee = bt.execute_month(target, w_prev, months_by_date[date].to_numpy())
        months.append({"i": i, "date": date, "net": net, "fee": fee,
                       "regime": labels[str(pd.Timestamp(date).to_period("M"))]})
    return months


def markdown(frame):
    columns = ["Forecasts", "Regime", *COLUMNS]
    fmt = {c: "{:.3f}" if c == "Annual Sharpe" or "costs" in c else "{:.2f}" for c in COLUMNS}
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt[c].format(row[c]) if c in fmt else str(row[c])
                                       for c in columns) + " |")
    return "\n".join(lines)


def build_tables(run, heading, csv_path, md_path):
    """Run one model for every forecast x covariance source and write the tables."""
    validation = bt.find_returns("validation")
    days = bt.load_trading_days()
    dates = sorted(days)
    months_by_date = bt.month_slices(validation, dates)
    labels = common_hmm_labels()
    cache, rows, skipped = {}, [], []
    for source in COVARIANCE_SOURCES:
        for name, folder in FORECASTS.items():
            try:
                forecasts = forecasts_for(source, name, folder, dates, days, cache)
            except FileNotFoundError as missing:
                print(f"SKIPPED {source} table: {missing}", flush=True)
                skipped.append(source)
                break
            months = run(forecasts, months_by_date, labels)
            for regime_label, regime in (("Overall", None), ("Bull", "bull"), ("Bear", "bear")):
                rows.append({"Forecasts": name, "Covariance": source, "Regime": regime_label,
                             **regime_metrics(months, regime)})
            print(f"done: {name} forecasts, {source} covariance", flush=True)

    table = pd.DataFrame(rows)
    bt.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    titles = {"EWMA 0.93": "EWMA covariance (lambda = 0.93)",
              "DCC GARCH": "DCC-GARCH scenario covariance (paired with the CVaR runs)",
              "Native scenarios": "Native regime-model scenarios (paired with the CVaR native run)"}
    parts = [heading + " H=3, validation Feb 2022 - May 2024. Sharpe uses rf = 4%. "
             f"Max-Sharpe return target uses rf = {aware.RISK_FREE_ANNUAL:.0%}. "
             "Bull/bear rows = common HMM labels.\n"]
    for i, source in enumerate(COVARIANCE_SOURCES, 1):
        if source in skipped:
            parts.append(f"Table {i}: {titles[source]}\n\nNOT RUN: input files missing (see console message).\n")
            continue
        parts.append(f"Table {i}: {titles[source]}\n\n" + markdown(table[table["Covariance"] == source]) + "\n")
    text = "\n".join(parts)
    md_path.write_text(text)
    print("\n" + text)
    print(f"Saved {csv_path.relative_to(bt.ROOT)} and {md_path.relative_to(bt.ROOT)}")


def main():
    build_tables(run_specification, "Regime-aware Max Sharpe MVO,", OUTPUT_CSV, OUTPUT_MD)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
