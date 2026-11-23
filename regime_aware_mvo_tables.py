"""Regime-aware MVO results tables, in the same format as the regime-aware CVaR tables.

Run from anywhere inside the repo:
    python Regime_MVO_MPO/regime_aware_mvo_tables.py

Model: regime-aware Max Sharpe MVO (regime_aware_mvo.py), H = 3 only, HMM and
tuned jump-model forecasts. Two tables:
  1. Native scenarios: covariance = sample covariance (ddof=1) of each model's own
     10,000 saved scenarios (cvar_scenarios.npz), as in the CVaR max-Sharpe
     reference (regime_cvar.max_sharpe_reference).
  2. DCC-GARCH scenarios: covariance = sample covariance of the team's 10,000
     DCC-GARCH scenarios (same cache as regime_mvo_backtest.py).

Settings match the regime-aware CVaR run: matched max-Sharpe floor with zero
risk-free rate in the optimizer, 10 bps cost, caps (1, 1, 0.2), start in cash.

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
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_aware_mvo as aware          # noqa: E402
import regime_mvo_backtest as bt          # noqa: E402

# ---------------------------------------------------------------- settings
HORIZON = 3
POLICY = "Max Sharpe MVO"                 # the regime-aware model
RISK_FREE_ANNUAL = 0.04                   # for reported Sharpe only
BTC_CAP = 0.20
FORECASTS = {"HMM": bt.HMM_DIR, "Jump Model": bt.JUMP_DIR}   # JUMP_DIR = jump_tuned_outputs
TRADING_DAYS = 252
OUTPUT_CSV = bt.OUTPUT_DIR / "regime_aware_mvo_tables.csv"
OUTPUT_MD = bt.OUTPUT_DIR / "regime_aware_mvo_tables.md"

COLUMNS = ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe", "Daily 95% VaR (%)",
           "Monthly 95% VaR (%)", "Max drawdown (%)", "Annual transaction costs (%)"]


# ---------------------------------------------------------------- covariances
def native_scenario_covariances(folder, dates):
    """Per-stage sample covariance of the model's own saved scenarios."""
    with np.load(folder / "cvar_scenarios.npz") as cubes:
        missing = [d for d in dates if d not in cubes.files]
        if missing:
            raise ValueError(f"{folder.name}/cvar_scenarios.npz lacks {missing[:3]}")
        out = {}
        for d in dates:
            values = cubes[d].astype(float)            # (scenario, month, asset)
            if values.ndim != 3 or values.shape[1] < HORIZON or values.shape[2] != 3:
                raise ValueError(f"{folder.name}: scenarios for {d} have shape {values.shape}")
            out[d] = np.stack([np.cov(values[:, k], rowvar=False, ddof=1) for k in range(values.shape[1])])
    return out


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


def main():
    train, validation = bt.find_returns("train"), bt.find_returns("validation")
    days = bt.load_trading_days()
    dates = sorted(days)
    months_by_date = bt.month_slices(validation, dates)
    labels = bt.common_hmm_labels()
    dcc = bt.dcc_covariances(train, validation, dates)

    rows = []
    for scenarios in ("Native", "DCC GARCH"):
        for name, folder in FORECASTS.items():
            forecasts = bt.load_file_source(folder)
            if sorted(forecasts) != dates:
                raise ValueError(f"{name} forecasts cover different dates than the trading-day file")
            covariances = native_scenario_covariances(folder, dates) if scenarios == "Native" else dcc
            forecasts = bt.with_trading_days(bt.with_covariances(forecasts, covariances), days)
            months = run_specification(forecasts, months_by_date, labels)
            for regime_label, regime in (("Overall", None), ("Bull", "bull"), ("Bear", "bear")):
                rows.append({"Forecasts": name,
                             "Scenarios": name if scenarios == "Native" else "DCC GARCH",
                             "Regime": regime_label, **regime_metrics(months, regime)})
            print(f"done: {name} forecasts, {scenarios} scenarios", flush=True)

    table = pd.DataFrame(rows)
    bt.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)

    def markdown(frame, columns):
        fmt = {c: "{:.3f}" if c == "Annual Sharpe" or "costs" in c else "{:.2f}" for c in COLUMNS}
        lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(fmt[c].format(row[c]) if c in fmt else str(row[c])
                                           for c in columns) + " |")
        return "\n".join(lines)

    native = table[table["Scenarios"] != "DCC GARCH"]
    dcc_rows = table[table["Scenarios"] == "DCC GARCH"]
    text = ("Regime-aware Max Sharpe MVO, H=3, validation Feb 2022 - May 2024. "
            "Sharpe uses rf = 4%. Bull/bear = common HMM labels.\n\n"
            "Table 1: native scenarios\n\n" + markdown(native, ["Forecasts", "Scenarios", "Regime", *COLUMNS])
            + "\n\nTable 2: DCC-GARCH scenarios\n\n" + markdown(dcc_rows, ["Forecasts", "Regime", *COLUMNS]) + "\n")
    OUTPUT_MD.write_text(text)
    print("\n" + text)
    print(f"Saved {OUTPUT_CSV.relative_to(bt.ROOT)} and {OUTPUT_MD.relative_to(bt.ROOT)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()