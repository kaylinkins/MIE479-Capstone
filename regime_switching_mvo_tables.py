"""Regime-switching MVO results tables, same format and metrics as the regime-aware tables.

Run from anywhere inside the repo (needs regime_aware_mvo_tables.py in the same folder):
    python Regime_MVO_MPO/regime_switching_mvo_tables.py

Model: regime-switching MVO (regime_switching_mvo.py), H = 3 only, HMM and tuned
jump-model forecasts. Each planned month uses Max Sharpe MVO if that forecast
model's own month-end P(bear) < 0.5 and Min Var MVO if > 0.5, as in the
switching CVaR runs. Two tables:
  1. Native scenarios: covariance = sample covariance of each model's own
     10,000 saved scenarios.
  2. DCC-GARCH scenarios: covariance = sample covariance of the team's 10,000
     DCC-GARCH scenarios.

Metrics, rf = 4% for reported Sharpe, and the bull/bear rows (common HMM labels,
the same for every row) are exactly those of regime_aware_mvo_tables.py.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_aware_mvo_tables as tables     # noqa: E402  (metrics, covariances, formatting)
import regime_mvo_backtest as bt             # noqa: E402
import regime_switching_mvo as switching     # noqa: E402

OUTPUT_CSV = bt.OUTPUT_DIR / "regime_switching_mvo_tables.csv"
OUTPUT_MD = bt.OUTPUT_DIR / "regime_switching_mvo_tables.md"


def run_specification(forecasts, months_by_date, labels):
    caps = np.array([1.0, 1.0, tables.BTC_CAP])
    w_prev, months = np.zeros(3), []
    for i, date in enumerate(sorted(forecasts)):
        target, info = switching.solve_plan(w_prev, forecasts[date], tables.HORIZON, caps,
                                            cost_rate=bt.COST_RATE)
        net, w_prev, fee = bt.execute_month(target, w_prev, months_by_date[date].to_numpy())
        months.append({"i": i, "date": date, "net": net, "fee": fee,
                       "regime": labels[str(pd.Timestamp(date).to_period("M"))],
                       "stage_policies": "-".join(info["stages"])})
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
        for name, folder in tables.FORECASTS.items():
            forecasts = bt.load_file_source(folder)
            if sorted(forecasts) != dates:
                raise ValueError(f"{name} forecasts cover different dates than the trading-day file")
            covariances = tables.native_scenario_covariances(folder, dates) if scenarios == "Native" else dcc
            forecasts = bt.with_trading_days(bt.with_covariances(forecasts, covariances), days)
            months = run_specification(forecasts, months_by_date, labels)
            for regime_label, regime in (("Overall", None), ("Bull", "bull"), ("Bear", "bear")):
                rows.append({"Forecasts": name,
                             "Scenarios": name if scenarios == "Native" else "DCC GARCH",
                             "Regime": regime_label, **tables.regime_metrics(months, regime)})
            n_ms = sum(m["stage_policies"].startswith("MS") for m in months)
            print(f"done: {name} forecasts, {scenarios} scenarios "
                  f"(executed {n_ms} Max Sharpe / {len(months) - n_ms} Min Var months)", flush=True)

    table = pd.DataFrame(rows)
    bt.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)

    def markdown(frame, columns):
        fmt = {c: "{:.3f}" if c == "Annual Sharpe" or "costs" in c else "{:.2f}" for c in tables.COLUMNS}
        lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(fmt[c].format(row[c]) if c in fmt else str(row[c])
                                           for c in columns) + " |")
        return "\n".join(lines)

    native = table[table["Scenarios"] != "DCC GARCH"]
    dcc_rows = table[table["Scenarios"] == "DCC GARCH"]
    text = ("Regime-switching MVO (bull: Max Sharpe MVO, bear: Min Var MVO), H=3, validation "
            "Feb 2022 - May 2024. Sharpe uses rf = 4%. Bull/bear rows = common HMM labels.\n\n"
            "Table 1: native scenarios\n\n"
            + markdown(native, ["Forecasts", "Scenarios", "Regime", *tables.COLUMNS])
            + "\n\nTable 2: DCC-GARCH scenarios\n\n"
            + markdown(dcc_rows, ["Forecasts", "Regime", *tables.COLUMNS]) + "\n")
    OUTPUT_MD.write_text(text)
    print("\n" + text)
    print(f"Saved {OUTPUT_CSV.relative_to(bt.ROOT)} and {OUTPUT_MD.relative_to(bt.ROOT)}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()