"""Regime-switching MVO results tables, same format and metrics as the regime-aware tables.

Run from anywhere inside the repo (needs regime_aware_mvo_tables.py in the same folder):
    python Regime_MVO_MPO/regime_switching_mvo_tables.py

Model: regime-switching MVO (regime_switching_mvo.py), H = 3 only, HMM and tuned
jump-model forecasts. Each planned month uses Max Sharpe MVO if that forecast
model's own month-end P(bear) < 0.5 and Min Var MVO if > 0.5, as in the
switching CVaR runs. One table per covariance source, as set by
COVARIANCE_SOURCES in regime_aware_mvo_tables.py (default: EWMA lambda = 0.93 and
DCC-GARCH; add "Native scenarios" for the table matching the native CVaR run).

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
    tables.build_tables(run_specification,
                        "Regime-switching MVO (bull: Max Sharpe MVO, bear: Min Var MVO),",
                        OUTPUT_CSV, OUTPUT_MD)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
