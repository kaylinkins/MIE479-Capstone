"""Regime-independent REGRESSION expected returns, walk-forward on validation.

Imports Return_Forecasting/return_forecasting.py unchanged. Reads train and
validation only (never portfolio_test.csv). Writes a NEW file; overwrites nothing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Return_Forecasting"))
import return_forecasting as rf          # main() is guarded; import only

# Fixed in advance, not tuned on validation.
REG_METHOD = "reg_ar1_recursive_expanding"
METHODS = {a: REG_METHOD for a in rf.ASSETS}

HIST_FILE = ROOT / "Return_Forecasting/outputs/walk_forward_validation_expected_returns_h1_h3.csv"
OUT_DIR = ROOT / "Return_Forecasting/outputs"
OUT_FILE = OUT_DIR / "walk_forward_validation_expected_returns_h1_h3_regression.csv"


def main():
    data_dir = rf.find_split_dir()
    train = rf.read_return_split(data_dir, "train")
    validation = rf.read_return_split(data_dir, "validation")
    daily = pd.concat([train, validation]).sort_index()

    long_prices = rf.load_long_equity_bond_prices(validation.index.max())
    monthly_history = rf.build_asset_monthly_history(daily, long_prices)
    calendar = rf.build_calendar(daily, ())

    selected = {a: {"method": METHODS[a]} for a in rf.ASSETS}
    base = {a: "expanding_mean" for a in rf.ASSETS}      # unused (no shrinkage)

    long_df = rf.build_walk_forward_validation_forecasts(
        validation, daily, monthly_history, selected, base, calendar)
    wide = rf.selected_forecasts_to_wide(long_df, include_realized=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_FILE, index=False)

    # ---- checks (no realized returns used) ----
    print(f"Wrote {OUT_FILE}")
    print(f"rows={len(wide)} (expect 84), origins={wide['rebalance_date'].nunique()} "
          f"(expect 28), NaNs={int(wide[list(rf.ASSETS)].isna().sum().sum())}")
    print("methods:", {a: wide[f'method_{a}'].unique().tolist() for a in rf.ASSETS})

    hist = pd.read_csv(HIST_FILE)
    key = ["rebalance_date", "month_ahead"]
    m = wide.merge(hist[key + list(rf.ASSETS)], on=key, suffixes=("_reg", "_hist"))
    print("\nMonthly forecasts (%), regression vs historical-mean file")
    print(f"{'asset':7s}{'h':>3s}{'reg min':>9s}{'reg mean':>9s}{'reg max':>9s}"
          f"{'hist mean':>10s}{'corr':>7s}")
    for a in rf.ASSETS:
        for h in (1, 2, 3):
            g = m[m["month_ahead"] == h]
            r, s = g[f"{a}_reg"] * 100, g[f"{a}_hist"] * 100
            print(f"{a:7s}{h:3d}{r.min():9.2f}{r.mean():9.2f}{r.max():9.2f}"
                  f"{s.mean():10.2f}{np.corrcoef(r, s)[0, 1]:7.2f}")

    print("\nFirst origin, horizons 1-3 (%):")
    first = wide[wide["rebalance_date"] == wide["rebalance_date"].min()]
    print((first[["month_ahead", *rf.ASSETS]]
           .assign(**{a: first[a] * 100 for a in rf.ASSETS}))
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()