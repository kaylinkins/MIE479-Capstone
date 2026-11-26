"""BTC regression candidates scored on TRAINING data only (option 1b).

Equity and Bond: lowest training combined RMSE (h1-h3) among eligible
regression methods, read from walk_forward_training_method_scores.csv.

BTC rule (fixed in advance):
  * candidates: reg_ar1_expanding, reg_ar1_recursive_expanding,
    reg_mom12_1_expanding (same list as Equity/Bond)
  * same 60-pair minimum (inside return_forecasting.forecast_one)
  * eligible if >= MIN_VALIDATION_MONTHS scored months at every horizon
  * scored on the training dates shared by all eligible candidates
    (the module's "same dates as expanding mean" rule would exclude them)
  * pick lowest combined RMSE h1-h3; expanding mean only if none eligible

Does not read portfolio_test.csv. Does not modify return_forecasting.py.
Writes only new files under
Return_Forecasting/outputs/regression_trainbest_btcreg/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent  # Return_Forecasting/
sys.path.insert(0, str(HERE))

import return_forecasting as rf  # noqa: E402

HORIZONS = (1, 2, 3)
BTC_CANDIDATES = list(rf.REGRESSION_METHODS)
SCORES_FILE = rf.OUTPUT_DIR / "misc" / "walk_forward_training_method_scores.csv"
PREV_FILE = (
    rf.OUTPUT_DIR / "walk_forward_validation_expected_returns_h1_h3_regression_trainbest.csv"
)
OUT_DIR = rf.OUTPUT_DIR / "regression_trainbest_btcreg"
OUT_FILE = OUT_DIR / (
    "walk_forward_validation_expected_returns_h1_h3_"
    "regression_trainbest_btcreg.csv"
)


def load_inputs():
    data_dir = rf.find_split_dir()
    train = rf.read_return_split(data_dir, "train")
    validation = rf.read_return_split(data_dir, "validation")
    if train.index.max() >= validation.index.min():
        raise ValueError("Training and validation dates overlap.")
    daily = pd.concat([train, validation]).sort_index()
    long_prices = rf.load_long_equity_bond_prices(validation.index.max())
    monthly_history = rf.build_asset_monthly_history(daily, long_prices)
    calendar = rf.build_calendar(daily, rf.REQUESTED_REBALANCE_DATES)
    return train, validation, daily, monthly_history, calendar


def btc_training_forecasts(train, monthly_history, calendar) -> pd.DataFrame:
    """Walk-forward BTC forecasts at every training origin (history < origin)."""
    complete = rf.complete_validation_months(train, calendar)
    actuals = rf.validation_actuals(train, complete)
    series = monthly_history["BTC"].dropna()
    rows = []
    for origin_month in sorted(train.index.to_period("M").unique()):
        history = series.loc[series.index < origin_month]
        for method in ["expanding_mean", *BTC_CANDIDATES]:
            for h in HORIZONS:
                pred = rf.forecast_one(history, origin_month, method, horizon=h)
                if pred is None:
                    continue
                target = origin_month + h - 1
                if target not in actuals.index:
                    continue
                realized = actuals.loc[target, "BTC"]
                if pd.isna(realized):
                    continue
                rows.append({
                    "origin": str(origin_month),
                    "target": str(target),
                    "method": method,
                    "h": h,
                    "forecast": float(pred),
                    "realized": float(realized),
                })
    return pd.DataFrame(rows)


def pair_sets(df: pd.DataFrame, method: str) -> dict:
    out = {}
    for h in HORIZONS:
        sub = df[(df["method"] == method) & (df["h"] == h)]
        out[h] = set(zip(sub["origin"], sub["target"]))
    return out


def score_on(df: pd.DataFrame, method: str, keep: dict) -> dict:
    """Combined RMSE h1-h3 using only (origin, target) pairs in keep[h]."""
    mse, n = {}, {}
    for h in HORIZONS:
        sub = df[(df["method"] == method) & (df["h"] == h)]
        mask = np.array(
            [(o, t) in keep[h] for o, t in zip(sub["origin"], sub["target"])],
            dtype=bool,
        )
        sub = sub.loc[mask]
        err = sub["forecast"].to_numpy() - sub["realized"].to_numpy()
        n[h] = int(len(err))
        mse[h] = float(np.mean(err ** 2)) if len(err) else np.nan
    values = [mse[h] for h in HORIZONS]
    combined = (
        float(np.sqrt(np.mean(values))) if np.isfinite(values).all() else np.nan
    )
    return {"n": n, "mse": mse, "combined": combined}


def pick_equity_bond() -> dict:
    scores = pd.read_csv(SCORES_FILE)
    eligible = scores["eligible"].astype(str).str.lower() == "true"
    picks = {}
    for asset in ("Equity", "Bond"):
        sub = scores[
            (scores["asset"] == asset)
            & scores["method"].str.startswith("reg_")
            & eligible
        ].copy()
        if sub.empty:
            raise RuntimeError(f"{asset}: no eligible regression in training scores.")
        sub["_rank"] = sub["method"].map(
            {m: i for i, m in enumerate(rf.REGRESSION_METHODS)}
        )
        best = sub.sort_values(
            ["combined_rmse_h1_to_h3", "_rank"], kind="stable"
        ).iloc[0]
        picks[asset] = str(best["method"])
    return picks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, daily, monthly_history, calendar = load_inputs()
    btc = btc_training_forecasts(train, monthly_history, calendar)
    if btc.empty:
        raise RuntimeError("No BTC training forecasts were produced.")

    # Fidelity check: my expanding-mean score must equal the module's.
    scores = pd.read_csv(SCORES_FILE)
    ref = float(scores[
        (scores["asset"] == "BTC") & (scores["method"] == "expanding_mean")
    ]["combined_rmse_h1_to_h3"].iloc[0])
    own = {m: pair_sets(btc, m) for m in ["expanding_mean", *BTC_CANDIDATES]}
    mine = score_on(btc, "expanding_mean", own["expanding_mean"])["combined"]
    print(f"BTC expanding_mean training combined RMSE: "
          f"this script {mine:.6f} | module file {ref:.6f}")
    if abs(mine - ref) > 1e-9:
        raise RuntimeError(
            "Replication does not match the module's training score; stop."
        )

    # Eligibility on each candidate's own dates.
    print("\nBTC regression candidates, own training dates:")
    eligible = []
    for m in BTC_CANDIDATES:
        n_own = {h: len(own[m][h]) for h in HORIZONS}
        ok = all(n_own[h] >= rf.MIN_VALIDATION_MONTHS for h in HORIZONS)
        print(f"  {m}: scored months h1/h2/h3 = "
              f"{n_own[1]}/{n_own[2]}/{n_own[3]} | eligible={ok}")
        if ok:
            eligible.append(m)

    btc_pick = "expanding_mean"
    pick_note = "no eligible BTC regression; historical mean used"
    common_window = None
    if eligible:
        common = {
            h: set.intersection(*[own[m][h] for m in eligible])
            for h in HORIZONS
        }
        n_common = {h: len(common[h]) for h in HORIZONS}
        if min(n_common.values()) >= rf.MIN_VALIDATION_MONTHS:
            exp_res = score_on(btc, "expanding_mean", common)
            result_rows = []
            for m in eligible:
                res = score_on(btc, m, common)
                result_rows.append({
                    "method": m,
                    "n_h1": res["n"][1], "n_h2": res["n"][2], "n_h3": res["n"][3],
                    "combined_rmse": res["combined"],
                    "expanding_mean_rmse_same_dates": exp_res["combined"],
                    "skill_vs_expanding": 1.0 - res["combined"] / exp_res["combined"],
                })
            table = pd.DataFrame(result_rows)
            order = {m: i for i, m in enumerate(BTC_CANDIDATES)}
            table["_rank"] = table["method"].map(order)
            table = table.sort_values(
                ["combined_rmse", "_rank"], kind="stable"
            ).drop(columns="_rank")
            btc_pick = str(table.iloc[0]["method"])
            origins = sorted(o for o, _ in common[1])
            common_window = (origins[0], origins[-1])
            pick_note = "lowest combined RMSE among eligible regressions"
            print(f"\nCommon training window (h1 origins): "
                  f"{common_window[0]} to {common_window[1]}")
            print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
            table.to_csv(OUT_DIR / "btc_regression_training_scores.csv",
                         index=False)
            skill = float(table.iloc[0]["skill_vs_expanding"])
            print(f"\nBTC pick: {btc_pick} (skill vs historical mean on the "
                  f"same training dates: {skill:+.4f}; positive = better)")
            if skill <= 0:
                print("NOTE: the picked regression does NOT beat the historical "
                      "mean on training.")
        else:
            pick_note = ("eligible regressions share fewer than "
                         f"{rf.MIN_VALIDATION_MONTHS} common months; "
                         "historical mean used")
            print("\n" + pick_note)

    methods = pick_equity_bond()
    methods["BTC"] = btc_pick
    print("\nMethods by asset:")
    for asset in rf.ASSETS:
        print(f"  {asset}: {methods[asset]}")
    print(f"  (BTC: {pick_note})")

    # Build walk-forward validation forecasts with the unchanged module code.
    selected = {a: {"method": m} for a, m in methods.items()}
    long = rf.build_walk_forward_validation_forecasts(
        validation, daily, monthly_history, selected, dict(methods), calendar
    )
    wide = rf.selected_forecasts_to_wide(long, include_realized=False)
    n_orig = wide["rebalance_date"].nunique()
    n_nan = int(wide[list(rf.ASSETS)].isna().sum().sum())
    wide.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {OUT_FILE}")
    print(f"rows={len(wide)} (expect 84), origins={n_orig} (expect 28), NaNs={n_nan}")

    print("\nMonthly forecasts (%):")
    summary = []
    for asset in rf.ASSETS:
        for h in HORIZONS:
            v = wide.loc[wide["month_ahead"].astype(int) == h, asset] * 100
            summary.append({"asset": asset, "h": h, "min": v.min(),
                            "mean": v.mean(), "max": v.max()})
    print(pd.DataFrame(summary).to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    # Equity and Bond should equal the previous training-best file.
    if PREV_FILE.exists():
        prev = pd.read_csv(PREV_FILE)
        key = ["rebalance_date", "month_ahead"]
        merged = wide.merge(prev, on=key, suffixes=("", "_prev"))
        same = all(
            np.allclose(merged[a], merged[f"{a}_prev"], rtol=0, atol=1e-12)
            for a in ("Equity", "Bond")
        )
        print(f"\nEquity and Bond identical to previous training-best file: {same}")
        if not same:
            print("STOP: they should be identical. Investigate before running MPO.")

    meta = {
        "methods_by_asset": methods,
        "btc_rule": pick_note,
        "btc_common_training_window_h1_origins": common_window,
        "btc_candidates": BTC_CANDIDATES,
        "note": ("BTC regression scored on the training dates shared by all "
                 "eligible candidates; Equity/Bond from the module's "
                 "training scores. portfolio_test.csv not read."),
    }
    with (OUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()