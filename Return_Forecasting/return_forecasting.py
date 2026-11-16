"""Separate monthly expected-return forecasts for Equity, Bond and BTC.

Run from any directory:
    python Return_Forecasting/return_forecasting.py

Reads:
  - portfolio_train.csv and portfolio_validation.csv only
  - long adjusted-price histories for ^SP500TR and AGG from Yahoo Finance,
    downloaded only through the final validation date (or a local cache)
  - HMM expected_returns.csv, if present, for a separate comparison

Does NOT read portfolio_test.csv and does NOT estimate covariance matrices.

Return convention:
  - Monthly compounded simple returns, in decimals.
  - Static forecasts are held constant across forecast months 1-3.
  - Equity and Bond use their own longer histories before 2015; from 2015
    onward, their model history uses the team's aligned daily returns.
  - BTC uses the aligned daily returns from 2015 onward. January 2015 is
    excluded for BTC because the aligned price sample does not include a
    preceding common-session close.

Predictors:
  - AR(1): previous completed month's return.
  - 12-1 momentum: compounded return from months t-12 through t-2 when
    predicting month t; the most recent completed month is excluded.
  - Both regressions are fit separately by asset with an expanding sample.

Selection:
  - Choose among static candidates using the square root of the equally
    weighted average of MSE for horizons 1, 2, and 3.
  - A candidate must have at least MIN_VALIDATION_MONTHS scored forecasts
    at each horizon and match the expanding-mean benchmark's scored dates.
  - The separate one-month optimizer output is for rebalance dates after
    the latest complete validation month.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import exchange_calendars as xcals
except ImportError as exc:
    raise ImportError(
        "Install dependencies with: "
        "python -m pip install numpy pandas yfinance exchange_calendars"
    ) from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
PRICE_CACHE = HERE / "long_equity_bond_prices.csv"

ASSETS = ("Equity", "Bond", "BTC")
CALENDAR = "XNYS"
YF_START = "1900-01-01"
HORIZON = 3
MIN_HISTORY_MONTHS = 12
MIN_REGRESSION_PAIRS = 60
MIN_VALIDATION_MONTHS = 12

ROLLING_WINDOWS = {
    "rolling_mean_1y": 12,
    "rolling_mean_3y": 36,
    "rolling_mean_5y": 60,
    "rolling_mean_10y": 120,
}
REGRESSION_METHODS = [
    "reg_ar1_expanding",
    "reg_ar1_recursive_expanding",
    "reg_mom12_1_expanding",
]
STATIC_METHODS = [
    "expanding_mean",
    *ROLLING_WINDOWS.keys(),
    *REGRESSION_METHODS,
]

SHRINKAGE_WEIGHT = 0.5
SHRINKAGE_METHOD = "selected_shrinkage_50"

# Optional explicit dates for the one-month optimizer output.
# Leave empty to use the first XNYS session after the latest complete
# validation month.
REQUESTED_REBALANCE_DATES: tuple[str, ...] = ()

# Freeze the current selected methods as the bases for this comparison.
SHRINKAGE_BASE_METHODS = {
    "Equity": "reg_ar1_expanding",
    "Bond": "rolling_mean_10y",
    "BTC": "rolling_mean_1y",
}

# ---------------------------------------------------------------------
# Input and data preparation
# ---------------------------------------------------------------------

def normalise_dates(index) -> pd.DatetimeIndex:
    """Return a sorted-style, timezone-naive, midnight-normalized index."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def find_split_dir() -> Path:
    """Find the existing train/validation splits without opening the test set."""
    candidates = [
        ROOT / "Data Work",
        ROOT / "HMM_regime_forecasts",
        ROOT / "Data" / "datasets",
    ]
    for folder in candidates:
        if all((folder / f"portfolio_{s}.csv").exists()
               for s in ("train", "validation")):
            return folder
    raise FileNotFoundError(
        "Could not find portfolio_train.csv and portfolio_validation.csv. "
        "Expected them in Data Work/ or HMM_regime_forecasts/."
    )


def read_return_split(folder: Path, split: str) -> pd.DataFrame:
    """Read one allowed split; this module never calls this with split='test'."""
    frame = pd.read_csv(
        folder / f"portfolio_{split}.csv",
        index_col=0,
        parse_dates=True,
    )
    if frame.empty or set(frame.columns) != set(ASSETS):
        raise ValueError(
            f"{split}: expected nonempty columns {ASSETS}, "
            f"got {tuple(frame.columns)}"
        )

    frame = frame.loc[:, list(ASSETS)].astype(float)
    frame.index = normalise_dates(frame.index)
    frame = frame.sort_index()

    values = frame.to_numpy(dtype=float)
    if (frame.index.hasnans or not frame.index.is_unique
            or not np.isfinite(values).all() or (values <= -1).any()):
        raise ValueError(f"{split}: invalid dates or returns")

    return frame


def extract_adjusted_close(download: pd.DataFrame, ticker: str) -> pd.Series:
    """Handle both flat and MultiIndex columns returned by yfinance."""
    if download.empty:
        raise ValueError(f"Yahoo Finance returned no observations for {ticker}.")

    if isinstance(download.columns, pd.MultiIndex):
        close_columns = [
            col for col in download.columns
            if "Close" in [str(level) for level in col]
        ]
        if not close_columns:
            raise ValueError(f"No Close column found for {ticker}.")
        series = download[close_columns[0]]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    else:
        if "Close" not in download.columns:
            raise ValueError(f"No Close column found for {ticker}.")
        series = download["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

    series = pd.to_numeric(series, errors="coerce").dropna()
    series.index = normalise_dates(series.index)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[series > 0]
    return series


def load_long_equity_bond_prices(validation_end: pd.Timestamp) -> pd.DataFrame:
    """Load/cache long adjusted prices, stopping at the validation endpoint.

    The local cache is also restricted to the validation endpoint before it is
    used. No post-validation price observations enter the forecasting history.
    """
    validation_end = pd.Timestamp(validation_end).normalize()

    if PRICE_CACHE.exists():
        cached = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        if set(("Equity", "Bond")).issubset(cached.columns):
            cached = cached.loc[:, ["Equity", "Bond"]].astype(float)
            cached.index = normalise_dates(cached.index)
            cached = cached.sort_index()
            cached = cached.loc[cached.index <= validation_end]
            if (not cached.empty
                    and cached.index.max() >= validation_end
                    and cached[["Equity", "Bond"]].notna().any().all()):
                return cached

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "The first run needs yfinance. Install it with: "
            "python -m pip install yfinance"
        ) from exc

    # yfinance's end parameter is exclusive. Add one calendar day so the
    # validation endpoint itself can be included if it is a trading session.
    end_exclusive = (
        validation_end + pd.Timedelta(days=1)
    ).date().isoformat()

    prices = {}
    for asset, ticker in (("Equity", "^SP500TR"), ("Bond", "AGG")):
        raw = yf.download(
            ticker,
            start=YF_START,
            end=end_exclusive,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        prices[asset] = extract_adjusted_close(raw, ticker)

    result = pd.concat(prices, axis=1).sort_index()
    result = result.loc[result.index <= validation_end]
    result = result.loc[:, ["Equity", "Bond"]]

    if result.empty:
        raise ValueError("No long Equity/Bond price observations were downloaded.")

    # Cache only data up to the validation endpoint.
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(PRICE_CACHE)

    for asset in ("Equity", "Bond"):
        first = result[asset].first_valid_index()
        if first is None or first >= pd.Timestamp("2015-01-01"):
            warnings.warn(
                f"{asset} Yahoo history does not extend before 2015 "
                f"(first valid date: {first}); long-history estimates for "
                "this asset will not be available.",
                stacklevel=2,
            )

    return result


def monthly_compounded_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily simple returns within calendar months."""
    rows = []
    periods = []
    for period, group in daily.groupby(daily.index.to_period("M"), sort=True):
        values = group.to_numpy(dtype=float)
        rows.append(np.prod(1.0 + values, axis=0) - 1.0)
        periods.append(period)

    result = pd.DataFrame(
        rows,
        index=pd.PeriodIndex(periods, freq="M"),
        columns=daily.columns,
    )
    result.index.name = "month"
    return result


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert adjusted daily closes to close-to-close monthly simple returns."""
    monthly_close = prices.groupby(
        prices.index.to_period("M"), sort=True
    ).last()
    result = monthly_close.pct_change(fill_method=None)
    result.index = pd.PeriodIndex(result.index, freq="M")
    result.index.name = "month"
    return result


def build_asset_monthly_history(
    daily_returns: pd.DataFrame,
    long_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build independent monthly histories; no shared start date is required."""
    aligned = monthly_compounded_returns(daily_returns)
    long_monthly = monthly_returns_from_prices(long_prices)

    jan_2015 = pd.Period("2015-01", freq="M")
    feb_2015 = pd.Period("2015-02", freq="M")

    histories = {}
    for asset in ("Equity", "Bond"):
        # The long price history supplies the pre-2015 series and a complete
        # January 2015 return. Use the team's aligned returns thereafter so
        # the recent history matches the portfolio/HMM return convention.
        pre = long_monthly.loc[
            long_monthly.index <= jan_2015, asset
        ].dropna()
        recent = aligned.loc[
            aligned.index >= feb_2015, asset
        ].dropna()
        histories[asset] = pd.concat([pre, recent]).sort_index()

    # BTC starts with the 2015-onward common-session return history. January
    # is omitted because the common price sample has no preceding close.
    histories["BTC"] = aligned.loc[
        aligned.index >= feb_2015, "BTC"
    ].dropna()

    result = pd.DataFrame(histories).sort_index()
    for asset in ASSETS:
        s = result[asset].dropna()
        if len(s) and (not np.isfinite(s.to_numpy()).all()
                       or (s.to_numpy() <= -1).any()):
            raise ValueError(f"{asset}: invalid monthly simple returns")
    return result


# ---------------------------------------------------------------------
# Forecast models
# ---------------------------------------------------------------------

def regression_feature(
    history: pd.Series,
    target_month: pd.Period,
    kind: str,
) -> float | None:
    """Create a predictor using only returns before target_month."""
    if kind == "ar1":
        value = history.get(target_month - 1, np.nan)
        return float(value) if pd.notna(value) else None

    if kind == "mom12_1":
        # For target t, use months t-12 through t-2 (11 observations), leaving
        # out the latest completed month t-1.
        months = pd.period_range(
            target_month - 12, target_month - 2, freq="M"
        )
        values = history.reindex(months)
        if values.isna().any():
            return None
        gross = 1.0 + values.to_numpy(dtype=float)
        if (gross <= 0).any():
            return None
        return float(np.prod(gross) - 1.0)

    raise ValueError(f"Unknown regression feature: {kind}")


def forecast_one(
    history: pd.Series,
    origin_month: pd.Period,
    method: str,
    horizon: int = 1,
) -> float | None:
    """Forecast a monthly return using information before origin_month.

    For reg_ar1_recursive_expanding, forecast months 2 and 3 recursively:
        forecast[h] = intercept + slope * forecast[h - 1]
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    history = history.loc[history.index < origin_month].dropna().sort_index()

    if method == "expanding_mean":
        if len(history) < MIN_HISTORY_MONTHS:
            return None
        return float(history.mean())

    if method in ROLLING_WINDOWS:
        window = ROLLING_WINDOWS[method]
        needed = pd.period_range(
            end=origin_month - 1, periods=window, freq="M"
        )
        values = history.reindex(needed)
        if values.isna().any():
            return None
        return float(values.mean())

    if method in REGRESSION_METHODS:
        recursive_ar1 = method == "reg_ar1_recursive_expanding"
        kind = (
            "mom12_1"
            if method == "reg_mom12_1_expanding"
            else "ar1"
        )

        xs, ys = [], []

        # Build past predictor/return pairs using only information
        # available before each historical target month.
        for target_month, realized in history.items():
            x = regression_feature(history, target_month, kind)
            if x is not None and np.isfinite(x):
                xs.append(float(x))
                ys.append(float(realized))

        if len(xs) < MIN_REGRESSION_PAIRS:
            return None

        current_x = regression_feature(history, origin_month, kind)
        if current_x is None:
            return None

        design = np.column_stack([np.ones(len(xs)), np.asarray(xs)])
        coefficients, *_ = np.linalg.lstsq(
            design, np.asarray(ys), rcond=None
        )

        # First forecast month, using the last observed predictor.
        prediction = float(coefficients[0] + coefficients[1] * current_x)

        # For recursive AR(1), use each predicted return as the next month's
        # predictor. Other methods retain their existing behavior.
        if recursive_ar1:
            for _ in range(1, horizon):
                prediction = float(
                    coefficients[0] + coefficients[1] * prediction
                )
                if not np.isfinite(prediction) or prediction <= -1.0:
                    return None

        if not np.isfinite(prediction) or prediction <= -1.0:
            return None
        return prediction

    raise ValueError(f"Unknown return-forecast method: {method}")

def forecast_selected_method(
    history: pd.Series,
    origin_month: pd.Period,
    asset: str,
    method: str,
    horizon: int = 1,
    shrinkage_base_method: str | None = None,
) -> float | None:
    """Forecast using a normal method or the selected-method shrinkage blend."""
    if method != SHRINKAGE_METHOD:
        return forecast_one(
            history,
            origin_month,
            method,
            horizon=horizon,
        )

    # The historical walk-forward series can use a base method selected
    # from training data. Existing forward forecasts keep using the
    # original configured base method.
    base_method = (
        shrinkage_base_method or SHRINKAGE_BASE_METHODS[asset]
    )

    base_forecast = forecast_one(
        history,
        origin_month,
        base_method,
        horizon=horizon,
    )
    expanding_forecast = forecast_one(
        history,
        origin_month,
        "expanding_mean",
        horizon=horizon,
    )

    if base_forecast is None or expanding_forecast is None:
        return None

    return (
        SHRINKAGE_WEIGHT * base_forecast
        + (1.0 - SHRINKAGE_WEIGHT) * expanding_forecast
    )

# ---------------------------------------------------------------------
# Validation and scoring
# ---------------------------------------------------------------------

def build_calendar(validation: pd.DataFrame, extra_dates=()):
    start = validation.index.min() - pd.Timedelta(days=7)
    end = validation.index.max() + pd.DateOffset(months=HORIZON + 2)

    if extra_dates:
        requested = normalise_dates(extra_dates)
        if requested.hasnans:
            raise ValueError("Requested rebalance dates contain an invalid date.")

        requested_start = (
            requested.min().to_period("M").start_time.normalize()
            - pd.Timedelta(days=7)
        )
        requested_end = max(
            date.to_period("M").end_time.normalize()
            for date in requested
        ) + pd.Timedelta(days=7)

        start = min(start, requested_start)
        end = max(end, requested_end)

    return xcals.get_calendar(
        CALENDAR,
        start=str(start.date()),
        end=str(end.date()),
    )


def sessions_for_month(calendar, month: pd.Period) -> pd.DatetimeIndex:
    sessions = calendar.sessions_in_range(
        month.start_time.normalize(),
        month.end_time.normalize(),
    )
    return normalise_dates(sessions)


def complete_validation_months(
    validation: pd.DataFrame,
    calendar,
) -> list[pd.Period]:
    complete = []
    for month in sorted(validation.index.to_period("M").unique()):
        actual_dates = validation.index[
            validation.index.to_period("M") == month
        ]
        expected_dates = sessions_for_month(calendar, month)
        if len(expected_dates) and actual_dates.equals(expected_dates):
            complete.append(month)
    return complete

def validation_selection_cutoff(
    validation: pd.DataFrame,
    complete_months: list[pd.Period],
) -> pd.Timestamp:
    """Last date whose completed-month outcomes can affect model selection."""
    if not complete_months:
        raise ValueError("No complete validation month is available.")

    last_complete_month = max(complete_months)
    dates = validation.index[
        validation.index.to_period("M") == last_complete_month
    ]
    if not len(dates):
        raise ValueError(
            f"Could not find validation dates for {last_complete_month}."
        )
    return dates[-1]


def default_rebalance_date(
    complete_months: list[pd.Period],
    calendar,
) -> pd.Timestamp:
    """First XNYS session after the latest complete validation month."""
    if not complete_months:
        raise ValueError("No complete validation month is available.")

    forecast_month = max(complete_months) + 1
    sessions = sessions_for_month(calendar, forecast_month)
    if not len(sessions):
        raise ValueError(f"No exchange sessions found for {forecast_month}.")
    return sessions[0]

def validation_actuals(
    validation: pd.DataFrame,
    complete_months: list[pd.Period],
) -> pd.DataFrame:
    actual = monthly_compounded_returns(validation)
    return actual.reindex(complete_months)


def add_validation_forecasts(
    daily_returns: pd.DataFrame,
    validation: pd.DataFrame,
    monthly_history: pd.DataFrame,
    actuals: pd.DataFrame,
) -> list[dict]:
    rows = []
    validation_months = sorted(validation.index.to_period("M").unique())

    for origin_month in validation_months:
        origin_dates = validation.index[
            validation.index.to_period("M") == origin_month
        ]
        rebalance_date = origin_dates[0]
        previous_dates = daily_returns.index[
            daily_returns.index < rebalance_date
        ]
        information_date = (
            previous_dates[-1] if len(previous_dates) else pd.NaT
        )

        for asset in ASSETS:
            series = monthly_history[asset].dropna()
            history = series.loc[series.index < origin_month]
            methods = list(STATIC_METHODS)
            if asset == "BTC":
                methods = [
                    "expanding_mean",
                    *ROLLING_WINDOWS.keys(),
                ]

            for method in methods:
                for horizon in range(1, HORIZON + 1):
                    prediction = forecast_one(
                        history,
                        origin_month,
                        method,
                        horizon=horizon,
                    )
                    if prediction is None:
                        continue

                    target_month = origin_month + horizon - 1
                    realized = (
                        float(actuals.loc[target_month, asset])
                        if target_month in actuals.index
                        else np.nan
                    )

                    rows.append({
                        "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                        "information_date": (
                            information_date.strftime("%Y-%m-%d")
                            if pd.notna(information_date) else ""
                        ),
                        "asset": asset,
                        "method": method,
                        "month_ahead": horizon,
                        "forecast_month": str(target_month),
                        "forecast_return": prediction,
                        "realized_return": realized,
                        "error": (
                            prediction - realized
                            if pd.notna(realized) else np.nan
                        ),
                    })

    return rows


def add_hmm_forecasts(
    validation: pd.DataFrame,
    actuals: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> list[dict]:
    """Append the existing HMM's already probability-weighted forecasts."""
    path = ROOT / "HMM_regime_forecasts" / "hmm_outputs" / "expected_returns.csv"
    if not path.exists():
        print(f"HMM forecast file not found; skipping comparison: {path}")
        return []

    frame = pd.read_csv(path)
    required = {
        "rebalance_date", "month_ahead", "forecast_month", *ASSETS
    }
    if not required.issubset(frame.columns):
        warnings.warn(
            f"HMM forecast file has unexpected columns; skipping: {path}",
            stacklevel=2,
        )
        return []

    validation_rebalances = {
        group.index[0].strftime("%Y-%m-%d")
        for _, group in validation.groupby(
            validation.index.to_period("M"), sort=True
        )
    }
    rows = []

    for _, record in frame.iterrows():
        rebalance_date = pd.Timestamp(record["rebalance_date"]).strftime(
            "%Y-%m-%d"
        )
        if rebalance_date not in validation_rebalances:
            continue

        month_ahead = int(record["month_ahead"])
        if not 1 <= month_ahead <= HORIZON:
            continue
        target_month = pd.Period(str(record["forecast_month"]), freq="M")

        previous_dates = daily_returns.index[
            daily_returns.index < pd.Timestamp(rebalance_date)
        ]
        information_date = (
            previous_dates[-1] if len(previous_dates) else pd.NaT
        )

        for asset in ASSETS:
            prediction = float(record[asset])
            realized = (
                float(actuals.loc[target_month, asset])
                if target_month in actuals.index
                else np.nan
            )
            rows.append({
                "rebalance_date": rebalance_date,
                "information_date": (
                    information_date.strftime("%Y-%m-%d")
                    if pd.notna(information_date) else ""
                ),
                "asset": asset,
                "method": "hmm_regime",
                "month_ahead": month_ahead,
                "forecast_month": str(target_month),
                "forecast_return": prediction,
                "realized_return": realized,
                "error": (
                    prediction - realized
                    if pd.notna(realized) else np.nan
                ),
            })

    return rows


def score_forecasts(
    forecasts: pd.DataFrame,
    methods_by_asset: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute monthly forecast errors and a consistent OOS signal/noise ratio.

    SNR = RMS(predicted return) / RMSE(forecast error).
    """
    results = []
    key_columns = ["rebalance_date", "forecast_month"]

    for asset in ASSETS:
        baseline = forecasts.loc[
            (forecasts["asset"] == asset)
            & (forecasts["method"] == "expanding_mean")
            & forecasts["realized_return"].notna()
        ]

        baseline_h1 = baseline.loc[baseline["month_ahead"] == 1]
        baseline_pairs_h1 = set(zip(
            baseline_h1["rebalance_date"],
            baseline_h1["forecast_month"],
        ))

        for method in methods_by_asset[asset]:
            for horizon in range(1, HORIZON + 1):
                all_rows = forecasts.loc[
                    (forecasts["asset"] == asset)
                    & (forecasts["method"] == method)
                    & (forecasts["month_ahead"] == horizon)
                ]
                scored = all_rows.dropna(
                    subset=["forecast_return", "realized_return"]
                )

                n_forecasts = int(all_rows["forecast_return"].notna().sum())
                n_scored = len(scored)

                rmse = mae = bias = signal_rms = snr = np.nan
                skill = paired_n = np.nan
                full_coverage = False

                if n_scored:
                    errors = (
                        scored["forecast_return"].to_numpy()
                        - scored["realized_return"].to_numpy()
                    )
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    mae = float(np.mean(np.abs(errors)))
                    bias = float(np.mean(errors))
                    signal_rms = float(np.sqrt(np.mean(
                        scored["forecast_return"].to_numpy() ** 2
                    )))
                    snr = (
                        signal_rms / rmse if rmse > 0 else np.nan
                    )

                    candidate_pairs = set(zip(
                        scored["rebalance_date"],
                        scored["forecast_month"],
                    ))
                    if horizon == 1:
                        full_coverage = (
                            bool(baseline_pairs_h1)
                            and candidate_pairs == baseline_pairs_h1
                        )

                    if method == "expanding_mean":
                        skill = 0.0
                        paired_n = n_scored
                    else:
                        base_for_pairing = baseline[
                            ["rebalance_date", "forecast_month",
                             "forecast_return"]
                        ].rename(columns={
                            "forecast_return": "baseline_forecast"
                        })
                        paired = scored[
                            key_columns + ["forecast_return", "realized_return"]
                        ].merge(
                            base_for_pairing,
                            on=key_columns,
                            how="inner",
                        )
                        paired_n = len(paired)
                        if paired_n:
                            candidate_rmse = float(np.sqrt(np.mean(
                                (paired["forecast_return"]
                                 - paired["realized_return"]) ** 2
                            )))
                            baseline_rmse = float(np.sqrt(np.mean(
                                (paired["baseline_forecast"]
                                 - paired["realized_return"]) ** 2
                            )))
                            if baseline_rmse > 0:
                                skill = 1.0 - candidate_rmse / baseline_rmse

                results.append({
                    "asset": asset,
                    "method": method,
                    "horizon_months": horizon,
                    "n_forecasts": n_forecasts,
                    "n_scored": n_scored,
                    "rmse": rmse,
                    "mae": mae,
                    "bias_forecast_minus_realized": bias,
                    "rms_forecast": signal_rms,
                    "oos_snr_rms_over_rmse": snr,
                    "rmse_skill_vs_expanding_common_dates": skill,
                    "paired_n_vs_expanding": paired_n,
                    "full_validation_coverage_h1": (
                        full_coverage if horizon == 1 else np.nan
                    ),
                })

    return pd.DataFrame(results)

def summarize_shrinkage_comparison(
    scores: pd.DataFrame,
    base_methods: dict[str, str],
) -> pd.DataFrame:
    """Compare expanding mean, selected forecast, and its 50/50 blend."""
    rows = []

    for asset in ASSETS:
        comparisons = [
            ("expanding_mean", "expanding_mean"),
            ("selected_method", base_methods[asset]),
            ("shrinkage_50_50", SHRINKAGE_METHOD),
        ]
        asset_rows = []

        for label, method in comparisons:
            method_scores = scores.loc[
                (scores["asset"] == asset)
                & (scores["method"] == method)
            ]

            row = {
                "asset": asset,
                "comparison": label,
                "forecast_method": method,
                "shrinkage_base_method": base_methods[asset],
            }
            horizon_rmse = []

            for horizon in range(1, HORIZON + 1):
                result = method_scores.loc[
                    method_scores["horizon_months"] == horizon
                ]

                if len(result) != 1:
                    raise ValueError(
                        f"Expected one score for {asset}, {method}, "
                        f"horizon {horizon}."
                    )

                result = result.iloc[0]
                rmse = result["rmse"]
                row[f"n_scored_h{horizon}"] = int(result["n_scored"])
                row[f"rmse_h{horizon}"] = rmse

                if pd.notna(rmse):
                    horizon_rmse.append(float(rmse))

            row["combined_rmse_h1_to_h3"] = (
                float(np.sqrt(np.mean(np.square(horizon_rmse))))
                if len(horizon_rmse) == HORIZON else np.nan
            )
            asset_rows.append(row)

        reference = {
            row["comparison"]: row["combined_rmse_h1_to_h3"]
            for row in asset_rows
        }
        selected_rmse = reference["selected_method"]
        expanding_rmse = reference["expanding_mean"]

        for row in asset_rows:
            row["rmse_skill_vs_selected_method"] = np.nan
            row["rmse_skill_vs_expanding_mean"] = np.nan

            if row["comparison"] == "shrinkage_50_50":
                blend_rmse = row["combined_rmse_h1_to_h3"]
                if pd.notna(blend_rmse) and selected_rmse > 0:
                    row["rmse_skill_vs_selected_method"] = (
                        1.0 - blend_rmse / selected_rmse
                    )
                if pd.notna(blend_rmse) and expanding_rmse > 0:
                    row["rmse_skill_vs_expanding_mean"] = (
                        1.0 - blend_rmse / expanding_rmse
                    )

        rows.extend(asset_rows)

    return pd.DataFrame(rows)

def select_static_methods(
    forecasts: pd.DataFrame,
    static_methods_by_asset: dict[str, list[str]],
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Select static methods using equally weighted horizons 1, 2 and 3.

    Each candidate must have the same scored validation forecast dates as
    the expanding-mean benchmark at every horizon.
    """
    horizons = list(range(1, HORIZON + 1))
    score_rows = []

    for asset in ASSETS:
        # The expanding mean is the benchmark for checking that candidates
        # are scored on the same validation origins.
        benchmark_pairs = {}

        for horizon in horizons:
            baseline = forecasts.loc[
                (forecasts["asset"] == asset)
                & (forecasts["method"] == "expanding_mean")
                & (forecasts["month_ahead"] == horizon)
            ].dropna(subset=["forecast_return", "realized_return"])

            benchmark_pairs[horizon] = set(zip(
                baseline["rebalance_date"],
                baseline["forecast_month"],
            ))

        for method in static_methods_by_asset[asset]:
            row = {
                "asset": asset,
                "method": method,
                "eligible": True,
                "ineligible_reason": "",
            }
            mse_by_horizon = {}
            reasons = []

            for horizon in horizons:
                candidate = forecasts.loc[
                    (forecasts["asset"] == asset)
                    & (forecasts["method"] == method)
                    & (forecasts["month_ahead"] == horizon)
                ].dropna(subset=["forecast_return", "realized_return"])

                candidate_pairs = set(zip(
                    candidate["rebalance_date"],
                    candidate["forecast_month"],
                ))

                row[f"n_scored_h{horizon}"] = len(candidate)

                if candidate.empty:
                    row[f"rmse_h{horizon}"] = np.nan
                else:
                    errors = (
                        candidate["forecast_return"].to_numpy()
                        - candidate["realized_return"].to_numpy()
                    )
                    mse_by_horizon[horizon] = float(np.mean(errors ** 2))
                    row[f"rmse_h{horizon}"] = float(
                        np.sqrt(mse_by_horizon[horizon])
                    )

                if len(candidate) < MIN_VALIDATION_MONTHS:
                    reasons.append(
                        f"horizon {horizon}: fewer than "
                        f"{MIN_VALIDATION_MONTHS} scored months"
                    )

                if candidate_pairs != benchmark_pairs[horizon]:
                    reasons.append(
                        f"horizon {horizon}: validation dates differ "
                        "from expanding-mean benchmark"
                    )

            if not reasons and len(mse_by_horizon) == len(horizons):
                # Give each forecast horizon equal importance.
                row["combined_rmse_h1_to_h3"] = float(
                    np.sqrt(np.mean([
                        mse_by_horizon[h] for h in horizons
                    ]))
                )
            else:
                row["eligible"] = False
                row["combined_rmse_h1_to_h3"] = np.nan
                row["ineligible_reason"] = "; ".join(reasons)

            score_rows.append(row)

    combined_scores = pd.DataFrame(score_rows)
    combined_scores["combined_rmse_skill_vs_expanding"] = np.nan

    selected = {}

    for asset in ASSETS:
        asset_rows = combined_scores.loc[
            combined_scores["asset"] == asset
        ].copy()

        baseline_rows = asset_rows.loc[
            (asset_rows["method"] == "expanding_mean")
            & asset_rows["eligible"]
        ]

        if baseline_rows.empty:
            raise RuntimeError(
                f"{asset}: expanding-mean benchmark does not have "
                "enough complete validation forecasts for horizons 1-3."
            )

        baseline_score = float(
            baseline_rows.iloc[0]["combined_rmse_h1_to_h3"]
        )

        if baseline_score > 0:
            mask = combined_scores["asset"] == asset
            combined_scores.loc[
                mask, "combined_rmse_skill_vs_expanding"
            ] = (
                1.0
                - combined_scores.loc[
                    mask, "combined_rmse_h1_to_h3"
                ] / baseline_score
            )

        candidates = asset_rows.loc[asset_rows["eligible"]].copy()
        if candidates.empty:
            raise RuntimeError(
                f"{asset}: no static method has complete validation "
                "coverage at horizons 1-3."
            )

        # If scores tie, prefer the simpler method appearing first in the list.
        tie_rank = {
            method: rank
            for rank, method in enumerate(static_methods_by_asset[asset])
        }
        candidates["_tie_rank"] = candidates["method"].map(tie_rank)
        best = candidates.sort_values(
            ["combined_rmse_h1_to_h3", "_tie_rank"],
            kind="stable",
        ).iloc[0]

        selected[asset] = {
            "method": str(best["method"]),
            "status": "selected_by_combined_horizons_1_to_3_validation_rmse",
            "n_validation_origins_by_horizon": {
                str(h): int(best[f"n_scored_h{h}"])
                for h in horizons
            },
            "validation_rmse_by_horizon": {
                str(h): float(best[f"rmse_h{h}"])
                for h in horizons
            },
            "combined_validation_rmse_horizons_1_to_3": float(
                best["combined_rmse_h1_to_h3"]
            ),
            "combined_rmse_skill_vs_expanding": (
                float(
                    1.0
                    - float(best["combined_rmse_h1_to_h3"]) / baseline_score
                )
                if baseline_score > 0
                else None
            ),
        }

    return selected, combined_scores

def select_walk_forward_methods_from_train(
    train: pd.DataFrame,
    daily_returns: pd.DataFrame,
    monthly_history: pd.DataFrame,
    calendar,
    static_methods_by_asset: dict[str, list[str]],
) -> tuple[dict[str, dict], dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Select methods using training data only for validation-period forecasts.

    Candidate forecasts at historical training origins use only returns
    available before each origin. The selected methods are then held fixed
    during the validation period.
    """
    complete_train_months = complete_validation_months(train, calendar)
    if not complete_train_months:
        raise ValueError("Training data contains no complete calendar month.")

    train_actuals = validation_actuals(train, complete_train_months)
    train_forecasts = pd.DataFrame(add_validation_forecasts(
        daily_returns,
        train,
        monthly_history,
        train_actuals,
    ))
    if train_forecasts.empty:
        raise ValueError("No candidate forecasts were produced from training data.")

    # First select the best static candidate per asset using training
    # pseudo-out-of-sample forecasts.
    base_selected, training_static_scores = select_static_methods(
        train_forecasts,
        static_methods_by_asset,
    )
    base_methods = {
        asset: base_selected[asset]["method"]
        for asset in ASSETS
    }

    # Compare a 50/50 blend with the training-selected static method.
    shrinkage_rows = add_selected_shrinkage_validation_forecasts(
        train_forecasts,
        base_methods,
    )
    scored_forecasts = pd.concat(
        [train_forecasts, shrinkage_rows],
        ignore_index=True,
        sort=False,
    )

    methods_to_score = {
        asset: [*static_methods_by_asset[asset], SHRINKAGE_METHOD]
        for asset in ASSETS
    }
    training_scores = score_forecasts(scored_forecasts, methods_to_score)
    shrinkage_comparison = summarize_shrinkage_comparison(
        training_scores,
        base_methods,
    )

    selected = {}
    for asset in ASSETS:
        asset_comparison = shrinkage_comparison.loc[
            shrinkage_comparison["asset"] == asset
        ]
        base_rows = asset_comparison.loc[
            asset_comparison["comparison"] == "selected_method"
        ]
        blend_rows = asset_comparison.loc[
            asset_comparison["comparison"] == "shrinkage_50_50"
        ]

        if len(base_rows) != 1 or len(blend_rows) != 1:
            raise ValueError(
                f"Expected one training comparison for {asset}."
            )

        base_row = base_rows.iloc[0]
        blend_row = blend_rows.iloc[0]
        base_rmse = base_row["combined_rmse_h1_to_h3"]
        blend_rmse = blend_row["combined_rmse_h1_to_h3"]

        # Compare the blend only when it has exactly the same training
        # forecast coverage as its base method.
        same_coverage = True
        for horizon in range(1, HORIZON + 1):
            base_n = base_row[f"n_scored_h{horizon}"]
            blend_n = blend_row[f"n_scored_h{horizon}"]

            if (
                pd.isna(base_n)
                or pd.isna(blend_n)
                or int(base_n) < MIN_VALIDATION_MONTHS
                or int(blend_n) != int(base_n)
            ):
                same_coverage = False
                break

        use_blend = (
            same_coverage
            and pd.notna(base_rmse)
            and pd.notna(blend_rmse)
            and float(blend_rmse) < float(base_rmse)
        )

        chosen_row = blend_row if use_blend else base_row
        chosen_method = SHRINKAGE_METHOD if use_blend else base_methods[asset]

        selected[asset] = {
            "method": chosen_method,
            "status": "selected_using_training_only_pseudo_oos_rmse",
            "training_static_base_method": base_methods[asset],
            "training_combined_rmse_h1_to_h3": float(
                chosen_row["combined_rmse_h1_to_h3"]
            ),
            "training_rmse_by_horizon": {
                str(h): float(chosen_row[f"rmse_h{h}"])
                for h in range(1, HORIZON + 1)
            },
            "training_n_scored_by_horizon": {
                str(h): int(chosen_row[f"n_scored_h{h}"])
                for h in range(1, HORIZON + 1)
            },
            "blend_weight_on_base_method": (
                float(SHRINKAGE_WEIGHT) if use_blend else None
            ),
        }

    return (
        selected,
        base_methods,
        training_static_scores,
        shrinkage_comparison,
    )

# ---------------------------------------------------------------------
# Optimizer handoff tables
# ---------------------------------------------------------------------

HANDOFF_KEYS = [
    "rebalance_date", "information_date", "month_ahead", "forecast_month"
]


def selected_forecasts_to_wide(
    selected_long: pd.DataFrame,
    include_realized: bool,
) -> pd.DataFrame:
    """Produce an HMM-style wide return-vector table."""
    if selected_long.empty:
        return pd.DataFrame()

    rows = []
    for key, group in selected_long.groupby(
        HANDOFF_KEYS, sort=True, dropna=False
    ):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(HANDOFF_KEYS, key))

        for _, item in group.iterrows():
            asset = item["asset"]
            record[asset] = float(item["forecast_return"])
            record[f"method_{asset}"] = item["method"]
            if include_realized:
                record[f"realized_{asset}"] = item["realized_return"]

        rows.append(record)

    output = pd.DataFrame(rows)
    leading = HANDOFF_KEYS + list(ASSETS)
    methods = [f"method_{asset}" for asset in ASSETS]
    realized = (
        [f"realized_{asset}" for asset in ASSETS]
        if include_realized else []
    )
    columns = [c for c in leading + methods + realized if c in output.columns]
    return output.loc[:, columns].sort_values(
        ["rebalance_date", "month_ahead"]
    )

def build_walk_forward_validation_forecasts(
    validation: pd.DataFrame,
    daily_returns: pd.DataFrame,
    monthly_history: pd.DataFrame,
    selected_by_train: dict[str, dict],
    training_base_methods: dict[str, str],
    calendar,
) -> pd.DataFrame:
    """Produce 1-3 month forecasts at each validation rebalance.

    Rebalances are the first XNYS session of a month present in validation.
    Every horizon uses only history available before that rebalance month.
    """
    validation_dates = set(validation.index)
    rebalance_dates = []

    for month in sorted(validation.index.to_period("M").unique()):
        sessions = sessions_for_month(calendar, month)
        if len(sessions) and sessions[0] in validation_dates:
            rebalance_dates.append(sessions[0])

    if not rebalance_dates:
        raise ValueError(
            "No first-session monthly rebalance dates were found in validation."
        )

    complete_available_months = set(
        complete_validation_months(daily_returns, calendar)
    )
    rows = []

    for rebalance_date in sorted(rebalance_dates):
        origin_month = rebalance_date.to_period("M")
        information_month = origin_month - 1

        if information_month not in complete_available_months:
            raise ValueError(
                f"Cannot forecast {origin_month}: prior month "
                f"{information_month} is not complete in train/validation data."
            )

        prior_month_sessions = sessions_for_month(calendar, information_month)
        if not len(prior_month_sessions):
            raise ValueError(f"No XNYS sessions found for {information_month}.")

        dates_before_rebalance = daily_returns.index[
            daily_returns.index < rebalance_date
        ]
        if not len(dates_before_rebalance):
            raise ValueError(
                f"No daily history before {rebalance_date.date()}."
            )

        information_date = dates_before_rebalance[-1]
        expected_information_date = prior_month_sessions[-1]
        if information_date != expected_information_date:
            raise ValueError(
                f"Daily returns do not reach the final XNYS session of "
                f"{information_month}. Expected "
                f"{expected_information_date.date()}, found "
                f"{information_date.date()}."
            )

        for asset in ASSETS:
            series = monthly_history[asset].dropna()
            if information_month not in series.index:
                raise ValueError(
                    f"{asset} has no monthly return for completed month "
                    f"{information_month}."
                )

            # The forecast origin is the rebalance month; later outcomes are
            # not included in the model history.
            history = series.loc[series.index < origin_month]
            method = selected_by_train[asset]["method"]

            for horizon in range(1, HORIZON + 1):
                prediction = forecast_selected_method(
                    history,
                    origin_month,
                    asset,
                    method,
                    horizon=horizon,
                    shrinkage_base_method=training_base_methods[asset],
                )

                if prediction is None or not np.isfinite(prediction):
                    raise ValueError(
                        f"Could not forecast {asset}, horizon {horizon}, "
                        f"for rebalance {rebalance_date.date()} using {method}."
                    )

                target_month = origin_month + horizon - 1
                rows.append({
                    "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                    "information_date": information_date.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "method": method,
                    "month_ahead": horizon,
                    "forecast_month": str(target_month),
                    "forecast_return": float(prediction),
                })

    return pd.DataFrame(rows)
def build_optimizer_one_month_forecasts(
    requested_rebalance_dates,
    validation: pd.DataFrame,
    daily_returns: pd.DataFrame,
    monthly_history: pd.DataFrame,
    complete_months: list[pd.Period],
    selected: dict[str, dict],
    calendar,
) -> pd.DataFrame:
    """Build one full-calendar-month return vector per requested rebalance.

    Dates must be after the validation outcomes used for model selection and
    must be the first XNYS session of a month. The previous calendar month
    must be complete and present in the input history.
    """
    if not requested_rebalance_dates:
        raise ValueError("No optimizer rebalance dates were requested.")

    cutoff = validation_selection_cutoff(validation, complete_months)
    complete_month_set = set(complete_months)

    dates = []
    for value in requested_rebalance_dates:
        date = pd.Timestamp(value)
        if pd.isna(date):
            raise ValueError(f"Invalid rebalance date: {value!r}")
        if date.tz is not None:
            date = date.tz_localize(None)
        dates.append(date.normalize())

    if len(set(dates)) != len(dates):
        raise ValueError("Requested rebalance dates contain duplicates.")

    rows = []

    for rebalance_date in sorted(dates):
        if rebalance_date <= cutoff:
            raise ValueError(
                f"Rebalance date {rebalance_date.date()} must be after the "
                f"model-selection cutoff {cutoff.date()}."
            )

        forecast_month = rebalance_date.to_period("M")
        month_sessions = sessions_for_month(calendar, forecast_month)

        if not len(month_sessions) or rebalance_date not in month_sessions:
            raise ValueError(
                f"{rebalance_date.date()} is not an XNYS trading session."
            )
        if rebalance_date != month_sessions[0]:
            raise ValueError(
                f"{rebalance_date.date()} is not the first XNYS session of "
                f"{forecast_month}. This output forecasts a full calendar "
                "month, so rebalance dates must be the month's first session."
            )

        history_month = forecast_month - 1
        if history_month not in complete_month_set:
            raise ValueError(
                f"Cannot forecast {forecast_month}: prior month "
                f"{history_month} is not a complete validation month in the "
                "available data. Update the monthly data before requesting "
                "this rebalance date."
            )

        prior_month_sessions = sessions_for_month(calendar, history_month)
        if not len(prior_month_sessions):
            raise ValueError(f"No exchange sessions found for {history_month}.")

        dates_before_rebalance = daily_returns.index[
            daily_returns.index < rebalance_date
        ]
        if not len(dates_before_rebalance):
            raise ValueError(
                f"No daily return history is available before "
                f"{rebalance_date.date()}."
            )

        information_date = dates_before_rebalance[-1]
        expected_information_date = prior_month_sessions[-1]
        if information_date != expected_information_date:
            raise ValueError(
                f"Daily returns do not reach the final XNYS session of "
                f"{history_month}. Expected {expected_information_date.date()}, "
                f"found {information_date.date()}."
            )

        for asset in ASSETS:
            series = monthly_history[asset].dropna()
            if history_month not in series.index:
                raise ValueError(
                    f"{asset} has no monthly return for required completed "
                    f"month {history_month}."
                )

            # Keep only months completed before the forecast month.
            history = series.loc[series.index < forecast_month]
            method = selected[asset]["method"]
            prediction = forecast_selected_method(
                history,
                forecast_month,
                asset,
                method,
                horizon=1,
            )

            # Match the existing forward-forecast fallback behavior.
            if prediction is None:
                method = "expanding_mean"
                prediction = forecast_one(
                    history,
                    forecast_month,
                    method,
                    horizon=1,
                )

            if prediction is None:
                raise ValueError(
                    f"Could not produce a one-month forecast for {asset} "
                    f"at {rebalance_date.date()}."
                )

            rows.append({
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "information_date": information_date.strftime("%Y-%m-%d"),
                "asset": asset,
                "method": method,
                "month_ahead": 1,
                "forecast_month": str(forecast_month),
                "forecast_return": float(prediction),
            })

    return pd.DataFrame(rows)

def build_forward_forecast(
    validation: pd.DataFrame,
    monthly_history: pd.DataFrame,
    complete_months: list[pd.Period],
    selected: dict[str, dict],
    calendar,
) -> pd.DataFrame:
    """Forecast three months after the last complete validation month."""
    if not complete_months:
        raise ValueError("No complete validation month is available.")

    last_complete = max(complete_months)
    origin_month = last_complete + 1
    information_date = validation.index[
        validation.index.to_period("M") == last_complete
    ][-1]

    future_sessions = sessions_for_month(calendar, origin_month)
    if not len(future_sessions):
        raise ValueError(f"No exchange sessions found for {origin_month}.")
    rebalance_date = future_sessions[0]

    rows = []
    for asset in ASSETS:
        history = monthly_history[asset].dropna()
        history = history.loc[history.index < origin_month]

        selected_method = selected[asset]["method"]

        for horizon in range(1, HORIZON + 1):
            method = selected_method
            prediction = forecast_selected_method(
                history,
                origin_month,
                asset,
                method,
                horizon=horizon,
            )

            if prediction is None:
                method = "expanding_mean"
                prediction = forecast_one(
                    history,
                    origin_month,
                    method,
                    horizon=horizon,
                )

            if prediction is None:
                raise ValueError(
                    f"Could not produce a forward forecast for {asset}."
                )

            target_month = origin_month + horizon - 1
            rows.append({
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "information_date": information_date.strftime("%Y-%m-%d"),
                "asset": asset,
                "method": method,
                "month_ahead": horizon,
                "forecast_month": str(target_month),
                "forecast_return": float(prediction),
            })

    return pd.DataFrame(rows)

def add_selected_shrinkage_validation_forecasts(
    forecasts: pd.DataFrame,
    base_methods: dict[str, str],
    weight: float = SHRINKAGE_WEIGHT,
) -> pd.DataFrame:
    """Add a 50/50 blend of each selected forecast and expanding mean.

    The candidate is for comparison only; it does not participate in
    selecting the static method.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Shrinkage weight must be between 0 and 1.")

    keys = [
        "rebalance_date",
        "information_date",
        "month_ahead",
        "forecast_month",
    ]
    rows = []

    for asset in ASSETS:
        base_method = base_methods[asset]

        base = forecasts.loc[
            (forecasts["asset"] == asset)
            & (forecasts["method"] == base_method),
            keys + ["forecast_return", "realized_return"],
        ].rename(columns={
            "forecast_return": "selected_forecast",
            "realized_return": "selected_realized",
        })

        expanding = forecasts.loc[
            (forecasts["asset"] == asset)
            & (forecasts["method"] == "expanding_mean"),
            keys + ["forecast_return", "realized_return"],
        ].rename(columns={
            "forecast_return": "expanding_forecast",
            "realized_return": "expanding_realized",
        })

        joined = base.merge(
            expanding,
            on=keys,
            how="inner",
            validate="one_to_one",
        )

        if joined.empty:
            raise ValueError(
                f"No matching selected and expanding forecasts for {asset}."
            )

        # Both forecasts should be scored against the same realized return.
        observed = (
            joined["selected_realized"].notna()
            & joined["expanding_realized"].notna()
        )
        if observed.any() and not np.allclose(
            joined.loc[observed, "selected_realized"].to_numpy(dtype=float),
            joined.loc[observed, "expanding_realized"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Realized returns do not match for {asset}."
            )

        for _, item in joined.iterrows():
            prediction = (
                weight * float(item["selected_forecast"])
                + (1.0 - weight) * float(item["expanding_forecast"])
            )
            realized = item["selected_realized"]

            row = {key: item[key] for key in keys}
            row.update({
                "asset": asset,
                "method": SHRINKAGE_METHOD,
                "shrinkage_base_method": base_method,
                "forecast_return": prediction,
                "realized_return": realized,
                "error": (
                    prediction - float(realized)
                    if pd.notna(realized) else np.nan
                ),
            })
            rows.append(row)

    return pd.DataFrame(rows)
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    data_dir = find_split_dir()
    train = read_return_split(data_dir, "train")
    validation = read_return_split(data_dir, "validation")

    if train.index.max() >= validation.index.min():
        raise ValueError("Training and validation dates overlap or are misordered.")

    # Deliberately concatenate only train and validation. The test file is
    # never opened by this module.
    daily_returns = pd.concat([train, validation]).sort_index()
    if not daily_returns.index.is_unique:
        raise ValueError("Train/validation dates overlap.")

    long_prices = load_long_equity_bond_prices(validation.index.max())
    monthly_history = build_asset_monthly_history(daily_returns, long_prices)

    calendar = build_calendar(daily_returns, REQUESTED_REBALANCE_DATES)
    complete_months = complete_validation_months(validation, calendar)
    actuals = validation_actuals(validation, complete_months)

    static_methods_by_asset = {
        asset: (
            list(STATIC_METHODS) if asset in ("Equity", "Bond")
            else ["expanding_mean", *ROLLING_WINDOWS.keys()]
        )
        for asset in ASSETS
    }
    (
        walk_forward_selected,
        walk_forward_base_methods,
        walk_forward_training_scores,
        walk_forward_training_shrinkage,
    ) = select_walk_forward_methods_from_train(
        train,
        daily_returns,
        monthly_history,
        calendar,
        static_methods_by_asset,
    )

    walk_forward_long = build_walk_forward_validation_forecasts(
        validation,
        daily_returns,
        monthly_history,
        walk_forward_selected,
        walk_forward_base_methods,
        calendar,
    )

    walk_forward_wide_h1_h3 = selected_forecasts_to_wide(
        walk_forward_long,
        include_realized=False,
    )

    walk_forward_wide_h1 = walk_forward_wide_h1_h3.loc[
        walk_forward_wide_h1_h3["month_ahead"].astype(int) == 1
    ].copy()

    expected_rows = (
        HORIZON * walk_forward_wide_h1["rebalance_date"].nunique()
    )
    if len(walk_forward_wide_h1_h3) != expected_rows:
        raise ValueError(
            "Walk-forward output does not contain exactly one row per "
            "rebalance date and forecast horizon."
        )

    rows = add_validation_forecasts(
        daily_returns, validation, monthly_history, actuals
    )
    rows.extend(add_hmm_forecasts(validation, actuals, daily_returns))
    forecasts = pd.DataFrame(rows)

    shrinkage_rows = add_selected_shrinkage_validation_forecasts(
        forecasts,
        SHRINKAGE_BASE_METHODS,
    )

    forecasts = pd.concat(
        [forecasts, shrinkage_rows],
        ignore_index=True,
        sort=False,
    )

    if forecasts.empty:
        raise ValueError("No return forecasts were produced.")

    score_methods_by_asset = {
        asset: list(static_methods_by_asset[asset])
        for asset in ASSETS
    }
    if (forecasts["method"] == "hmm_regime").any():
        for asset in ASSETS:
            score_methods_by_asset[asset].append("hmm_regime")
    for asset in ASSETS:
        score_methods_by_asset[asset].append(SHRINKAGE_METHOD)

    scores = score_forecasts(forecasts, score_methods_by_asset)
    shrinkage_comparison = summarize_shrinkage_comparison(
        scores,
        SHRINKAGE_BASE_METHODS,
    )
    selected, combined_scores = select_static_methods(
        forecasts, static_methods_by_asset
    )
    # Keep the original static selections for transparency.
    base_selected = {
        asset: dict(choice)
        for asset, choice in selected.items()
    }

    # Confirm the tested BTC blend is still based on the selected 1-year mean.
    if base_selected["BTC"]["method"] != SHRINKAGE_BASE_METHODS["BTC"]:
        raise ValueError(
            "The BTC static selection differs from the base method used in "
            "the shrinkage comparison. Recheck the comparison before proceeding."
        )

    btc_blend = shrinkage_comparison.loc[
        (shrinkage_comparison["asset"] == "BTC")
        & (shrinkage_comparison["comparison"] == "shrinkage_50_50")
    ]

    if len(btc_blend) != 1:
        raise ValueError("Expected exactly one BTC shrinkage-comparison row.")

    btc_blend = btc_blend.iloc[0]

    # Use the already-tested 50/50 blend for BTC only.
    selected = {
        asset: dict(choice)
        for asset, choice in base_selected.items()
    }

    selected["BTC"] = {
        "method": SHRINKAGE_METHOD,
        "status": "selected_after_validation_shrinkage_comparison",
        "base_method": SHRINKAGE_BASE_METHODS["BTC"],
        "blend_weight_on_base_method": float(SHRINKAGE_WEIGHT),
        "blend_weight_on_expanding_mean": float(1.0 - SHRINKAGE_WEIGHT),
        "n_validation_origins_by_horizon": {
            str(h): int(btc_blend[f"n_scored_h{h}"])
            for h in range(1, HORIZON + 1)
        },
        "validation_rmse_by_horizon": {
            str(h): float(btc_blend[f"rmse_h{h}"])
            for h in range(1, HORIZON + 1)
        },
        "combined_validation_rmse_horizons_1_to_3": float(
            btc_blend["combined_rmse_h1_to_h3"]
        ),
        "combined_rmse_skill_vs_previous_method": float(
            btc_blend["rmse_skill_vs_selected_method"]
        ),
        "combined_rmse_skill_vs_expanding_mean": float(
            btc_blend["rmse_skill_vs_expanding_mean"]
        ),
    }
    # Validation-selected forecasts are useful for downstream diagnostics,
    # but selecting a method after observing validation errors means this
    # table is not an unbiased strategy-performance test.
    selected_val_parts = []
    for asset, choice in selected.items():
        part = forecasts.loc[
            (forecasts["asset"] == asset)
            & (forecasts["method"] == choice["method"])
        ]
        if not part.empty:
            selected_val_parts.append(part)
    selected_validation = (
        pd.concat(selected_val_parts, ignore_index=True)
        if selected_val_parts else pd.DataFrame()
    )

    forward_long = build_forward_forecast(
        validation,
        monthly_history,
        complete_months,
        selected,
        calendar,
    )
    requested_rebalance_dates = (
        list(REQUESTED_REBALANCE_DATES)
        if REQUESTED_REBALANCE_DATES
        else [default_rebalance_date(complete_months, calendar)]
    )

    optimizer_one_month_long = build_optimizer_one_month_forecasts(
        requested_rebalance_dates,
        validation,
        daily_returns,
        monthly_history,
        complete_months,
        selected,
        calendar,
    )
    optimizer_one_month_wide = selected_forecasts_to_wide(
        optimizer_one_month_long,
        include_realized=False,
    )
    # Source/history coverage makes the different start dates explicit.
    coverage_rows = []
    for asset in ASSETS:
        series = monthly_history[asset].dropna()
        coverage_rows.append({
            "asset": asset,
            "first_month": str(series.index.min()) if len(series) else "",
            "last_month": str(series.index.max()) if len(series) else "",
            "monthly_observations": int(len(series)),
            "pre_2015_observations": int(
                (series.index < pd.Period("2015-01", freq="M")).sum()
            ),
        })
    coverage = pd.DataFrame(coverage_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve the existing one-month file.
    walk_forward_wide_h1.to_csv(
        OUTPUT_DIR / "walk_forward_validation_expected_returns.csv",
        index=False,
    )

    # New file: horizons 1, 2 and 3 at each validation rebalance.
    walk_forward_wide_h1_h3.to_csv(
        OUTPUT_DIR / "walk_forward_validation_expected_returns_h1_h3.csv",
        index=False,
    )
    walk_forward_training_scores.to_csv(
        OUTPUT_DIR / "walk_forward_training_method_scores.csv",
        index=False,
    )
    walk_forward_training_shrinkage.to_csv(
        OUTPUT_DIR / "walk_forward_training_shrinkage_comparison.csv",
        index=False,
    )

    walk_forward_record = {
        "rebalance_schedule": (
            "First XNYS session of each validation month present in the "
            "validation input"
        ),
        "method_selection": (
            "Selected using training-only pseudo-out-of-sample forecasts; "
            "methods are held fixed through validation"
        ),
        "forecast_information": (
            "Each forecast uses monthly history strictly before its "
            "rebalance month"
        ),
        "monthly_return_convention": (
            "One-month compounded simple returns, decimals"
        ),
        "training_base_static_methods_by_asset": walk_forward_base_methods,
        "selected_by_asset": walk_forward_selected,
        "forecast_horizons_months": [1, 2, 3],
        "multi_horizon_file": (
            "walk_forward_validation_expected_returns_h1_h3.csv"
        ),
        "validation_rebalance_dates": (
            walk_forward_wide_h1["rebalance_date"].tolist()
        ),
        "warning": (
            "The candidate methods and 50/50 shrinkage idea were explored "
            "during prior model development. This is not a fresh independent "
            "holdout; portfolio_test.csv was not read."
        ),
    }
    with (
        OUTPUT_DIR / "walk_forward_training_method_selection.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(walk_forward_record, f, indent=2, allow_nan=False)
    forecasts.to_csv(OUTPUT_DIR / "validation_candidate_forecasts.csv",
                     index=False)
    scores.to_csv(OUTPUT_DIR / "validation_scores.csv", index=False)
    shrinkage_comparison.to_csv(
        OUTPUT_DIR / "shrinkage_validation_comparison.csv",
        index=False,
    )
    combined_scores.to_csv(OUTPUT_DIR / "combined_validation_scores.csv",index=False)
    coverage.to_csv(OUTPUT_DIR / "history_coverage.csv", index=False)

    if not selected_validation.empty:
        selected_wide = selected_forecasts_to_wide(
            selected_validation, include_realized=True
        )
        selected_wide.to_csv(
            OUTPUT_DIR / "validation_selected_expected_returns.csv",
            index=False,
        )

    forward_wide = selected_forecasts_to_wide(
        forward_long, include_realized=False
    )
    forward_wide.to_csv(
        OUTPUT_DIR / "selected_forward_expected_returns.csv",
        index=False,
    )
    optimizer_one_month_wide.to_csv(
        OUTPUT_DIR / "selected_one_month_optimizer_expected_returns.csv",
        index=False,
    )
    selection_record = {
        "primary_selection_metric": (
            "sqrt of the equally weighted average of MSE for horizons 1, 2 and 3"
        ),
        "minimum_complete_validation_months": MIN_VALIDATION_MONTHS,
        "monthly_return_convention": (
            "compounded simple returns, decimals; static forecasts repeated "
            "across months 1-3"
        ),
        "oos_snr_definition": (
            "RMS(forecast returns) / RMSE(forecast errors)"
        ),
        "selected_by_asset": selected,
        "base_static_selection_by_asset": base_selected,
        "warning": (
            "Validation-selected forecasts are post-selection diagnostics. "
            "Do not describe their validation portfolio performance as an "
            "unbiased holdout result. portfolio_test.csv was not read."
        ),
        "method_selection_cutoff": validation_selection_cutoff(
            validation, complete_months
        ).strftime("%Y-%m-%d"),
        "one_month_optimizer_rebalance_dates": [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in requested_rebalance_dates
        ],
    }
    with (OUTPUT_DIR / "selected_methods.json").open("w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2, allow_nan=False)

    print(f"Split data read from: {data_dir}")
    print("\nMONTHLY HISTORY COVERAGE:")
    print(coverage.to_string(index=False))

    print("\nONE-MONTH-AHEAD VALIDATION SCORES (returns are decimals):")
    display = scores.loc[scores["horizon_months"] == 1, [
        "asset", "method", "n_scored", "rmse", "mae",
        "bias_forecast_minus_realized",
        "oos_snr_rms_over_rmse",
        "rmse_skill_vs_expanding_common_dates",
        "full_validation_coverage_h1",
    ]]
    print(display.to_string(index=False))

    print("\nSELECTED STATIC METHODS:")
    print(json.dumps(selected, indent=2))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()