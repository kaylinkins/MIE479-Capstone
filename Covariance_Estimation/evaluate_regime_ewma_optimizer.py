"""Train-only selection and validation evaluation of regime-dependent EWMA.

Run:
    python Covariance_Estimation/evaluate_regime_ewma_optimizer.py

Reads training and validation returns only. It does not read portfolio_test.csv
or modify the HMM or MPO scripts.

The regime-dependent EWMA uses:
    lambda(origin, horizon) = 0.93 - delta * p_bear_month_horizon

where p_bear_month_horizon is the HMM's forecast average occupancy of the
higher-Equity-volatility state during that month. Delta is selected using
training-only one-month-ahead Gaussian NLL.

Validation comparisons:
  * Covariance forecasts: HMM, plain EWMA lambda=0.93, selected
    HMM-probability-gated EWMA, rolling 36m, rolling 60m,
    Ledoit-Wolf 60m, and constant-correlation shrinkage 60m.
  * Portfolio comparison: WP2.4 walk-forward expected returns held fixed
    across those covariance inputs, using Multi_Period_Baselines and the
    execution accounting used by mpo_baselines_walk_forward.py.
  * Separate native-HMM portfolio comparison: HMM expected returns plus
    HMM covariance. This is a whole-forecast-package comparison, not a
    covariance-only comparison.

Covariance NLL and Mahalanobis scores use a shared expanding historical mean
available at each rebalance. Portfolio optimization uses the supplied
expected-return forecasts. For covariance horizons 1-3, rolling, Ledoit-Wolf,
constant-correlation, and plain EWMA matrices are carried forward unchanged;
that is a persistence assumption. Regime-dependent EWMA may vary by horizon
because it uses the HMM's horizon-specific regime-occupancy probabilities.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs"

# WP2.4 walk-forward expected returns, selected using training data.
WP2_EXPECTED_RETURNS_FILE = (
    ROOT / "Return_Forecasting" / "outputs"
    / "walk_forward_validation_expected_returns_h1_h3.csv"
)

HMM_DIRECTORY = ROOT / "HMM_regime_forecasts"
HMM_OUTPUT_DIR = HMM_DIRECTORY / "hmm_outputs"
HMM_MODEL_FILE = HMM_DIRECTORY / "hmm_regime_model.py"
HMM_PROBABILITIES_FILE = HMM_OUTPUT_DIR / "regime_probabilities.csv"
HMM_COVARIANCE_FILE = HMM_OUTPUT_DIR / "covariances.csv"
HMM_EXPECTED_RETURNS_FILE = HMM_OUTPUT_DIR / "expected_returns.csv"

# The HMM script shown in the project writes expected_returns.csv. This
# optional file is used to cross-check those forecasts (or as a fallback if
# expected_returns.csv is absent). Change this path if yours is elsewhere.
HMM_CONDITIONAL_RETURNS_FILE = (
    ROOT / "Return_Forecasting" / "hmm_regime_conditional_returns.csv"
)

EWMA_COVARIANCE_FILE = (
    OUTPUT_DIR
    / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"
)

ASSETS = ("Equity", "Bond", "BTC")
HORIZONS = (1, 2, 3)

BASE_EWMA_LAMBDA = 0.93
# Includes delta=0, which is exactly plain EWMA lambda=0.93.
REGIME_DELTA_GRID = (0.00, 0.02, 0.04, 0.06)

# Match mpo_baselines_walk_forward.py.
BTC_CAP = 0.20
COST_RATE = 0.001
ALPHA = 0.95
TRADING_DAYS = 252
RISK_FREE = 0.0

# Numerical check that the supplied plain-EWMA export corresponds to the
# configured 60-month, lambda=0.93 estimator.
EWMA_EXPORT_RTOL = 1e-5
EWMA_EXPORT_ATOL = 1e-10


# ---------------------------------------------------------------------
# Imports of the existing diagnostics, HMM, and optimizer libraries
# ---------------------------------------------------------------------

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import covariance_diagnostics as cd  # noqa: E402


def find_optimizer_directory() -> Path:
    """Find the directory containing Multi_Period_Baselines.py."""
    configured = os.environ.get("MPO_OPTIMIZER_DIR")
    if configured:
        folder = Path(configured).expanduser().resolve()
        if (folder / "Multi_Period_Baselines.py").exists():
            return folder
        raise FileNotFoundError(
            "MPO_OPTIMIZER_DIR was set, but it does not contain "
            f"Multi_Period_Baselines.py: {folder}"
        )

    likely = [
        ROOT / "Portfolio Optimization",
        ROOT / "Portfolio_Optimization",
        ROOT / "PortfolioOptimization",
    ]
    for folder in likely:
        if (folder / "Multi_Period_Baselines.py").exists():
            return folder

    ignored = {
        ".git", ".venv", "venv", "site-packages", "__pycache__",
        "node_modules",
    }
    matches = []
    for path in ROOT.rglob("Multi_Period_Baselines.py"):
        if any(part in ignored for part in path.parts):
            continue
        matches.append(path.parent.resolve())

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            "Found multiple Multi_Period_Baselines.py files. Set the "
            "MPO_OPTIMIZER_DIR environment variable to the intended folder. "
            f"Matches: {matches}"
        )

    raise FileNotFoundError(
        "Could not find Multi_Period_Baselines.py. Set MPO_OPTIMIZER_DIR "
        "to the folder containing it."
    )


OPTIMIZER_DIRECTORY = find_optimizer_directory()
if str(OPTIMIZER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIRECTORY))

CVAR_DIRECTORY = ROOT / "Distribution Modelling (Scenario Generation)"
if str(CVAR_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CVAR_DIRECTORY))

mb = importlib.import_module("Multi_Period_Baselines")
mc = importlib.import_module("cvar_mpo")


def load_hmm_module():
    """Load the teammate's HMM module without running its main function."""
    if not HMM_MODEL_FILE.exists():
        raise FileNotFoundError(f"HMM model script not found: {HMM_MODEL_FILE}")

    spec = importlib.util.spec_from_file_location(
        "project_hmm_regime_model_for_covariance_evaluation",
        HMM_MODEL_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HMM module from {HMM_MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hmm_model = load_hmm_module()

if tuple(cd.ASSETS) != ASSETS:
    raise ValueError(f"Covariance diagnostics asset order differs: {cd.ASSETS}")
if tuple(mb.ASSETS) != ASSETS:
    raise ValueError(f"MPO asset order differs: {mb.ASSETS}")
if tuple(hmm_model.ASSETS_EXPECTED) != ASSETS:
    raise ValueError(f"HMM asset order differs: {hmm_model.ASSETS_EXPECTED}")

HMM_ESTIMATOR = "hmm_regime"
ROLLING_36_ESTIMATOR = cd.ROLLING_36_ESTIMATOR
ROLLING_60_ESTIMATOR = cd.ROLLING_60_ESTIMATOR
LEDOIT_WOLF_ESTIMATOR = cd.LEDOIT_WOLF_ESTIMATOR
CONSTANT_CORRELATION_ESTIMATOR = cd.CONSTANT_CORRELATION_ESTIMATOR
PLAIN_EWMA_ESTIMATOR = cd.ewma_estimator_name(BASE_EWMA_LAMBDA)


def regime_ewma_estimator_name(delta: float) -> str:
    return (
        f"regime_ewma_cov_60m_base_{BASE_EWMA_LAMBDA:.2f}_"
        f"delta_{delta:.2f}"
    ).replace(".", "_")


# ---------------------------------------------------------------------
# General input helpers
# ---------------------------------------------------------------------

def parse_timestamp(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def parse_month(value) -> pd.Period:
    return pd.Period(str(value), freq="M")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def check_forecast_key(
    rebalance_date: pd.Timestamp,
    month_ahead: int,
    forecast_period: pd.Period,
    source: str,
) -> None:
    expected_period = rebalance_date.to_period("M") + (month_ahead - 1)
    if forecast_period != expected_period:
        raise ValueError(
            f"{source}: rebalance {rebalance_date.date()}, "
            f"horizon {month_ahead} has forecast month {forecast_period}; "
            f"expected {expected_period}."
        )


def read_covariance_forecasts(path: Path, source: str) -> dict:
    """Read HMM-format monthly covariance matrices for horizons 1-3."""
    require_file(path)
    frame = pd.read_csv(path)

    required = {
        "rebalance_date", "month_ahead", "forecast_month", "asset", *ASSETS
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} file is missing columns: {sorted(missing)}")

    frame["rebalance_date"] = pd.DatetimeIndex(
        pd.to_datetime(frame["rebalance_date"], errors="raise")
    ).normalize()
    frame["month_ahead"] = pd.to_numeric(
        frame["month_ahead"], errors="raise"
    ).astype(int)
    frame["asset"] = frame["asset"].astype(str)
    frame = frame.loc[frame["month_ahead"].isin(HORIZONS)].copy()
    frame["forecast_period"] = frame["forecast_month"].map(parse_month)

    if frame.duplicated(["rebalance_date", "month_ahead", "asset"]).any():
        raise ValueError(
            f"{source} file has duplicate rebalance/horizon/asset rows."
        )

    result = {}
    for (rebalance_date, month_ahead), group in frame.groupby(
        ["rebalance_date", "month_ahead"], sort=True
    ):
        periods = group["forecast_period"].unique()
        if len(periods) != 1:
            raise ValueError(
                f"{source}: inconsistent forecast_month within a matrix."
            )
        forecast_period = periods[0]
        check_forecast_key(
            rebalance_date, int(month_ahead), forecast_period, source
        )

        if (
            len(group) != len(ASSETS)
            or set(group["asset"]) != set(ASSETS)
            or group["asset"].duplicated().any()
        ):
            raise ValueError(
                f"{source}: expected exactly one matrix row for each asset "
                f"at {rebalance_date.date()}, horizon {month_ahead}."
            )

        ordered = group.set_index("asset").loc[list(ASSETS), list(ASSETS)]
        matrix = ordered.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{source}: non-finite covariance at "
                f"{rebalance_date.date()}, horizon {month_ahead}."
            )

        key = (rebalance_date, int(month_ahead))
        result[key] = {
            "matrix": matrix,
            "forecast_period": forecast_period,
        }

    return result


def read_mean_forecasts(
    path: Path,
    *,
    source: str,
    rename_columns: dict | None = None,
    validation_only: bool = False,
) -> dict:
    """Read monthly expected-return forecasts into an origin/horizon map."""
    require_file(path)
    frame = pd.read_csv(path)

    if rename_columns:
        frame = frame.rename(columns=rename_columns)

    required = {
        "rebalance_date", "month_ahead", "forecast_month", *ASSETS
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} file is missing columns: {sorted(missing)}")

    if validation_only and "origin_type" in frame.columns:
        frame = frame.loc[
            frame["origin_type"].astype(str).str.lower() == "validation"
        ].copy()

    frame["rebalance_date"] = pd.DatetimeIndex(
        pd.to_datetime(frame["rebalance_date"], errors="raise")
    ).normalize()
    frame["month_ahead"] = pd.to_numeric(
        frame["month_ahead"], errors="raise"
    ).astype(int)
    frame = frame.loc[frame["month_ahead"].isin(HORIZONS)].copy()
    frame["forecast_period"] = frame["forecast_month"].map(parse_month)

    if frame.duplicated(["rebalance_date", "month_ahead"]).any():
        raise ValueError(f"{source} has duplicate rebalance/horizon rows.")

    result = {}
    for _, row in frame.iterrows():
        rebalance_date = row["rebalance_date"]
        month_ahead = int(row["month_ahead"])
        forecast_period = row["forecast_period"]
        check_forecast_key(
            rebalance_date, month_ahead, forecast_period, source
        )

        mean = pd.to_numeric(row.loc[list(ASSETS)], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(mean).all() or (mean <= -1.0).any():
            raise ValueError(
                f"{source}: invalid expected returns at "
                f"{rebalance_date.date()}, horizon {month_ahead}."
            )

        information_date = None
        if "information_date" in frame.columns and pd.notna(
            row.get("information_date")
        ):
            information_date = parse_timestamp(row["information_date"])
            if information_date >= rebalance_date:
                raise ValueError(
                    f"{source}: information_date is not strictly before "
                    f"rebalance_date {rebalance_date.date()}."
                )

        result[(rebalance_date, month_ahead)] = {
            "mean": mean,
            "forecast_period": forecast_period,
            "information_date": information_date,
        }

    return result


def read_regime_probabilities(path: Path) -> dict:
    """Read HMM average forecast occupancy probabilities for validation."""
    require_file(path)
    frame = pd.read_csv(path)

    required = {
        "rebalance_date", "information_date",
        *[f"p_bear_month{h}" for h in HORIZONS],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"HMM regime-probability file is missing columns: {sorted(missing)}"
        )

    frame["rebalance_date"] = pd.DatetimeIndex(
        pd.to_datetime(frame["rebalance_date"], errors="raise")
    ).normalize()
    frame["information_date"] = pd.DatetimeIndex(
        pd.to_datetime(frame["information_date"], errors="raise")
    ).normalize()

    if frame["rebalance_date"].duplicated().any():
        raise ValueError("HMM regime-probability file has duplicate origins.")

    result = {}
    for _, row in frame.iterrows():
        origin = row["rebalance_date"]
        information_date = row["information_date"]
        if information_date >= origin:
            raise ValueError(
                f"HMM probabilities at {origin.date()} use a non-prior "
                "information_date."
            )

        probabilities = {}
        for h in HORIZONS:
            probability = float(row[f"p_bear_month{h}"])
            if not np.isfinite(probability) or not -1e-10 <= probability <= 1 + 1e-10:
                raise ValueError(
                    f"Invalid p_bear_month{h} at {origin.date()}: "
                    f"{probability}"
                )
            probabilities[h] = float(np.clip(probability, 0.0, 1.0))

        result[origin] = {
            "information_date": information_date,
            "probabilities": probabilities,
        }

    return result


def find_optional_conditional_returns_file() -> Path | None:
    candidates = [
        HMM_CONDITIONAL_RETURNS_FILE,
        HMM_DIRECTORY / "hmm_regime_conditional_returns.csv",
        ROOT / "hmm_regime_conditional_returns.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_conditional_probability_weighted_returns(path: Path) -> dict:
    """Read the HMM's probability-weighted conditional means, if available."""
    rename = {
        f"{asset}_probability_weighted": asset
        for asset in ASSETS
    }
    return read_mean_forecasts(
        path,
        source="HMM conditional probability-weighted returns",
        rename_columns=rename,
        validation_only=True,
    )


def compare_hmm_mean_files(
    hmm_means: dict,
    conditional_means: dict | None,
) -> pd.DataFrame:
    """Cross-check HMM expected_returns.csv against weighted conditional means."""
    columns = [
        "rebalance_date", "month_ahead", "forecast_month", "asset",
        "hmm_expected_return", "conditional_probability_weighted_return",
        "absolute_difference",
    ]
    if conditional_means is None:
        return pd.DataFrame(columns=columns)

    rows = []
    common_keys = sorted(set(hmm_means) & set(conditional_means))
    for key in common_keys:
        for i, asset in enumerate(ASSETS):
            first = float(hmm_means[key]["mean"][i])
            second = float(conditional_means[key]["mean"][i])
            rows.append({
                "rebalance_date": key[0].strftime("%Y-%m-%d"),
                "month_ahead": key[1],
                "forecast_month": str(hmm_means[key]["forecast_period"]),
                "asset": asset,
                "hmm_expected_return": first,
                "conditional_probability_weighted_return": second,
                "absolute_difference": abs(first - second),
            })

    return pd.DataFrame(rows, columns=columns)


def require_forecast_coverage(
    mapping: dict,
    origins: list[pd.Timestamp],
    source: str,
) -> None:
    missing = [
        (origin, h)
        for origin in origins
        for h in HORIZONS
        if (origin, h) not in mapping
    ]
    if missing:
        preview = [
            (date.strftime("%Y-%m-%d"), h) for date, h in missing[:10]
        ]
        raise ValueError(
            f"{source} does not cover every validation origin and horizon. "
            f"Missing first entries: {preview}"
        )


# ---------------------------------------------------------------------
# Training-only HMM-regime EWMA selection
# ---------------------------------------------------------------------

def monthly_window_before(
    monthly_returns: pd.DataFrame,
    origin_month: pd.Period,
    window_months: int,
) -> np.ndarray | None:
    """Return the consecutive complete monthly window before an origin."""
    required_months = pd.period_range(
        end=origin_month - 1,
        periods=window_months,
        freq="M",
    )
    window = monthly_returns.reindex(required_months)
    if len(window) != window_months or window.isna().to_numpy().any():
        return None
    return window.to_numpy(dtype=float)


def score_one_covariance(
    *,
    metadata: dict,
    estimator: str,
    matrix: np.ndarray,
    n_estimation_months: float,
    common_mean: np.ndarray,
    realized_return: np.ndarray,
    p_bear: float = np.nan,
    ewma_lambda: float = np.nan,
) -> dict:
    """Use the existing diagnostic's matrix checks and Gaussian scoring."""
    structure, symmetric, is_pd = cd.inspect_matrix(matrix)
    row = cd.score_covariance(
        metadata=metadata,
        estimator=estimator,
        matrix=matrix,
        symmetric_matrix=symmetric,
        is_positive_definite=is_pd,
        n_estimation_months=n_estimation_months,
        common_mean=common_mean,
        realized_return=realized_return,
    )
    row.update(structure)
    row["p_bear_forecast"] = p_bear
    row["ewma_lambda"] = ewma_lambda
    return row


def build_training_covariance_scores(
    train_daily: pd.DataFrame,
    train_monthly: pd.DataFrame,
    calendar,
) -> pd.DataFrame:
    """Generate training-only one-month-ahead scores at rolling origins.

    At each eligible origin:
      * HMM parameters and filtered state probabilities use only prior
        training daily returns.
      * Actual returns are from a training month.
      * Every covariance candidate is scored about the same expanding-mean
        center.
    """
    rows = []
    eligible_origins = 0

    for origin_index, target_month in enumerate(train_monthly.index):
        sessions = cd.sessions_for_month(calendar, target_month)
        if not len(sessions):
            continue

        rebalance_date = sessions[0]
        values_60 = monthly_window_before(
            train_monthly, target_month, cd.ROLLING_WINDOW_MONTHS
        )
        if values_60 is None:
            continue

        values_36 = monthly_window_before(
            train_monthly, target_month, cd.SHORT_ROLLING_WINDOW_MONTHS
        )
        if values_36 is None:
            continue

        prior_months = train_monthly.loc[
            train_monthly.index < target_month
        ]
        history = train_daily.loc[train_daily.index < rebalance_date]
        if prior_months.empty or history.empty:
            continue
        if rebalance_date not in train_daily.index:
            raise ValueError(
                f"Training origin {rebalance_date.date()} is not present "
                "in the daily training data."
            )

        information_date = history.index[-1]
        if information_date >= rebalance_date:
            raise ValueError("Training HMM fit would contain look-ahead data.")

        # Import and call the exact HMM fitting/filtering functions used by the
        # teammate's script. The HMM code is not modified or run as a backtest.
        model = hmm_model.fit_hmm(
            history.to_numpy(dtype=float),
            seed=origin_index,
        )
        p_asof = hmm_model.filtered_state_probabilities(
            model, history.to_numpy(dtype=float)
        )[-1]

        forecast_sessions = hmm_model.forecast_sessions(
            calendar, rebalance_date
        )
        day_counts = [len(block) for block in forecast_sessions]
        if not day_counts or day_counts[0] < 1:
            raise ValueError(
                f"Could not get HMM forecast sessions for "
                f"{rebalance_date.date()}."
            )

        p_bear_by_month, _, hmm_covariances = (
            hmm_model.forecast_moments_exact(model, p_asof, day_counts)
        )
        p_bear = float(p_bear_by_month[0])

        common_mean = prior_months.loc[:, list(ASSETS)].mean().to_numpy(float)
        realized = train_monthly.loc[target_month, list(ASSETS)].to_numpy(float)

        candidates = [
            (
                ROLLING_36_ESTIMATOR,
                np.cov(values_36, rowvar=False, ddof=1),
                float(cd.SHORT_ROLLING_WINDOW_MONTHS),
                np.nan,
            ),
            (
                ROLLING_60_ESTIMATOR,
                np.cov(values_60, rowvar=False, ddof=1),
                float(cd.ROLLING_WINDOW_MONTHS),
                np.nan,
            ),
        ]

        lw_covariance, _ = cd.ledoit_wolf_correlation_covariance(values_60)
        candidates.append((
            LEDOIT_WOLF_ESTIMATOR,
            lw_covariance,
            float(cd.ROLLING_WINDOW_MONTHS),
            np.nan,
        ))

        constant_covariance = cd.constant_correlation_covariance(values_60)
        candidates.append((
            CONSTANT_CORRELATION_ESTIMATOR,
            constant_covariance,
            float(cd.ROLLING_WINDOW_MONTHS),
            np.nan,
        ))

        plain_covariance, _ = cd.ewma_sample_covariance(
            values_60, decay=BASE_EWMA_LAMBDA
        )
        candidates.append((
            PLAIN_EWMA_ESTIMATOR,
            plain_covariance,
            float(cd.ROLLING_WINDOW_MONTHS),
            BASE_EWMA_LAMBDA,
        ))

        for delta in REGIME_DELTA_GRID:
            decay = BASE_EWMA_LAMBDA - delta * p_bear
            if not 0.0 < decay < 1.0:
                raise ValueError(
                    f"Regime EWMA lambda out of range: {decay}."
                )
            covariance, _ = cd.ewma_sample_covariance(
                values_60, decay=decay
            )
            candidates.append((
                regime_ewma_estimator_name(delta),
                covariance,
                float(cd.ROLLING_WINDOW_MONTHS),
                decay,
            ))

        candidates.append((
            HMM_ESTIMATOR,
            hmm_covariances[0],
            float(len(prior_months)),
            np.nan,
        ))

        metadata = {
            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
            "information_date": information_date.strftime("%Y-%m-%d"),
            "forecast_month": str(target_month),
            "month_ahead": 1,
            "n_prior_complete_months": int(len(prior_months)),
        }

        for estimator, matrix, estimation_months, decay in candidates:
            rows.append(score_one_covariance(
                metadata=metadata,
                estimator=estimator,
                matrix=matrix,
                n_estimation_months=estimation_months,
                common_mean=common_mean,
                realized_return=realized,
                p_bear=p_bear,
                ewma_lambda=decay,
            ))

        eligible_origins += 1
        if eligible_origins % 4 == 0:
            print(
                f"Training-only HMM/EWMA origins fitted: "
                f"{eligible_origins}",
                flush=True,
            )

    if not rows:
        raise ValueError(
            "No training origins had 60 prior complete monthly returns. "
            "Cannot select the regime-EWMA response using training data."
        )

    return pd.DataFrame(rows)


def select_regime_delta(
    training_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Select delta by one-month-ahead training NLL on common origins."""
    candidates = [
        regime_ewma_estimator_name(delta)
        for delta in REGIME_DELTA_GRID
    ]
    nll_wide = training_scores.pivot(
        index=["rebalance_date", "forecast_month"],
        columns="estimator",
        values="multivariate_gaussian_nll_common_mean",
    )

    missing = set(candidates) - set(nll_wide.columns)
    if missing:
        raise ValueError(
            f"Training scores lack regime-EWMA candidates: {sorted(missing)}"
        )

    common_nll = nll_wide[candidates].dropna()
    if common_nll.empty:
        raise ValueError(
            "No common valid training NLL origins for all regime-EWMA "
            "delta candidates."
        )

    mahalanobis_wide = training_scores.pivot(
        index=["rebalance_date", "forecast_month"],
        columns="estimator",
        values="mahalanobis_squared_common_mean",
    )
    common_mahalanobis = mahalanobis_wide[candidates].reindex(
        common_nll.index
    )

    rows = []
    for delta, estimator in zip(REGIME_DELTA_GRID, candidates):
        rows.append({
            "delta": float(delta),
            "estimator": estimator,
            "n_common_training_origins": int(len(common_nll)),
            "mean_training_multivariate_nll": float(
                common_nll[estimator].mean()
            ),
            "mean_training_mahalanobis_squared": float(
                common_mahalanobis[estimator].mean()
            ),
            "lambda_definition": (
                f"{BASE_EWMA_LAMBDA:.2f} - delta * p_bear_month1"
            ),
            "selection_criterion": (
                "Lowest one-month-ahead multivariate Gaussian NLL on common "
                "training-only rolling-origin forecasts."
            ),
        })

    selection = pd.DataFrame(rows)
    selection["training_nll_rank"] = (
        selection["mean_training_multivariate_nll"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    selected_position = int(np.argmin(
        selection["mean_training_multivariate_nll"].to_numpy(float)
    ))
    selection["selected_by_training_nll"] = False
    selection.loc[selected_position, "selected_by_training_nll"] = True
    selected_delta = float(selection.loc[selected_position, "delta"])

    return selection, selected_delta


# ---------------------------------------------------------------------
# Validation covariance forecasts and scoring
# ---------------------------------------------------------------------

def build_validation_covariances(
    *,
    origins: list[pd.Timestamp],
    monthly_history: pd.DataFrame,
    hmm_covariances: dict,
    plain_ewma_covariances: dict,
    regime_probabilities: dict,
    selected_delta: float,
) -> tuple[dict, dict, pd.DataFrame]:
    """Build all covariance inputs for validation scoring and MPO."""
    selected_regime_estimator = regime_ewma_estimator_name(selected_delta)
    covariance_forecasts = {}
    covariance_info = {}
    lambda_rows = []
    ewma_export_check_rows = []

    for origin in origins:
        origin_month = origin.to_period("M")
        values_36 = monthly_window_before(
            monthly_history, origin_month, cd.SHORT_ROLLING_WINDOW_MONTHS
        )
        values_60 = monthly_window_before(
            monthly_history, origin_month, cd.ROLLING_WINDOW_MONTHS
        )
        if values_36 is None or values_60 is None:
            raise ValueError(
                f"Insufficient complete monthly history at "
                f"{origin.date()} for the configured rolling covariances."
            )
        if origin not in regime_probabilities:
            raise ValueError(
                f"Missing HMM regime probabilities for {origin.date()}."
            )

        common_covariances = {
            ROLLING_36_ESTIMATOR: np.cov(
                values_36, rowvar=False, ddof=1
            ),
            ROLLING_60_ESTIMATOR: np.cov(
                values_60, rowvar=False, ddof=1
            ),
            LEDOIT_WOLF_ESTIMATOR:
                cd.ledoit_wolf_correlation_covariance(values_60)[0],
            CONSTANT_CORRELATION_ESTIMATOR:
                cd.constant_correlation_covariance(values_60),
        }
        local_plain_ewma, _ = cd.ewma_sample_covariance(
            values_60, decay=BASE_EWMA_LAMBDA
        )

        for h in HORIZONS:
            expected_period = origin_month + (h - 1)
            hmm_key = (origin, h)

            if hmm_key not in hmm_covariances:
                raise ValueError(
                    f"Missing HMM covariance for {origin.date()}, horizon {h}."
                )
            if hmm_key not in plain_ewma_covariances:
                raise ValueError(
                    f"Missing EWMA covariance for {origin.date()}, horizon {h}."
                )

            hmm_record = hmm_covariances[hmm_key]
            ewma_record = plain_ewma_covariances[hmm_key]
            if hmm_record["forecast_period"] != expected_period:
                raise ValueError("HMM forecast month does not match origin/horizon.")
            if ewma_record["forecast_period"] != expected_period:
                raise ValueError("EWMA forecast month does not match origin/horizon.")

            exported_plain_ewma = ewma_record["matrix"]
            max_difference = float(np.max(np.abs(
                exported_plain_ewma - local_plain_ewma
            )))
            matches_local = bool(np.allclose(
                exported_plain_ewma,
                local_plain_ewma,
                rtol=EWMA_EXPORT_RTOL,
                atol=EWMA_EXPORT_ATOL,
            ))
            ewma_export_check_rows.append({
                "rebalance_date": origin.strftime("%Y-%m-%d"),
                "month_ahead": h,
                "forecast_month": str(expected_period),
                "max_abs_difference_vs_local_lambda_0_93": max_difference,
                "matches_local_60m_lambda_0_93": matches_local,
            })
            if not matches_local:
                raise ValueError(
                    "The exported EWMA covariance does not match a local "
                    "lambda=0.93 estimate from the trailing 60 complete "
                    f"months at {origin.date()}, horizon {h}. Check that "
                    "the EWMA export uses the expected data/window/method. "
                    f"Maximum absolute difference: {max_difference:g}"
                )

            matrices = {
                HMM_ESTIMATOR: hmm_record["matrix"],
                ROLLING_36_ESTIMATOR:
                    common_covariances[ROLLING_36_ESTIMATOR],
                ROLLING_60_ESTIMATOR:
                    common_covariances[ROLLING_60_ESTIMATOR],
                LEDOIT_WOLF_ESTIMATOR:
                    common_covariances[LEDOIT_WOLF_ESTIMATOR],
                CONSTANT_CORRELATION_ESTIMATOR:
                    common_covariances[CONSTANT_CORRELATION_ESTIMATOR],
                PLAIN_EWMA_ESTIMATOR: exported_plain_ewma,
            }
            info = {
                HMM_ESTIMATOR: float(len(
                    monthly_history.loc[monthly_history.index < origin_month]
                )),
                ROLLING_36_ESTIMATOR:
                    float(cd.SHORT_ROLLING_WINDOW_MONTHS),
                ROLLING_60_ESTIMATOR:
                    float(cd.ROLLING_WINDOW_MONTHS),
                LEDOIT_WOLF_ESTIMATOR:
                    float(cd.ROLLING_WINDOW_MONTHS),
                CONSTANT_CORRELATION_ESTIMATOR:
                    float(cd.ROLLING_WINDOW_MONTHS),
                PLAIN_EWMA_ESTIMATOR:
                    float(cd.ROLLING_WINDOW_MONTHS),
            }

            p_bear = regime_probabilities[origin]["probabilities"][h]
            decay = BASE_EWMA_LAMBDA - selected_delta * p_bear
            if not 0.0 < decay < 1.0:
                raise ValueError(
                    f"Invalid regime-EWMA lambda {decay} at "
                    f"{origin.date()}, horizon {h}."
                )

            regime_covariance, _ = cd.ewma_sample_covariance(
                values_60, decay=decay
            )
            matrices[selected_regime_estimator] = regime_covariance
            info[selected_regime_estimator] = float(cd.ROLLING_WINDOW_MONTHS)

            for estimator, matrix in matrices.items():
                covariance_forecasts[(origin, h, estimator)] = matrix
                covariance_info[(origin, h, estimator)] = {
                    "forecast_period": expected_period,
                    "n_estimation_months": info[estimator],
                    "p_bear_forecast": p_bear,
                    "ewma_lambda": (
                        decay if estimator == selected_regime_estimator
                        else BASE_EWMA_LAMBDA
                        if estimator == PLAIN_EWMA_ESTIMATOR
                        else np.nan
                    ),
                }

            lambda_rows.append({
                "rebalance_date": origin.strftime("%Y-%m-%d"),
                "month_ahead": h,
                "forecast_month": str(expected_period),
                "p_bear_month": p_bear,
                "selected_delta": selected_delta,
                "regime_ewma_lambda": decay,
                "base_ewma_lambda": BASE_EWMA_LAMBDA,
                "selected_regime_ewma_estimator": selected_regime_estimator,
            })

    return (
        covariance_forecasts,
        covariance_info,
        pd.DataFrame(lambda_rows),
        pd.DataFrame(ewma_export_check_rows),
    )


def score_validation_covariances(
    *,
    origins: list[pd.Timestamp],
    daily: pd.DataFrame,
    monthly_history: pd.DataFrame,
    validation_actuals: pd.DataFrame,
    covariance_forecasts: dict,
    covariance_info: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score available validation target months by forecast horizon."""
    rows = []
    exclusions = []

    for origin in origins:
        origin_month = origin.to_period("M")
        prior_months = monthly_history.loc[
            monthly_history.index < origin_month
        ]
        if prior_months.empty:
            raise ValueError(
                f"No prior complete monthly returns at {origin.date()}."
            )
        common_mean = prior_months.loc[:, list(ASSETS)].mean().to_numpy(float)

        prior_daily = daily.index[daily.index < origin]
        if not len(prior_daily):
            raise ValueError(f"No daily information date before {origin.date()}.")
        information_date = prior_daily[-1]

        for h in HORIZONS:
            target_period = origin_month + (h - 1)
            if target_period not in validation_actuals.index:
                exclusions.append({
                    "rebalance_date": origin.strftime("%Y-%m-%d"),
                    "month_ahead": h,
                    "forecast_month": str(target_period),
                    "reason": (
                        "Target month is not a complete observed validation "
                        "month; covariance forecast is retained for MPO but "
                        "cannot be scored."
                    ),
                })
                continue

            realized = validation_actuals.loc[
                target_period, list(ASSETS)
            ].to_numpy(float)

            estimators = sorted({
                estimator
                for (candidate_origin, candidate_h, estimator)
                in covariance_forecasts
                if candidate_origin == origin and candidate_h == h
            })

            metadata = {
                "rebalance_date": origin.strftime("%Y-%m-%d"),
                "information_date": information_date.strftime("%Y-%m-%d"),
                "forecast_month": str(target_period),
                "month_ahead": h,
                "n_prior_complete_months": int(len(prior_months)),
                "score_center": "expanding_mean_available_at_rebalance",
            }

            for estimator in estimators:
                key = (origin, h, estimator)
                detail = covariance_info[key]
                rows.append(score_one_covariance(
                    metadata=metadata,
                    estimator=estimator,
                    matrix=covariance_forecasts[key],
                    n_estimation_months=detail["n_estimation_months"],
                    common_mean=common_mean,
                    realized_return=realized,
                    p_bear=detail["p_bear_forecast"],
                    ewma_lambda=detail["ewma_lambda"],
                ))

    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def summarize_validation_covariance_scores(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()

    return (
        scores.groupby(["estimator", "month_ahead"], as_index=False)
        .agg(
            n_forecasts=("multivariate_gaussian_nll_common_mean", "count"),
            mean_multivariate_gaussian_nll=(
                "multivariate_gaussian_nll_common_mean", "mean"
            ),
            mean_mahalanobis_squared=(
                "mahalanobis_squared_common_mean", "mean"
            ),
            n_positive_definite=("matrix_positive_definite", "sum"),
            mean_regime_lambda=("ewma_lambda", "mean"),
        )
        .sort_values(["month_ahead", "mean_multivariate_gaussian_nll"])
    )


def paired_covariance_comparisons(
    scores: pd.DataFrame,
    selected_regime_estimator: str,
) -> pd.DataFrame:
    """Paired comparisons at the same origin and horizon."""
    pairs = [
        (HMM_ESTIMATOR, PLAIN_EWMA_ESTIMATOR, "HMM minus plain EWMA"),
        (HMM_ESTIMATOR, selected_regime_estimator, "HMM minus regime EWMA"),
        (HMM_ESTIMATOR, ROLLING_36_ESTIMATOR, "HMM minus rolling 36m"),
        (HMM_ESTIMATOR, ROLLING_60_ESTIMATOR, "HMM minus rolling 60m"),
        (HMM_ESTIMATOR, LEDOIT_WOLF_ESTIMATOR, "HMM minus Ledoit-Wolf"),
        (HMM_ESTIMATOR, CONSTANT_CORRELATION_ESTIMATOR,
         "HMM minus constant-correlation"),
        (selected_regime_estimator, PLAIN_EWMA_ESTIMATOR,
         "regime EWMA minus plain EWMA"),
    ]
    metrics = [
        ("multivariate_gaussian_nll_common_mean", "NLL"),
        ("mahalanobis_squared_common_mean", "Mahalanobis squared"),
    ]

    rows = []
    for h in HORIZONS:
        horizon_scores = scores.loc[scores["month_ahead"] == h]
        for first, second, pair_name in pairs:
            for metric, metric_label in metrics:
                wide = horizon_scores.pivot(
                    index=["rebalance_date", "forecast_month"],
                    columns="estimator",
                    values=metric,
                )
                if not {first, second}.issubset(wide.columns):
                    rows.append({
                        "month_ahead": h,
                        "comparison": pair_name,
                        "metric": metric_label,
                        "n_paired": 0,
                        "mean_first_minus_second": np.nan,
                        "fraction_first_lower": np.nan,
                    })
                    continue

                paired = wide[[first, second]].dropna()
                difference = paired[first] - paired[second]
                rows.append({
                    "month_ahead": h,
                    "comparison": pair_name,
                    "metric": metric_label,
                    "n_paired": int(len(paired)),
                    "mean_first_minus_second": (
                        float(difference.mean()) if len(difference) else np.nan
                    ),
                    "fraction_first_lower": (
                        float((difference < 0.0).mean())
                        if len(difference) else np.nan
                    ),
                    "interpretation": (
                        "For NLL and Mahalanobis squared, lower is better; "
                        "negative first-minus-second favors the first method."
                    ),
                })

    return pd.DataFrame(rows)


def export_covariance_long_format(
    covariance_forecasts: dict,
    estimator: str,
) -> pd.DataFrame:
    """Export a selected covariance series in HMM-compatible long format."""
    rows = []
    selected = [
        (key, matrix)
        for key, matrix in covariance_forecasts.items()
        if key[2] == estimator
    ]

    for (rebalance_date, month_ahead, _), matrix in sorted(
        selected, key=lambda item: (item[0][0], item[0][1])
    ):
        forecast_period = rebalance_date.to_period("M") + (month_ahead - 1)
        for i, row_asset in enumerate(ASSETS):
            rows.append({
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "month_ahead": month_ahead,
                "forecast_month": str(forecast_period),
                "asset": row_asset,
                **{
                    column_asset: float(matrix[i, j])
                    for j, column_asset in enumerate(ASSETS)
                },
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MPO portfolio comparison
# ---------------------------------------------------------------------

def portfolio_performance(returns: pd.Series) -> dict:
    values = returns.to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("Need at least two daily returns for performance metrics.")

    annualized_return = float(values.mean() * TRADING_DAYS)
    annualized_volatility = float(
        values.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )
    if annualized_volatility > 0.0:
        sharpe = (
            annualized_return - RISK_FREE
        ) / annualized_volatility
    else:
        sharpe = np.nan

    var, cvar = mc.loss_var_cvar(values, ALPHA)
    wealth = np.cumprod(1.0 + values)
    max_drawdown = float(
        (wealth / np.maximum.accumulate(wealth) - 1.0).min()
    )

    return {
        "Total Return": float(wealth[-1] - 1.0),
        "Ann. Return": annualized_return,
        "Ann. Volatility": annualized_volatility,
        "Sharpe": float(sharpe),
        "95% VaR (daily)": float(var),
        "95% CVaR (daily)": float(cvar),
        "Max Drawdown": max_drawdown,
        "Daily Observations": int(len(values)),
    }


def run_optimizer_comparison(
    *,
    validation: pd.DataFrame,
    origins: list[pd.Timestamp],
    expected_return_sources: list[tuple[str, dict, list[str]]],
    covariance_forecasts: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the same baseline MPO optimizers for each supplied input set."""
    model_names = list(mb.BASELINES)
    daily_results = {}
    turnover_results = {}
    fee_results = {}
    weight_rows = []

    month_periods = validation.index.to_period("M")
    total_cases = sum(
        len(estimators) * len(model_names) * len(HORIZONS)
        for _, _, estimators in expected_return_sources
    )
    case_number = 0

    for return_source, mean_forecasts, estimators in expected_return_sources:
        for horizon in HORIZONS:
            for estimator in estimators:
                for model_name in model_names:
                    case_number += 1
                    if case_number == 1 or case_number % 10 == 0:
                        print(
                            f"[optimizer {case_number}/{total_cases}] "
                            f"{return_source} | {estimator} | "
                            f"{model_name} | H={horizon}",
                            flush=True,
                        )

                    pretrade_weights = np.zeros(len(ASSETS), dtype=float)
                    daily_parts = []
                    monthly_turnover = []
                    fees = []

                    uses_btc_cap, settings_builder = mb.BASELINES[model_name]

                    caps = np.ones(len(ASSETS), dtype=float)
                    if uses_btc_cap:
                        caps[ASSETS.index("BTC")] = BTC_CAP

                    for month_index, origin in enumerate(origins):
                        period = origin.to_period("M")
                        days = validation.loc[
                            month_periods == period, list(ASSETS)
                        ]
                        if days.empty:
                            raise ValueError(
                                f"No validation returns for {period}."
                            )
                        if days.index[0] != origin:
                            raise ValueError(
                                f"Rebalance date {origin.date()} is not the "
                                f"first validation return of {period}."
                            )

                        mean_rows = []
                        covariance_rows = []
                        for stage in range(1, horizon + 1):
                            mean_key = (origin, stage)
                            covariance_key = (origin, stage, estimator)
                            if mean_key not in mean_forecasts:
                                raise ValueError(
                                    f"Missing {return_source} mean forecast "
                                    f"at {origin.date()}, horizon {stage}."
                                )
                            if covariance_key not in covariance_forecasts:
                                raise ValueError(
                                    f"Missing {estimator} covariance at "
                                    f"{origin.date()}, horizon {stage}."
                                )

                            expected_period = period + (stage - 1)
                            if (
                                mean_forecasts[mean_key]["forecast_period"]
                                != expected_period
                            ):
                                raise ValueError(
                                    f"{return_source} mean forecast month "
                                    "does not align with the MPO stage."
                                )
                            mean_rows.append(mean_forecasts[mean_key]["mean"])
                            covariance_rows.append(
                                covariance_forecasts[covariance_key]
                            )

                        means = np.asarray(mean_rows, dtype=float)
                        covariances = np.asarray(covariance_rows, dtype=float)

                        try:
                            model_settings = settings_builder(
                                means, covariances, caps
                            )
                            plan = mb.solve_mv_plan(
                                covariances,
                                pretrade_weights,
                                expected_returns=means,
                                upper=caps,
                                risk_aversion=float(mb.DEFAULT_RISK_AVERSION),
                                cost_rate=COST_RATE,
                                **model_settings,
                            )
                            executed = mb.execute_first_period(
                                plan, days.to_numpy(dtype=float)
                            )
                        except (RuntimeError, ValueError, AssertionError) as exc:
                            raise RuntimeError(
                                f"{return_source} | {estimator} | "
                                f"{model_name} | H={horizon} | "
                                f"{origin.date()}: {exc}"
                            ) from exc

                        if pretrade_weights.sum() > 0.0:
                            monthly_turnover.append(
                                float(
                                    0.5 * np.abs(
                                        plan.first_weights - pretrade_weights
                                    ).sum()
                                )
                            )
                        fees.append(float(executed.fee_fraction))
                        daily_parts.append(pd.Series(
                            executed.net_daily_returns,
                            index=days.index,
                        ))

                        weight_rows.append({
                            "Return Forecast": return_source,
                            "Covariance Estimator": estimator,
                            "Baseline": model_name,
                            "Horizon": horizon,
                            "Rebalance Date": origin.strftime("%Y-%m-%d"),
                            "Forecast Month": str(period),
                            **{
                                f"Weight {asset}": float(plan.first_weights[i])
                                for i, asset in enumerate(ASSETS)
                            },
                            **{
                                f"Pretrade Weight {asset}": float(
                                    pretrade_weights[i]
                                )
                                for i, asset in enumerate(ASSETS)
                            },
                            "Planned Expected Return H1": float(
                                plan.expected_returns[0]
                            ),
                            "Forecast Volatility H1": float(
                                plan.forecast_volatility[0]
                            ),
                            "Turnover": (
                                float(
                                    0.5 * np.abs(
                                        plan.first_weights - pretrade_weights
                                    ).sum()
                                )
                                if pretrade_weights.sum() > 0.0
                                else 0.0
                            ),
                            "Execution Fee Fraction": float(
                                executed.fee_fraction
                            ),
                        })

                        pretrade_weights = np.asarray(
                            executed.ending_weights, dtype=float
                        )

                    key = (
                        return_source, estimator, model_name, horizon
                    )
                    daily_results[key] = pd.concat(daily_parts)
                    turnover_results[key] = (
                        float(np.mean(monthly_turnover))
                        if monthly_turnover else 0.0
                    )
                    fee_results[key] = float(np.sum(fees))

    result_rows = []
    for key, series in daily_results.items():
        return_source, estimator, model_name, horizon = key
        row = {
            "Return Forecast": return_source,
            "Covariance Estimator": estimator,
            "Baseline": model_name,
            "Horizon": horizon,
            **portfolio_performance(series),
            "Avg Monthly Turnover": turnover_results[key],
            "Total Costs Paid": fee_results[key],
        }

        matching_weights = [
            item for item in weight_rows
            if item["Return Forecast"] == return_source
            and item["Covariance Estimator"] == estimator
            and item["Baseline"] == model_name
            and item["Horizon"] == horizon
        ]
        if matching_weights:
            row["Avg BTC Weight"] = float(np.mean([
                item[f"Weight {ASSETS[2]}"]
                for item in matching_weights
            ]))
        result_rows.append(row)

    result_frame = pd.DataFrame(result_rows)
    returns_frame = pd.DataFrame({
        f"{source} | {estimator} | {model} | H={horizon}": series
        for (source, estimator, model, horizon), series in daily_results.items()
    })

    return (
        result_frame,
        pd.DataFrame(weight_rows),
        returns_frame,
    )


def optimizer_sharpe_comparisons(
    results: pd.DataFrame,
    selected_regime_estimator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate covariance effects from HMM-vs-WP2.4 mean-forecast effects."""
    wp_label = "WP2.4 training-selected"
    hmm_label = "HMM native"

    primary = results.loc[results["Return Forecast"] == wp_label]
    if primary.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = primary.pivot(
        index=["Baseline", "Horizon"],
        columns="Covariance Estimator",
        values="Sharpe",
    )

    covariance_rows = []
    if HMM_ESTIMATOR in wide.columns:
        comparators = [
            PLAIN_EWMA_ESTIMATOR,
            selected_regime_estimator,
            ROLLING_36_ESTIMATOR,
            ROLLING_60_ESTIMATOR,
            LEDOIT_WOLF_ESTIMATOR,
            CONSTANT_CORRELATION_ESTIMATOR,
        ]
        for (baseline, horizon), row in wide.iterrows():
            for estimator in comparators:
                if estimator not in wide.columns:
                    continue
                covariance_rows.append({
                    "Baseline": baseline,
                    "Horizon": int(horizon),
                    "Comparison": f"{estimator} minus HMM covariance",
                    "Covariance Estimator": estimator,
                    "Sharpe": float(row[estimator]),
                    "HMM Covariance Sharpe": float(row[HMM_ESTIMATOR]),
                    "Sharpe Difference vs HMM": float(
                        row[estimator] - row[HMM_ESTIMATOR]
                    ),
                })

    # Same HMM covariance and optimizer; only the expected-return source
    # changes in this comparison.
    wp_hmm_cov = results.loc[
        (results["Return Forecast"] == wp_label)
        & (results["Covariance Estimator"] == HMM_ESTIMATOR),
        ["Baseline", "Horizon", "Sharpe", "Ann. Return", "Total Return"],
    ].rename(columns={
        "Sharpe": "WP2.4 Mean Sharpe",
        "Ann. Return": "WP2.4 Mean Ann. Return",
        "Total Return": "WP2.4 Mean Total Return",
    })
    native_hmm = results.loc[
        (results["Return Forecast"] == hmm_label)
        & (results["Covariance Estimator"] == HMM_ESTIMATOR),
        ["Baseline", "Horizon", "Sharpe", "Ann. Return", "Total Return"],
    ].rename(columns={
        "Sharpe": "HMM Mean Sharpe",
        "Ann. Return": "HMM Mean Ann. Return",
        "Total Return": "HMM Mean Total Return",
    })
    mean_rows = wp_hmm_cov.merge(
        native_hmm,
        on=["Baseline", "Horizon"],
        how="inner",
    )
    if not mean_rows.empty:
        mean_rows["HMM Mean Sharpe Difference"] = (
            mean_rows["HMM Mean Sharpe"] - mean_rows["WP2.4 Mean Sharpe"]
        )
        mean_rows["Interpretation"] = (
            "Same HMM covariance and optimizer; expected-return source differs."
        )

    return pd.DataFrame(covariance_rows), mean_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_directory = cd.find_split_dir()
    train, validation, daily = cd.load_train_validation(data_directory)
    calendar = cd.build_calendar(daily)

    monthly_history, incomplete_history = cd.complete_monthly_returns(
        daily, calendar
    )
    train_monthly, incomplete_train = cd.complete_monthly_returns(
        train, calendar
    )
    validation_actuals, incomplete_validation = cd.complete_monthly_returns(
        validation, calendar
    )

    validation_periods = set(validation.index.to_period("M"))
    complete_validation_periods = set(validation_actuals.index)
    if validation_periods != complete_validation_periods:
        raise ValueError(
            "This comparison expects complete validation months. Incomplete "
            f"validation months: {incomplete_validation}"
        )

    origins = cd.expected_validation_rebalances(validation, calendar)
    if not origins:
        raise ValueError("No first-session validation rebalance dates found.")

    # Read all validation forecast inputs.
    wp_means = read_mean_forecasts(
        WP2_EXPECTED_RETURNS_FILE,
        source="WP2.4 expected returns",
    )
    hmm_covariances = read_covariance_forecasts(
        HMM_COVARIANCE_FILE, "HMM covariance"
    )
    plain_ewma_covariances = read_covariance_forecasts(
        EWMA_COVARIANCE_FILE, "EWMA lambda=0.93 covariance"
    )
    validation_probabilities = read_regime_probabilities(
        HMM_PROBABILITIES_FILE
    )

    require_forecast_coverage(wp_means, origins, "WP2.4 expected returns")
    require_forecast_coverage(hmm_covariances, origins, "HMM covariance")
    require_forecast_coverage(
        plain_ewma_covariances, origins, "EWMA lambda=0.93 covariance"
    )
    for origin in origins:
        if origin not in validation_probabilities:
            raise ValueError(
                f"Missing validation regime probabilities for {origin.date()}."
            )

    # Official HMM expected_returns.csv is preferred. The probability-weighted
    # conditional file is a cross-check and a fallback if the official file
    # is missing.
    conditional_path = find_optional_conditional_returns_file()
    conditional_means = None
    if conditional_path is not None:
        conditional_means = read_conditional_probability_weighted_returns(
            conditional_path
        )

    if HMM_EXPECTED_RETURNS_FILE.exists():
        hmm_means = read_mean_forecasts(
            HMM_EXPECTED_RETURNS_FILE,
            source="HMM expected returns",
        )
        hmm_mean_source_detail = str(HMM_EXPECTED_RETURNS_FILE)
    elif conditional_means is not None:
        hmm_means = conditional_means
        hmm_mean_source_detail = (
            f"{conditional_path} (probability-weighted fallback)"
        )
        warnings.warn(
            "HMM expected_returns.csv is absent; using the conditional "
            "returns file's probability-weighted forecasts as the HMM mean.",
            stacklevel=2,
        )
    else:
        raise FileNotFoundError(
            "Need HMM expected_returns.csv for the native-HMM optimizer run, "
            "or the probability-weighted conditional-returns CSV as fallback."
        )

    require_forecast_coverage(hmm_means, origins, "HMM expected returns")
    conditional_check = compare_hmm_mean_files(
        hmm_means, conditional_means
    )

    if (
        not conditional_check.empty
        and conditional_check["absolute_difference"].max() > 1e-6
    ):
        warnings.warn(
            "HMM expected_returns.csv differs from the probability-weighted "
            "conditional-returns file by more than 1e-6 for at least one "
            "forecast. The official expected_returns.csv is used for the "
            "native-HMM portfolio run; inspect the consistency-check output.",
            stacklevel=2,
        )

    # Fit HMM separately at eligible training origins using only prior
    # training observations, then select delta using training NLL only.
    print(
        f"Training origins: {train.index.min().date()} to "
        f"{train.index.max().date()} | building training-only regime "
        "probabilities and covariance scores.",
        flush=True,
    )
    training_scores = build_training_covariance_scores(
        train, train_monthly, calendar
    )
    delta_selection, selected_delta = select_regime_delta(training_scores)
    selected_regime_estimator = regime_ewma_estimator_name(selected_delta)

    print("\nTraining-only regime-EWMA delta selection:")
    print(delta_selection.to_string(index=False))
    print(
        f"\nSelected delta={selected_delta:.2f}; "
        f"validation regime-EWMA estimator={selected_regime_estimator}"
    )

    # Build validation matrices for all horizons. Past validation returns are
    # allowed into later origins' windows only after they have occurred.
    (
        covariance_forecasts,
        covariance_info,
        lambda_details,
        ewma_export_check,
    ) = build_validation_covariances(
        origins=origins,
        monthly_history=monthly_history,
        hmm_covariances=hmm_covariances,
        plain_ewma_covariances=plain_ewma_covariances,
        regime_probabilities=validation_probabilities,
        selected_delta=selected_delta,
    )

    validation_scores, covariance_exclusions = score_validation_covariances(
        origins=origins,
        daily=daily,
        monthly_history=monthly_history,
        validation_actuals=validation_actuals,
        covariance_forecasts=covariance_forecasts,
        covariance_info=covariance_info,
    )
    validation_summary = summarize_validation_covariance_scores(
        validation_scores
    )
    validation_paired = paired_covariance_comparisons(
        validation_scores, selected_regime_estimator
    )

    selected_regime_long = export_covariance_long_format(
        covariance_forecasts, selected_regime_estimator
    )

    # Save covariance diagnostics before the slower optimizer runs.
    training_scores.to_csv(
        OUTPUT_DIR / "regime_ewma_training_covariance_scores.csv",
        index=False,
    )
    delta_selection.to_csv(
        OUTPUT_DIR / "regime_ewma_training_delta_selection.csv",
        index=False,
    )
    validation_scores.to_csv(
        OUTPUT_DIR / "regime_ewma_validation_covariance_scores.csv",
        index=False,
    )
    validation_summary.to_csv(
        OUTPUT_DIR / "regime_ewma_validation_covariance_summary.csv",
        index=False,
    )
    validation_paired.to_csv(
        OUTPUT_DIR / "regime_ewma_validation_paired_comparisons.csv",
        index=False,
    )
    covariance_exclusions.to_csv(
        OUTPUT_DIR / "regime_ewma_validation_covariance_exclusions.csv",
        index=False,
    )
    lambda_details.to_csv(
        OUTPUT_DIR / "regime_ewma_validation_lambda_details.csv",
        index=False,
    )
    ewma_export_check.to_csv(
        OUTPUT_DIR / "ewma_export_consistency_check.csv",
        index=False,
    )
    selected_regime_long.to_csv(
        OUTPUT_DIR
        / "covariances_regime_ewma_training_selected_walk_forward_validation.csv",
        index=False,
    )
    conditional_check.to_csv(
        OUTPUT_DIR / "hmm_conditional_expected_return_consistency.csv",
        index=False,
    )

    print("\nValidation covariance summary by horizon:")
    print(validation_summary.to_string(index=False))

    # Primary experiment: WP2.4 expected returns fixed across all covariance
    # methods. Secondary experiment: HMM-native means plus HMM covariance.
    covariance_estimators = [
        HMM_ESTIMATOR,
        ROLLING_36_ESTIMATOR,
        ROLLING_60_ESTIMATOR,
        LEDOIT_WOLF_ESTIMATOR,
        CONSTANT_CORRELATION_ESTIMATOR,
        PLAIN_EWMA_ESTIMATOR,
        selected_regime_estimator,
    ]
    return_source_specs = [
        ("WP2.4 training-selected", wp_means, covariance_estimators),
        ("HMM native", hmm_means, [HMM_ESTIMATOR]),
    ]

    print(
        "\nStarting optimizer comparisons with the same baseline planner, "
        f"cost rate={COST_RATE:g}, BTC cap={BTC_CAP:g}, "
        f"risk aversion={mb.DEFAULT_RISK_AVERSION:g}.",
        flush=True,
    )
    optimizer_results, optimizer_weights, optimizer_returns = (
        run_optimizer_comparison(
            validation=validation,
            origins=origins,
            expected_return_sources=return_source_specs,
            covariance_forecasts=covariance_forecasts,
        )
    )

    sharpe_covariance_comparisons, sharpe_mean_comparisons = (
        optimizer_sharpe_comparisons(
            optimizer_results, selected_regime_estimator
        )
    )

    optimizer_results.to_csv(
        OUTPUT_DIR / "covariance_optimizer_results.csv",
        index=False,
    )
    optimizer_weights.to_csv(
        OUTPUT_DIR / "covariance_optimizer_weights.csv",
        index=False,
    )
    optimizer_returns.to_csv(
        OUTPUT_DIR / "covariance_optimizer_daily_returns.csv",
    )
    sharpe_covariance_comparisons.to_csv(
        OUTPUT_DIR / "covariance_optimizer_sharpe_vs_hmm_covariance.csv",
        index=False,
    )
    sharpe_mean_comparisons.to_csv(
        OUTPUT_DIR / "covariance_optimizer_hmm_native_vs_wp2_4_means.csv",
        index=False,
    )

    metadata = {
        "train_and_validation_only": True,
        "test_file_read": False,
        "training_data_directory": str(data_directory),
        "optimizer_directory": str(OPTIMIZER_DIRECTORY),
        "wp2_expected_returns_file": str(WP2_EXPECTED_RETURNS_FILE),
        "hmm_covariance_file": str(HMM_COVARIANCE_FILE),
        "hmm_probabilities_file": str(HMM_PROBABILITIES_FILE),
        "hmm_expected_returns_file": str(HMM_EXPECTED_RETURNS_FILE),
        "hmm_conditional_returns_file": (
            str(conditional_path) if conditional_path is not None else None
        ),
        "ewma_covariance_file": str(EWMA_COVARIANCE_FILE),
        "validation_first_date": validation.index.min().strftime("%Y-%m-%d"),
        "validation_last_date": validation.index.max().strftime("%Y-%m-%d"),
        "validation_rebalances": int(len(origins)),
        "horizons": list(HORIZONS),
        "assets": list(ASSETS),
        "plain_ewma_lambda": BASE_EWMA_LAMBDA,
        "regime_delta_grid": list(REGIME_DELTA_GRID),
        "selected_regime_delta_training_only": selected_delta,
        "selected_regime_ewma_estimator": selected_regime_estimator,
        "regime_ewma_formula": (
            "lambda(origin,horizon)=0.93-delta*p_bear_month_horizon; "
            "p_bear is forecast average occupancy of the higher-Equity-"
            "volatility state."
        ),
        "delta_selection": (
            "Lowest one-month-ahead multivariate Gaussian NLL on common "
            "training-only rolling-origin forecasts."
        ),
        "covariance_scoring_center": (
            "Shared expanding mean of complete monthly returns available "
            "at each rebalance; used to isolate covariance forecast scores."
        ),
        "portfolio_expected_returns": {
            "primary": "WP2.4 training-selected walk-forward expected returns",
            "separate_native_comparison": hmm_mean_source_detail,
        },
        "portfolio_covariances": covariance_estimators,
        "portfolio_optimizer": "Multi_Period_Baselines.solve_mv_plan",
        "portfolio_execution": (
            "Multi_Period_Baselines.execute_first_period; exact first-period "
            "execution accounting and self-financing fees."
        ),
        "cost_rate": COST_RATE,
        "btc_cap": BTC_CAP,
        "risk_aversion": float(mb.DEFAULT_RISK_AVERSION),
        "validation_is_exploratory": True,
        "notes": [
            "The WP2.4 experiment changes covariance while holding expected "
            "returns and optimizer settings fixed.",
            "The separate HMM-native experiment changes expected returns too; "
            "it is a whole-forecast-package comparison.",
            "Plain EWMA and static rolling/shrinkage covariances are carried "
            "unchanged across horizons 1-3.",
            "HMM covariance and regime-gated EWMA can vary by horizon.",
            "No HMM script or existing MPO runner was modified.",
        ],
    }
    (OUTPUT_DIR / "covariance_optimizer_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nOptimizer Sharpe results — WP2.4 expected returns:")
    primary_results = optimizer_results.loc[
        optimizer_results["Return Forecast"] == "WP2.4 training-selected"
    ]
    if primary_results.empty:
        print("No primary optimizer results.")
    else:
        sharpe_table = primary_results.pivot(
            index=["Baseline", "Covariance Estimator"],
            columns="Horizon",
            values="Sharpe",
        )
        sharpe_table.columns = [f"H={column}" for column in sharpe_table.columns]
        print(sharpe_table.round(3).to_string())

    print(
        f"\nFinished in {time.perf_counter() - start_time:.1f}s. "
        f"Outputs written to: {OUTPUT_DIR}"
    )
    print(
        "Validation is exploratory, not a fresh independent holdout. "
        "The project test split was not read."
    )


if __name__ == "__main__":
    main()
