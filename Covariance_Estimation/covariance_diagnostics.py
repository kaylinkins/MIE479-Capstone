"""Compare HMM and static one-month covariance forecasts.

Run from any directory:
    python Covariance_Estimation/covariance_diagnostics.py

Reads only:
  - portfolio_train.csv and portfolio_validation.csv
  - HMM_regime_forecasts/hmm_outputs/covariances.csv

Does NOT read portfolio_test.csv or modify the HMM script.

Validation-period estimates compared:
  - HMM covariance
  - 36-month rolling sample covariance
  - 60-month rolling sample covariance
  - 60-month Ledoit-Wolf correlation-shrinkage covariance
  - 60-month constant-correlation shrinkage covariance, with a fixed 50%
    weight on the constant-correlation target
  - 60-month EWMA covariance for each lambda in EWMA_LAMBDA_GRID

The EWMA lambda is selected using training-only rolling-origin forecasts:
the selected value minimizes mean multivariate Gaussian NLL on training
origins common to every value in the grid. Validation scores are reported
for comparison but are not used to select lambda.

All covariance methods are scored using the same expanding historical mean
as the expected-return center. This is intended to compare covariance
forecasts rather than different mean forecasts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import exchange_calendars as xcals
except ImportError as exc:
    raise ImportError(
        "Install dependencies with: "
        "python -m pip install numpy pandas exchange_calendars scikit-learn"
    ) from exc

try:
    from sklearn.covariance import LedoitWolf
except ImportError as exc:
    raise ImportError(
        "Ledoit-Wolf diagnostics need scikit-learn. Install it with: "
        "python -m pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

HMM_COVARIANCE_FILE = (
    ROOT / "HMM_regime_forecasts" / "hmm_outputs" / "covariances.csv"
)

ASSETS = ("Equity", "Bond", "BTC")
CALENDAR_NAME = "XNYS"

SHORT_ROLLING_WINDOW_MONTHS = 36
ROLLING_WINDOW_MONTHS = 60

ROLLING_36_ESTIMATOR = "rolling_sample_36m"
ROLLING_60_ESTIMATOR = "rolling_sample_60m"
LEDOIT_WOLF_ESTIMATOR = "ledoit_wolf_corr_60m"

# Fixed, pre-specified weight on the constant-correlation target.
# This is not tuned using validation scores.
CONSTANT_CORRELATION_SHRINKAGE = 0.50
CONSTANT_CORRELATION_ESTIMATOR = (
    "constant_corr_60m_alpha_"
    f"{CONSTANT_CORRELATION_SHRINKAGE:.2f}".replace(".", "_")
)

# EWMA decay values to evaluate. The 0.97 value is retained as a reference
# to the previously reported EWMA estimator.
EWMA_LAMBDA_MONTHLY = 0.97
EWMA_LAMBDA_GRID = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99)


def ewma_estimator_name(decay: float) -> str:
    """Return a stable estimator name for an EWMA decay value."""
    return f"ewma_cov_60m_lambda_{decay:.2f}".replace(".", "_")


EWMA_ESTIMATOR_BY_LAMBDA = {
    decay: ewma_estimator_name(decay)
    for decay in EWMA_LAMBDA_GRID
}
EWMA_ESTIMATORS = tuple(EWMA_ESTIMATOR_BY_LAMBDA.values())

if EWMA_LAMBDA_MONTHLY not in EWMA_ESTIMATOR_BY_LAMBDA:
    raise ValueError(
        "EWMA_LAMBDA_MONTHLY must be included in EWMA_LAMBDA_GRID."
    )

EWMA_ESTIMATOR = EWMA_ESTIMATOR_BY_LAMBDA[EWMA_LAMBDA_MONTHLY]

# Numerical tolerances used for matrix diagnostics.
SYMMETRY_ATOL = 1e-10
PSD_EIGENVALUE_ATOL = 1e-10


# ---------------------------------------------------------------------
# Inputs: train and validation only
# ---------------------------------------------------------------------

def normalise_dates(index) -> pd.DatetimeIndex:
    """Return a timezone-naive, normalized DatetimeIndex."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def find_split_dir() -> Path:
    """Find the allowed train/validation files without checking the test file."""
    candidates = [
        ROOT / "Data Work",
        ROOT / "HMM_regime_forecasts",
        ROOT / "Data" / "datasets",
    ]

    for folder in candidates:
        if all(
            (folder / f"portfolio_{split}.csv").exists()
            for split in ("train", "validation")
        ):
            return folder

    raise FileNotFoundError(
        "Could not find portfolio_train.csv and portfolio_validation.csv. "
        "Expected them in Data Work/, HMM_regime_forecasts/, or Data/datasets/."
    )


def read_return_split(folder: Path, split: str) -> pd.DataFrame:
    """Read one permitted split."""
    frame = pd.read_csv(
        folder / f"portfolio_{split}.csv",
        index_col=0,
        parse_dates=True,
    )

    if frame.empty or set(frame.columns) != set(ASSETS):
        raise ValueError(
            f"{split}: expected nonempty columns {ASSETS}; "
            f"got {tuple(frame.columns)}."
        )

    frame = frame.loc[:, list(ASSETS)].astype(float)
    frame.index = normalise_dates(frame.index)
    frame = frame.sort_index()

    values = frame.to_numpy(dtype=float)
    if (
        frame.index.hasnans
        or not frame.index.is_unique
        or not np.isfinite(values).all()
        or (values <= -1.0).any()
    ):
        raise ValueError(f"{split}: invalid dates or daily returns.")

    return frame


def load_train_validation(
    folder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train and validation only, and return their combined history."""
    train = read_return_split(folder, "train")
    validation = read_return_split(folder, "validation")

    if train.index.max() >= validation.index.min():
        raise ValueError(
            "Training and validation dates overlap or are misordered."
        )

    daily = pd.concat([train, validation]).sort_index()
    if not daily.index.is_unique:
        raise ValueError("Training and validation dates overlap.")

    return train, validation, daily


# ---------------------------------------------------------------------
# Calendar and monthly return construction
# ---------------------------------------------------------------------

def build_calendar(daily: pd.DataFrame):
    start = daily.index.min() - pd.Timedelta(days=10)

    # Include the full next forecast month; the exporter needs to identify
    # its first trading session even though no returns from that month are used.
    end = daily.index.max() + pd.Timedelta(days=60)

    return xcals.get_calendar(
        CALENDAR_NAME,
        start=str(start.date()),
        end=str(end.date()),
    )


def sessions_for_month(calendar, month: pd.Period) -> pd.DatetimeIndex:
    sessions = calendar.sessions_in_range(
        month.start_time.normalize(),
        month.end_time.normalize(),
    )
    return normalise_dates(sessions)


def complete_monthly_returns(
    daily: pd.DataFrame,
    calendar,
) -> tuple[pd.DataFrame, list[str]]:
    """Compound daily simple returns for complete XNYS calendar months.

    A month is included only when every XNYS session in that month is present
    in the input. Incomplete months are returned separately for reporting.
    """
    rows = []
    periods = []
    incomplete_months = []

    for month, group in daily.groupby(
        daily.index.to_period("M"),
        sort=True,
    ):
        expected_sessions = sessions_for_month(calendar, month)

        if (
            not len(expected_sessions)
            or not group.index.equals(expected_sessions)
        ):
            incomplete_months.append(str(month))
            continue

        values = group.to_numpy(dtype=float)
        monthly_return = np.prod(1.0 + values, axis=0) - 1.0

        rows.append(monthly_return)
        periods.append(month)

    result = pd.DataFrame(
        rows,
        index=pd.PeriodIndex(periods, freq="M"),
        columns=list(ASSETS),
    )
    result.index.name = "month"

    return result, incomplete_months


def expected_validation_rebalances(
    validation: pd.DataFrame,
    calendar,
) -> list[pd.Timestamp]:
    """Find first-XNYS-session rebalance dates present in validation."""
    validation_dates = set(validation.index)
    dates = []

    months = sorted(validation.index.to_period("M").unique())
    for month in months:
        sessions = sessions_for_month(calendar, month)
        if len(sessions) and sessions[0] in validation_dates:
            dates.append(sessions[0])

    return dates


# ---------------------------------------------------------------------
# HMM covariance input
# ---------------------------------------------------------------------

def read_hmm_one_month_matrices(path: Path) -> list[dict]:
    """Read HMM horizon-1 covariance matrices from the long-format CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"HMM covariance file not found: {path}\n"
            "Run the HMM script first, or check the configured path."
        )

    frame = pd.read_csv(path)

    required = {
        "rebalance_date",
        "month_ahead",
        "forecast_month",
        "asset",
        *ASSETS,
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"HMM covariance file is missing required columns: "
            f"{sorted(required - set(frame.columns))}"
        )

    parsed_dates = pd.to_datetime(
        frame["rebalance_date"],
        errors="coerce",
    )
    if parsed_dates.isna().any():
        raise ValueError("HMM covariance file has invalid rebalance dates.")

    frame["rebalance_date"] = normalise_dates(parsed_dates)
    frame["month_ahead"] = pd.to_numeric(
        frame["month_ahead"],
        errors="raise",
    ).astype(int)
    frame["asset"] = frame["asset"].astype(str)

    # This diagnostic compares one-month covariance forecasts only.
    frame = frame.loc[frame["month_ahead"] == 1].copy()
    if frame.empty:
        raise ValueError(
            "The HMM covariance file contains no month_ahead = 1 rows."
        )

    try:
        frame["forecast_period"] = pd.PeriodIndex(
            frame["forecast_month"].astype(str),
            freq="M",
        )
    except Exception as exc:
        raise ValueError(
            "Could not parse HMM forecast_month values."
        ) from exc

    duplicate_rows = frame.duplicated(
        ["rebalance_date", "month_ahead", "asset"],
        keep=False,
    )
    if duplicate_rows.any():
        raise ValueError(
            "HMM covariance file contains duplicate horizon-1 "
            "rebalance/asset rows."
        )

    records = []

    for (rebalance_date, forecast_period), group in frame.groupby(
        ["rebalance_date", "forecast_period"],
        sort=True,
    ):
        issues = []

        if forecast_period != rebalance_date.to_period("M"):
            issues.append(
                "horizon-1 forecast month differs from rebalance month"
            )

        if (
            len(group) != len(ASSETS)
            or set(group["asset"]) != set(ASSETS)
            or group["asset"].duplicated().any()
        ):
            issues.append(
                "expected exactly one covariance row for each asset"
            )
            matrix = None
        else:
            ordered = group.set_index("asset").loc[
                list(ASSETS),
                list(ASSETS),
            ]
            numeric = ordered.apply(pd.to_numeric, errors="coerce")
            matrix = numeric.to_numpy(dtype=float)

        records.append({
            "rebalance_date": rebalance_date,
            "forecast_period": forecast_period,
            "matrix": matrix,
            "issue": "; ".join(issues),
        })

    return records


# ---------------------------------------------------------------------
# Covariance estimators and matrix diagnostics
# ---------------------------------------------------------------------

def inspect_matrix(
    matrix: np.ndarray,
) -> tuple[dict, np.ndarray | None, bool]:
    """Check matrix structure and return its symmetric version if finite."""
    matrix = np.asarray(matrix, dtype=float)
    n_assets = len(ASSETS)

    fields = {
        "matrix_finite": bool(np.isfinite(matrix).all()),
        "matrix_symmetric": False,
        "max_abs_asymmetry": np.nan,
        "minimum_eigenvalue": np.nan,
        "is_positive_semidefinite": False,
        "is_positive_definite": False,
        "condition_number": np.nan,
    }

    for i, row_asset in enumerate(ASSETS):
        for j, col_asset in enumerate(ASSETS):
            value = (
                float(matrix[i, j])
                if matrix.shape == (n_assets, n_assets)
                else np.nan
            )
            fields[f"cov_{row_asset}_{col_asset}"] = value

    if matrix.shape != (n_assets, n_assets) or not np.isfinite(matrix).all():
        return fields, None, False

    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    symmetric = asymmetry <= SYMMETRY_ATOL
    symmetric_matrix = (matrix + matrix.T) / 2.0

    fields["matrix_symmetric"] = symmetric
    fields["max_abs_asymmetry"] = asymmetry

    try:
        eigenvalues = np.linalg.eigvalsh(symmetric_matrix)
        minimum_eigenvalue = float(eigenvalues.min())
        fields["minimum_eigenvalue"] = minimum_eigenvalue
        fields["is_positive_semidefinite"] = (
            minimum_eigenvalue >= -PSD_EIGENVALUE_ATOL
        )

        if minimum_eigenvalue > 0.0:
            fields["condition_number"] = float(
                np.linalg.cond(symmetric_matrix)
            )

        try:
            np.linalg.cholesky(symmetric_matrix)
            positive_definite = symmetric and minimum_eigenvalue > 0.0
        except np.linalg.LinAlgError:
            positive_definite = False

        fields["is_positive_definite"] = positive_definite
    except np.linalg.LinAlgError:
        positive_definite = False

    diagonal = np.diag(symmetric_matrix)
    for i, asset in enumerate(ASSETS):
        variance = float(diagonal[i])
        fields[f"variance_{asset}"] = variance
        fields[f"volatility_{asset}"] = (
            float(np.sqrt(variance)) if variance > 0.0 else np.nan
        )

    for i, asset_i in enumerate(ASSETS):
        for j in range(i + 1, n_assets):
            asset_j = ASSETS[j]
            denominator = np.sqrt(diagonal[i] * diagonal[j])
            correlation = (
                float(symmetric_matrix[i, j] / denominator)
                if denominator > 0.0
                else np.nan
            )
            fields[f"correlation_{asset_i}_{asset_j}"] = correlation

    return fields, symmetric_matrix, bool(fields["is_positive_definite"])


def ledoit_wolf_correlation_covariance(
    monthly_returns: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Estimate a Ledoit-Wolf-shrunk correlation matrix and preserve volatilities.

    Ledoit-Wolf estimates the shrinkage intensity using standardized returns.
    The resulting correlation matrix is then rescaled using each asset's
    sample standard deviation from the same estimation window.
    """
    values = np.asarray(monthly_returns, dtype=float)

    if (
        values.ndim != 2
        or values.shape[1] != len(ASSETS)
        or values.shape[0] < 2
    ):
        raise ValueError(
            f"Expected at least two observations and {len(ASSETS)} assets."
        )
    if not np.isfinite(values).all():
        raise ValueError("Monthly returns contain non-finite values.")

    sample_volatility = values.std(axis=0, ddof=1)
    if (
        not np.isfinite(sample_volatility).all()
        or (sample_volatility <= 0).any()
    ):
        raise ValueError(
            "Cannot estimate correlations when an asset has zero or "
            "invalid volatility."
        )

    standardized = (
        values - values.mean(axis=0)
    ) / sample_volatility

    estimator = LedoitWolf(assume_centered=True).fit(standardized)
    shrunk_covariance_standardized = estimator.covariance_

    diagonal = np.diag(shrunk_covariance_standardized)
    if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError("Ledoit-Wolf produced invalid standardized variances.")

    # Convert the shrunk covariance of standardized returns back to
    # a correlation matrix, then restore each asset's sample volatility.
    scale = np.sqrt(diagonal)
    shrunk_correlation = (
        shrunk_covariance_standardized / np.outer(scale, scale)
    )
    shrunk_correlation = (
        shrunk_correlation + shrunk_correlation.T
    ) / 2.0
    np.fill_diagonal(shrunk_correlation, 1.0)

    covariance = (
        np.outer(sample_volatility, sample_volatility)
        * shrunk_correlation
    )
    covariance = (covariance + covariance.T) / 2.0

    return covariance, float(estimator.shrinkage_)


def constant_correlation_covariance(
    monthly_returns: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Estimate a fixed-alpha constant-correlation-shrunk covariance.

    The sample correlation matrix is blended with a target whose off-diagonal
    entries equal the sample's average pairwise correlation. The fixed
    shrinkage weight is CONSTANT_CORRELATION_SHRINKAGE. Sample volatilities
    are retained when the shrunk correlation matrix is rescaled to covariance.
    """
    values = np.asarray(monthly_returns, dtype=float)

    if (
        values.ndim != 2
        or values.shape[1] != len(ASSETS)
        or values.shape[0] < 2
    ):
        raise ValueError(
            f"Expected at least two observations and {len(ASSETS)} assets."
        )
    if not np.isfinite(values).all():
        raise ValueError("Monthly returns contain non-finite values.")

    sample_volatility = values.std(axis=0, ddof=1)
    if (
        not np.isfinite(sample_volatility).all()
        or (sample_volatility <= 0).any()
    ):
        raise ValueError(
            "Cannot estimate correlations when an asset has zero or "
            "invalid volatility."
        )

    sample_covariance = np.cov(
        values,
        rowvar=False,
        ddof=1,
    )
    sample_correlation = (
        sample_covariance
        / np.outer(sample_volatility, sample_volatility)
    )
    sample_correlation = (
        sample_correlation + sample_correlation.T
    ) / 2.0
    np.fill_diagonal(sample_correlation, 1.0)

    pairwise_correlations = sample_correlation[
        np.triu_indices(len(ASSETS), k=1)
    ]
    average_pairwise_correlation = float(pairwise_correlations.mean())

    target_correlation = np.full(
        (len(ASSETS), len(ASSETS)),
        average_pairwise_correlation,
        dtype=float,
    )
    np.fill_diagonal(target_correlation, 1.0)

    alpha = CONSTANT_CORRELATION_SHRINKAGE
    shrunk_correlation = (
        (1.0 - alpha) * sample_correlation
        + alpha * target_correlation
    )
    shrunk_correlation = (
        shrunk_correlation + shrunk_correlation.T
    ) / 2.0
    np.fill_diagonal(shrunk_correlation, 1.0)

    covariance = (
        np.outer(sample_volatility, sample_volatility)
        * shrunk_correlation
    )
    covariance = (covariance + covariance.T) / 2.0

    return covariance


def ewma_sample_covariance(
    monthly_returns: np.ndarray | pd.DataFrame,
    decay: float = EWMA_LAMBDA_MONTHLY,
) -> tuple[np.ndarray, float]:
    """Estimate a demeaned, exponentially weighted monthly covariance.

    The most recent observation receives the highest weight. The returned
    effective sample size summarizes how concentrated those weights are.
    """
    values = np.asarray(monthly_returns, dtype=float)

    if not 0.0 < decay < 1.0:
        raise ValueError("EWMA decay factor must be between 0 and 1.")
    if (
        values.ndim != 2
        or values.shape[1] != len(ASSETS)
        or values.shape[0] < 2
    ):
        raise ValueError(
            f"Expected at least two observations and {len(ASSETS)} assets."
        )
    if not np.isfinite(values).all():
        raise ValueError("Monthly returns contain non-finite values.")

    # Rows are oldest to newest; the newest observation has exponent zero.
    weights = decay ** np.arange(
        values.shape[0] - 1,
        -1,
        -1,
        dtype=float,
    )
    weights /= weights.sum()

    correction = 1.0 - float(weights @ weights)
    if correction <= 0.0:
        raise ValueError("EWMA weights do not support a covariance estimate.")

    weighted_mean = weights @ values
    centered = values - weighted_mean
    covariance = (
        (centered * weights[:, None]).T @ centered
    ) / correction
    covariance = (covariance + covariance.T) / 2.0

    effective_months = 1.0 / float(weights @ weights)
    return covariance, effective_months


# ---------------------------------------------------------------------
# Covariance scoring
# ---------------------------------------------------------------------

def score_covariance(
    *,
    metadata: dict,
    estimator: str,
    matrix: np.ndarray,
    symmetric_matrix: np.ndarray | None,
    is_positive_definite: bool,
    n_estimation_months: float,
    common_mean: np.ndarray,
    realized_return: np.ndarray,
) -> dict:
    """Score a covariance using a shared expanding-mean forecast center."""
    residual = realized_return - common_mean

    row = {
        **metadata,
        "estimator": estimator,
        "n_estimation_months": n_estimation_months,
        "multivariate_gaussian_nll_common_mean": np.nan,
        "mahalanobis_squared_common_mean": np.nan,
        "matrix_positive_definite": is_positive_definite,
    }

    if is_positive_definite and symmetric_matrix is not None:
        sign, log_determinant = np.linalg.slogdet(symmetric_matrix)
        if sign > 0:
            try:
                solved = np.linalg.solve(symmetric_matrix, residual)
                mahalanobis_squared = float(residual @ solved)
                nll = 0.5 * (
                    len(ASSETS) * np.log(2.0 * np.pi)
                    + log_determinant
                    + mahalanobis_squared
                )
                row["multivariate_gaussian_nll_common_mean"] = float(nll)
                row["mahalanobis_squared_common_mean"] = (
                    mahalanobis_squared
                )
            except np.linalg.LinAlgError:
                pass

    for i, asset in enumerate(ASSETS):
        variance = (
            float(matrix[i, i])
            if matrix.shape == (len(ASSETS), len(ASSETS))
            else np.nan
        )
        innovation = float(residual[i])
        realized = float(realized_return[i])
        mean_forecast = float(common_mean[i])

        row[f"common_mean_{asset}"] = mean_forecast
        row[f"realized_return_{asset}"] = realized
        row[f"innovation_{asset}"] = innovation
        row[f"innovation_squared_{asset}"] = innovation ** 2
        row[f"forecast_variance_{asset}"] = variance
        row[f"forecast_volatility_{asset}"] = (
            float(np.sqrt(variance))
            if np.isfinite(variance) and variance > 0.0
            else np.nan
        )

        if np.isfinite(variance) and variance > 0.0:
            row[f"univariate_gaussian_nll_{asset}"] = float(
                0.5 * (
                    np.log(2.0 * np.pi)
                    + np.log(variance)
                    + innovation ** 2 / variance
                )
            )
        else:
            row[f"univariate_gaussian_nll_{asset}"] = np.nan

    return row


# ---------------------------------------------------------------------
# Summary and paired-comparison tables
# ---------------------------------------------------------------------

def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize multivariate scores and per-asset variance calibration."""
    if scores.empty:
        return pd.DataFrame()

    rows = []

    for estimator, group in scores.groupby("estimator", sort=True):
        multivariate = group.dropna(
            subset=["multivariate_gaussian_nll_common_mean"]
        )

        row = {
            "estimator": estimator,
            "n_forecast_matrices": int(len(group)),
            "n_multivariate_scores": int(len(multivariate)),
            "mean_multivariate_gaussian_nll_common_mean": (
                float(
                    multivariate[
                        "multivariate_gaussian_nll_common_mean"
                    ].mean()
                )
                if len(multivariate) else np.nan
            ),
            "mean_mahalanobis_squared_common_mean": (
                float(
                    multivariate[
                        "mahalanobis_squared_common_mean"
                    ].mean()
                )
                if len(multivariate) else np.nan
            ),
            "mean_minimum_eigenvalue": float(
                group["minimum_eigenvalue"].mean()
            ),
            "maximum_condition_number": float(
                group["condition_number"].max()
            ),
            "n_non_positive_definite_matrices": int(
                (~group["matrix_positive_definite"].astype(bool)).sum()
            ),
            "mean_shrinkage_intensity": (
                float(group["shrinkage_intensity"].mean())
                if "shrinkage_intensity" in group
                else np.nan
            ),
            "mean_decay_factor": (
                float(group["decay_factor"].mean())
                if "decay_factor" in group else np.nan
            ),
            "mean_effective_sample_months": (
                float(group["effective_sample_months"].mean())
                if "effective_sample_months" in group else np.nan
            ),
        }

        for asset in ASSETS:
            variance_col = f"forecast_variance_{asset}"
            innovation_sq_col = f"innovation_squared_{asset}"
            univariate_nll_col = f"univariate_gaussian_nll_{asset}"
            volatility_col = f"forecast_volatility_{asset}"

            usable = group.dropna(
                subset=[variance_col, innovation_sq_col]
            )

            mean_forecast_variance = (
                float(usable[variance_col].mean())
                if len(usable) else np.nan
            )
            mean_realized_squared_innovation = (
                float(usable[innovation_sq_col].mean())
                if len(usable) else np.nan
            )

            if (
                np.isfinite(mean_forecast_variance)
                and np.isfinite(mean_realized_squared_innovation)
                and mean_realized_squared_innovation > 0.0
            ):
                variance_ratio = (
                    mean_forecast_variance
                    / mean_realized_squared_innovation
                )
            else:
                variance_ratio = np.nan

            row[f"n_univariate_scores_{asset}"] = int(
                group[univariate_nll_col].notna().sum()
            )
            row[f"mean_univariate_gaussian_nll_{asset}"] = float(
                group[univariate_nll_col].mean()
            )
            row[f"mean_forecast_variance_{asset}"] = (
                mean_forecast_variance
            )
            row[f"mean_realized_squared_innovation_{asset}"] = (
                mean_realized_squared_innovation
            )
            row[f"mean_variance_ratio_{asset}"] = variance_ratio
            row[f"mean_forecast_volatility_{asset}"] = float(
                group[volatility_col].mean()
            )

            if len(usable):
                row[
                    f"rmse_variance_vs_innovation_squared_{asset}"
                ] = float(np.sqrt(np.mean(
                    (
                        usable[variance_col].to_numpy(dtype=float)
                        - usable[innovation_sq_col].to_numpy(dtype=float)
                    ) ** 2
                )))
            else:
                row[
                    f"rmse_variance_vs_innovation_squared_{asset}"
                ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def paired_comparison(
    scores: pd.DataFrame,
    pair_specs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Compare covariance estimators on identical forecast months."""
    if scores.empty:
        return pd.DataFrame()

    if pair_specs is None:
        pair_specs = [
            (
                "hmm_regime",
                ROLLING_60_ESTIMATOR,
                "hmm_minus_rolling60",
            ),
            (
                "hmm_regime",
                LEDOIT_WOLF_ESTIMATOR,
                "hmm_minus_ledoit_wolf",
            ),
            (
                "hmm_regime",
                CONSTANT_CORRELATION_ESTIMATOR,
                "hmm_minus_constant_corr",
            ),
            (
                "hmm_regime",
                EWMA_ESTIMATOR,
                "hmm_minus_ewma_0_97",
            ),
            (
                EWMA_ESTIMATOR,
                ROLLING_60_ESTIMATOR,
                "ewma_0_97_minus_rolling60",
            ),
            (
                EWMA_ESTIMATOR,
                LEDOIT_WOLF_ESTIMATOR,
                "ewma_0_97_minus_ledoit_wolf",
            ),
            (
                "hmm_regime",
                ROLLING_36_ESTIMATOR,
                "hmm_minus_rolling36",
            ),
            (
                ROLLING_36_ESTIMATOR,
                ROLLING_60_ESTIMATOR,
                "rolling36_minus_rolling60",
            ),
            (
                ROLLING_36_ESTIMATOR,
                LEDOIT_WOLF_ESTIMATOR,
                "rolling36_minus_ledoit_wolf",
            ),
            (
                ROLLING_36_ESTIMATOR,
                EWMA_ESTIMATOR,
                "rolling36_minus_ewma_0_97",
            ),
            (
                ROLLING_36_ESTIMATOR,
                CONSTANT_CORRELATION_ESTIMATOR,
                "rolling36_minus_constant_corr",
            ),
        ]

    metrics = [
        (
            "multivariate_gaussian_nll_common_mean",
            "multivariate_nll",
        ),
        (
            "mahalanobis_squared_common_mean",
            "mahalanobis_squared",
        ),
        *[
            (
                f"univariate_gaussian_nll_{asset}",
                f"univariate_nll_{asset}",
            )
            for asset in ASSETS
        ],
    ]

    result = {
        "interpretation": (
            "For each pair, a negative first-minus-second difference favors "
            "the first estimator; lower scores are better."
        ),
    }
    index_columns = ["rebalance_date", "forecast_month"]

    for first, second, pair_name in pair_specs:
        for metric, metric_name in metrics:
            wide = scores.pivot(
                index=index_columns,
                columns="estimator",
                values=metric,
            )
            output_name = f"{pair_name}_{metric_name}"

            if not {first, second}.issubset(set(wide.columns)):
                result[f"n_paired_{output_name}"] = 0
                result[f"mean_{output_name}"] = np.nan
                result[f"fraction_{first}_lower_{output_name}"] = np.nan
                continue

            paired = wide[[first, second]].dropna()
            difference = paired[first] - paired[second]

            result[f"n_paired_{output_name}"] = int(len(paired))
            result[f"mean_{output_name}"] = (
                float(difference.mean()) if len(difference) else np.nan
            )
            result[f"fraction_{first}_lower_{output_name}"] = (
                float((difference < 0.0).mean())
                if len(difference) else np.nan
            )

    return pd.DataFrame([result])


def validation_pair_specs(
    selected_ewma_estimator: str,
) -> list[tuple[str, str, str]]:
    """Build validation comparisons including the training-selected EWMA."""
    pair_specs = [
        (
            "hmm_regime",
            ROLLING_36_ESTIMATOR,
            "hmm_minus_rolling36",
        ),
        (
            "hmm_regime",
            ROLLING_60_ESTIMATOR,
            "hmm_minus_rolling60",
        ),
        (
            "hmm_regime",
            LEDOIT_WOLF_ESTIMATOR,
            "hmm_minus_ledoit_wolf",
        ),
        (
            "hmm_regime",
            CONSTANT_CORRELATION_ESTIMATOR,
            "hmm_minus_constant_corr",
        ),
        (
            "hmm_regime",
            selected_ewma_estimator,
            "hmm_minus_training_selected_ewma",
        ),
        (
            ROLLING_36_ESTIMATOR,
            ROLLING_60_ESTIMATOR,
            "rolling36_minus_rolling60",
        ),
        (
            ROLLING_36_ESTIMATOR,
            LEDOIT_WOLF_ESTIMATOR,
            "rolling36_minus_ledoit_wolf",
        ),
        (
            ROLLING_36_ESTIMATOR,
            CONSTANT_CORRELATION_ESTIMATOR,
            "rolling36_minus_constant_corr",
        ),
        (
            selected_ewma_estimator,
            ROLLING_36_ESTIMATOR,
            "training_selected_ewma_minus_rolling36",
        ),
        (
            selected_ewma_estimator,
            ROLLING_60_ESTIMATOR,
            "training_selected_ewma_minus_rolling60",
        ),
        (
            selected_ewma_estimator,
            LEDOIT_WOLF_ESTIMATOR,
            "training_selected_ewma_minus_ledoit_wolf",
        ),
        (
            selected_ewma_estimator,
            CONSTANT_CORRELATION_ESTIMATOR,
            "training_selected_ewma_minus_constant_corr",
        ),
    ]

    # Keep the original 0.97 EWMA as a reference if training selected a
    # different value. This is a diagnostic comparison, not another choice.
    if selected_ewma_estimator != EWMA_ESTIMATOR:
        pair_specs.extend([
            (
                "hmm_regime",
                EWMA_ESTIMATOR,
                "hmm_minus_ewma_0_97",
            ),
            (
                EWMA_ESTIMATOR,
                ROLLING_36_ESTIMATOR,
                "ewma_0_97_minus_rolling36",
            ),
            (
                EWMA_ESTIMATOR,
                ROLLING_60_ESTIMATOR,
                "ewma_0_97_minus_rolling60",
            ),
            (
                EWMA_ESTIMATOR,
                LEDOIT_WOLF_ESTIMATOR,
                "ewma_0_97_minus_ledoit_wolf",
            ),
            (
                EWMA_ESTIMATOR,
                CONSTANT_CORRELATION_ESTIMATOR,
                "ewma_0_97_minus_constant_corr",
            ),
            (
                selected_ewma_estimator,
                EWMA_ESTIMATOR,
                "training_selected_ewma_minus_ewma_0_97",
            ),
        ])

    return pair_specs


# ---------------------------------------------------------------------
# Training-only EWMA lambda selection
# ---------------------------------------------------------------------

def select_ewma_lambda_from_training(
    training_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, str, float]:
    """Select lambda by mean NLL on training origins common to all lambdas.

    This function uses training scores only. It does not inspect validation
    scores or test data.
    """
    if training_scores.empty:
        raise ValueError(
            "No training covariance forecasts are available for EWMA "
            "lambda selection."
        )

    index_columns = ["rebalance_date", "forecast_month"]
    expected_estimators = list(EWMA_ESTIMATORS)

    ewma_scores = training_scores.loc[
        training_scores["estimator"].isin(expected_estimators)
    ].copy()

    nll_by_date = ewma_scores.pivot(
        index=index_columns,
        columns="estimator",
        values="multivariate_gaussian_nll_common_mean",
    )

    missing_estimators = set(expected_estimators) - set(nll_by_date.columns)
    if missing_estimators:
        raise ValueError(
            "Training scores are missing EWMA lambda candidate(s): "
            f"{sorted(missing_estimators)}"
        )

    # Use identical forecast origins for every candidate in the selection.
    common_nll = nll_by_date[expected_estimators].dropna()
    if common_nll.empty:
        raise ValueError(
            "No common, valid training forecast origins exist for all "
            "EWMA lambda candidates."
        )

    common_index = common_nll.index
    indexed_scores = ewma_scores.set_index(index_columns)

    rows = []
    for decay in EWMA_LAMBDA_GRID:
        estimator = EWMA_ESTIMATOR_BY_LAMBDA[decay]
        candidate_scores = indexed_scores.loc[
            indexed_scores["estimator"] == estimator
        ].reindex(common_index)

        rows.append({
            "ewma_lambda": float(decay),
            "estimator": estimator,
            "n_common_training_forecasts": int(len(common_nll)),
            "mean_training_multivariate_gaussian_nll": float(
                common_nll[estimator].mean()
            ),
            "mean_training_mahalanobis_squared": float(
                candidate_scores[
                    "mahalanobis_squared_common_mean"
                ].mean()
            ),
            "mean_training_minimum_eigenvalue": float(
                candidate_scores["minimum_eigenvalue"].mean()
            ),
            "maximum_training_condition_number": float(
                candidate_scores["condition_number"].max()
            ),
            "n_non_positive_definite_training_matrices": int(
                (~candidate_scores[
                    "matrix_positive_definite"
                ].astype(bool)).sum()
            ),
            "selection_criterion": (
                "Lowest mean multivariate Gaussian NLL on common "
                "training-only rolling-origin forecasts."
            ),
        })

    selection = pd.DataFrame(rows)
    nll_column = "mean_training_multivariate_gaussian_nll"

    selection["training_nll_rank"] = (
        selection[nll_column]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    # np.argmin returns the first candidate in grid order in case of a tie.
    selected_position = int(
        np.argmin(selection[nll_column].to_numpy(dtype=float))
    )
    selection["selected_by_training_nll"] = False
    selection.loc[selected_position, "selected_by_training_nll"] = True

    selected_estimator = str(
        selection.loc[selected_position, "estimator"]
    )
    selected_lambda = float(
        selection.loc[selected_position, "ewma_lambda"]
    )

    return selection, selected_estimator, selected_lambda


def build_ewma_lambda_validation_comparison(
    training_selection: pd.DataFrame,
    validation_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Report each grid value on validation and compare it with the HMM."""
    validation_summary = summarize_scores(validation_scores)

    if validation_summary.empty:
        summary_by_estimator = {}
    else:
        summary_by_estimator = (
            validation_summary
            .set_index("estimator")
            .to_dict(orient="index")
        )

    index_columns = ["rebalance_date", "forecast_month"]
    validation_nll = validation_scores.pivot(
        index=index_columns,
        columns="estimator",
        values="multivariate_gaussian_nll_common_mean",
    )
    validation_mahalanobis = validation_scores.pivot(
        index=index_columns,
        columns="estimator",
        values="mahalanobis_squared_common_mean",
    )

    rows = []

    for _, training_row in training_selection.iterrows():
        estimator = str(training_row["estimator"])
        validation_row = summary_by_estimator.get(estimator, {})

        row = {
            "ewma_lambda": float(training_row["ewma_lambda"]),
            "estimator": estimator,
            "selected_by_training_nll": bool(
                training_row["selected_by_training_nll"]
            ),
            "n_common_training_forecasts": int(
                training_row["n_common_training_forecasts"]
            ),
            "mean_training_multivariate_gaussian_nll": float(
                training_row[
                    "mean_training_multivariate_gaussian_nll"
                ]
            ),
            "mean_training_mahalanobis_squared": float(
                training_row[
                    "mean_training_mahalanobis_squared"
                ]
            ),
            "n_validation_multivariate_scores": int(
                validation_row.get("n_multivariate_scores", 0)
            ),
            "mean_validation_multivariate_gaussian_nll": (
                validation_row.get(
                    "mean_multivariate_gaussian_nll_common_mean",
                    np.nan,
                )
            ),
            "mean_validation_mahalanobis_squared": (
                validation_row.get(
                    "mean_mahalanobis_squared_common_mean",
                    np.nan,
                )
            ),
            "n_paired_validation_months_vs_hmm": 0,
            "mean_validation_nll_difference_ewma_minus_hmm": np.nan,
            "fraction_validation_ewma_lower_nll_than_hmm": np.nan,
            "mean_validation_mahalanobis_difference_ewma_minus_hmm": np.nan,
            "fraction_validation_ewma_lower_mahalanobis_than_hmm": np.nan,
        }

        if (
            estimator in validation_nll.columns
            and "hmm_regime" in validation_nll.columns
        ):
            paired_nll = validation_nll[
                [estimator, "hmm_regime"]
            ].dropna()

            if len(paired_nll):
                nll_difference = (
                    paired_nll[estimator] - paired_nll["hmm_regime"]
                )
                row["n_paired_validation_months_vs_hmm"] = int(
                    len(paired_nll)
                )
                row[
                    "mean_validation_nll_difference_ewma_minus_hmm"
                ] = float(nll_difference.mean())
                row[
                    "fraction_validation_ewma_lower_nll_than_hmm"
                ] = float((nll_difference < 0.0).mean())

        if (
            estimator in validation_mahalanobis.columns
            and "hmm_regime" in validation_mahalanobis.columns
        ):
            paired_mahalanobis = validation_mahalanobis[
                [estimator, "hmm_regime"]
            ].dropna()

            if len(paired_mahalanobis):
                mahalanobis_difference = (
                    paired_mahalanobis[estimator]
                    - paired_mahalanobis["hmm_regime"]
                )
                row[
                    "mean_validation_mahalanobis_difference_ewma_minus_hmm"
                ] = float(mahalanobis_difference.mean())
                row[
                    "fraction_validation_ewma_lower_mahalanobis_than_hmm"
                ] = float((mahalanobis_difference < 0.0).mean())

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Training-period rolling-origin diagnostics
# ---------------------------------------------------------------------

def build_training_walk_forward_scores(
    train_daily: pd.DataFrame,
    train_monthly_history: pd.DataFrame,
    calendar,
) -> pd.DataFrame:
    """Score static covariance forecasts at training-period monthly origins.

    Each covariance uses only complete training months before the target
    month. The HMM is not included because the available HMM covariance file
    contains validation-period origins.
    """
    rows = []

    def get_window_values(
        forecast_month: pd.Period,
        window_months: int,
    ) -> np.ndarray | None:
        required_months = pd.period_range(
            end=forecast_month - 1,
            periods=window_months,
            freq="M",
        )
        window_data = train_monthly_history.reindex(required_months)

        if (
            len(window_data) != window_months
            or window_data.isna().to_numpy().any()
        ):
            return None

        return window_data.to_numpy(dtype=float)

    for forecast_month in train_monthly_history.index:
        sessions = sessions_for_month(calendar, forecast_month)
        if not len(sessions):
            continue

        rebalance_date = sessions[0]
        prior_daily_dates = train_daily.index[
            train_daily.index < rebalance_date
        ]
        if not len(prior_daily_dates):
            continue

        information_date = prior_daily_dates[-1]
        past_months = train_monthly_history.loc[
            train_monthly_history.index < forecast_month
        ]
        if past_months.empty:
            continue

        values_36 = get_window_values(
            forecast_month,
            SHORT_ROLLING_WINDOW_MONTHS,
        )
        values_60 = get_window_values(
            forecast_month,
            ROLLING_WINDOW_MONTHS,
        )

        candidates = []

        if values_36 is not None:
            covariance_36 = np.cov(
                values_36,
                rowvar=False,
                ddof=1,
            )
            candidates.append((
                ROLLING_36_ESTIMATOR,
                covariance_36,
                SHORT_ROLLING_WINDOW_MONTHS,
                np.nan,
                np.nan,
                np.nan,
            ))

        if values_60 is not None:
            covariance_60 = np.cov(
                values_60,
                rowvar=False,
                ddof=1,
            )
            candidates.append((
                ROLLING_60_ESTIMATOR,
                covariance_60,
                ROLLING_WINDOW_MONTHS,
                np.nan,
                np.nan,
                np.nan,
            ))

            try:
                lw_covariance, lw_shrinkage = (
                    ledoit_wolf_correlation_covariance(values_60)
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"Training Ledoit-Wolf estimate failed for "
                    f"{forecast_month}: {exc}",
                    stacklevel=2,
                )
            else:
                candidates.append((
                    LEDOIT_WOLF_ESTIMATOR,
                    lw_covariance,
                    ROLLING_WINDOW_MONTHS,
                    lw_shrinkage,
                    np.nan,
                    np.nan,
                ))

            try:
                constant_corr_cov = constant_correlation_covariance(
                    values_60
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"Training constant-correlation estimate failed for "
                    f"{forecast_month}: {exc}",
                    stacklevel=2,
                )
            else:
                candidates.append((
                    CONSTANT_CORRELATION_ESTIMATOR,
                    constant_corr_cov,
                    ROLLING_WINDOW_MONTHS,
                    CONSTANT_CORRELATION_SHRINKAGE,
                    np.nan,
                    np.nan,
                ))

            for decay in EWMA_LAMBDA_GRID:
                estimator_name = EWMA_ESTIMATOR_BY_LAMBDA[decay]
                try:
                    ewma_covariance, effective_months = (
                        ewma_sample_covariance(
                            values_60,
                            decay=decay,
                        )
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    warnings.warn(
                        f"Training EWMA estimate failed for "
                        f"{forecast_month}, lambda={decay:.2f}: {exc}",
                        stacklevel=2,
                    )
                else:
                    candidates.append((
                        estimator_name,
                        ewma_covariance,
                        ROLLING_WINDOW_MONTHS,
                        np.nan,
                        decay,
                        effective_months,
                    ))

        if not candidates:
            continue

        prior_month_sessions = sessions_for_month(
            calendar,
            forecast_month - 1,
        )
        if (
            not len(prior_month_sessions)
            or information_date != prior_month_sessions[-1]
        ):
            raise ValueError(
                f"Training forecast for {forecast_month} does not have data "
                "through the final trading session of the prior month."
            )

        common_mean = past_months.loc[
            :,
            list(ASSETS),
        ].mean().to_numpy(dtype=float)
        realized_return = train_monthly_history.loc[
            forecast_month,
            list(ASSETS),
        ].to_numpy(dtype=float)

        metadata = {
            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
            "information_date": information_date.strftime("%Y-%m-%d"),
            "forecast_month": str(forecast_month),
            "n_prior_complete_months": int(len(past_months)),
        }

        for (
            estimator,
            matrix,
            window_months,
            shrinkage_intensity,
            decay_factor,
            effective_months,
        ) in candidates:
            structure, symmetric_matrix, positive_definite = inspect_matrix(
                matrix
            )

            score_row = score_covariance(
                metadata=metadata,
                estimator=estimator,
                matrix=matrix,
                symmetric_matrix=symmetric_matrix,
                is_positive_definite=positive_definite,
                n_estimation_months=window_months,
                common_mean=common_mean,
                realized_return=realized_return,
            )
            score_row.update(structure)
            score_row["shrinkage_intensity"] = shrinkage_intensity
            score_row["decay_factor"] = decay_factor
            score_row["effective_sample_months"] = effective_months
            rows.append(score_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------

def main() -> None:
    data_dir = find_split_dir()
    train, validation, daily = load_train_validation(data_dir)

    calendar = build_calendar(daily)

    # Keep separate monthly histories so training-origin estimates cannot
    # accidentally use validation returns.
    monthly_history, incomplete_history_months = complete_monthly_returns(
        daily,
        calendar,
    )
    train_monthly_history, incomplete_training_months = (
        complete_monthly_returns(train, calendar)
    )
    validation_actuals, incomplete_validation_months = (
        complete_monthly_returns(validation, calendar)
    )

    if incomplete_history_months:
        warnings.warn(
            "Excluded incomplete train/validation month(s) from the "
            f"common covariance history: {incomplete_history_months}",
            stacklevel=2,
        )

    if incomplete_training_months:
        warnings.warn(
            "Excluded incomplete training month(s) from the training-only "
            f"history: {incomplete_training_months}",
            stacklevel=2,
        )

    if incomplete_validation_months:
        warnings.warn(
            "Excluded incomplete validation month(s) from realized "
            f"scoring: {incomplete_validation_months}",
            stacklevel=2,
        )

    expected_rebalances = expected_validation_rebalances(
        validation,
        calendar,
    )

    hmm_groups = read_hmm_one_month_matrices(HMM_COVARIANCE_FILE)
    hmm_by_key = {
        (record["rebalance_date"], record["forecast_period"]): record
        for record in hmm_groups
    }

    expected_keys = {
        (date, date.to_period("M"))
        for date in expected_rebalances
    }

    # Select EWMA lambda using training data only, before constructing the
    # validation comparison.
    training_scores = build_training_walk_forward_scores(
        train,
        train_monthly_history,
        calendar,
    )
    training_summary = summarize_scores(training_scores)

    (
        ewma_training_selection,
        selected_ewma_estimator,
        selected_ewma_lambda,
    ) = select_ewma_lambda_from_training(training_scores)

    training_pair_specs = [
        (
            ROLLING_36_ESTIMATOR,
            ROLLING_60_ESTIMATOR,
            "rolling36_minus_rolling60",
        ),
        (
            ROLLING_36_ESTIMATOR,
            LEDOIT_WOLF_ESTIMATOR,
            "rolling36_minus_ledoit_wolf",
        ),
        (
            ROLLING_36_ESTIMATOR,
            CONSTANT_CORRELATION_ESTIMATOR,
            "rolling36_minus_constant_corr",
        ),
        (
            ROLLING_36_ESTIMATOR,
            EWMA_ESTIMATOR,
            "rolling36_minus_ewma_0_97",
        ),
        (
            CONSTANT_CORRELATION_ESTIMATOR,
            ROLLING_60_ESTIMATOR,
            "constant_corr_minus_rolling60",
        ),
        (
            CONSTANT_CORRELATION_ESTIMATOR,
            LEDOIT_WOLF_ESTIMATOR,
            "constant_corr_minus_ledoit_wolf",
        ),
        (
            CONSTANT_CORRELATION_ESTIMATOR,
            EWMA_ESTIMATOR,
            "constant_corr_minus_ewma_0_97",
        ),
    ]
    training_paired = paired_comparison(
        training_scores,
        pair_specs=training_pair_specs,
    )

    score_rows = []
    matrix_diagnostic_rows = []
    exclusion_rows = []

    rolling_36_available_dates = set()
    rolling_60_available_dates = set()
    ledoit_wolf_available_dates = set()
    constant_corr_available_dates = set()
    ewma_available_dates = {
        estimator: set()
        for estimator in EWMA_ESTIMATORS
    }

    for rebalance_date in expected_rebalances:
        forecast_month = rebalance_date.to_period("M")
        key = (rebalance_date, forecast_month)

        metadata = {
            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
            "information_date": "",
            "forecast_month": str(forecast_month),
        }

        dates_before_rebalance = daily.index[
            daily.index < rebalance_date
        ]
        if len(dates_before_rebalance):
            information_date = dates_before_rebalance[-1]
            metadata["information_date"] = information_date.strftime(
                "%Y-%m-%d"
            )

        # Score only a fully observed validation target month.
        if forecast_month not in validation_actuals.index:
            exclusion_rows.append({
                **metadata,
                "estimator": "all",
                "reason": (
                    "validation target month is incomplete or unavailable"
                ),
            })
            continue

        past_months = monthly_history.loc[
            monthly_history.index < forecast_month
        ]
        if past_months.empty:
            exclusion_rows.append({
                **metadata,
                "estimator": "all",
                "reason": (
                    "no completed monthly history before forecast month"
                ),
            })
            continue

        # Use the same no-look-ahead mean forecast for all covariance
        # estimates so the comparison isolates covariance performance.
        common_mean = past_months.loc[
            :,
            list(ASSETS),
        ].mean().to_numpy(dtype=float)
        realized_return = validation_actuals.loc[
            forecast_month,
            list(ASSETS),
        ].to_numpy(dtype=float)
        residual = realized_return - common_mean

        metadata["n_prior_complete_months"] = int(len(past_months))
        for i, asset in enumerate(ASSETS):
            metadata[f"common_mean_{asset}"] = float(common_mean[i])
            metadata[f"realized_return_{asset}"] = float(
                realized_return[i]
            )
            metadata[f"common_mean_innovation_{asset}"] = float(
                residual[i]
            )

        candidates = []

        # HMM forecast for this rebalance and target month.
        hmm_record = hmm_by_key.get(key)
        if hmm_record is None:
            exclusion_rows.append({
                **metadata,
                "estimator": "hmm_regime",
                "reason": "missing horizon-1 HMM covariance matrix",
            })
        elif hmm_record["issue"]:
            exclusion_rows.append({
                **metadata,
                "estimator": "hmm_regime",
                "reason": hmm_record["issue"],
            })
        elif hmm_record["matrix"] is None:
            exclusion_rows.append({
                **metadata,
                "estimator": "hmm_regime",
                "reason": "HMM covariance matrix could not be constructed",
            })
        else:
            candidates.append((
                "hmm_regime",
                hmm_record["matrix"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ))

        # 36-month sample covariance, using only months before forecast_month.
        lookback_36_months = pd.period_range(
            end=forecast_month - 1,
            periods=SHORT_ROLLING_WINDOW_MONTHS,
            freq="M",
        )
        lookback_36 = monthly_history.reindex(lookback_36_months)

        enough_36_history = (
            len(lookback_36) == SHORT_ROLLING_WINDOW_MONTHS
            and not lookback_36.isna().to_numpy().any()
        )

        if enough_36_history:
            covariance_36 = np.cov(
                lookback_36.to_numpy(dtype=float),
                rowvar=False,
                ddof=1,
            )
            candidates.append((
                ROLLING_36_ESTIMATOR,
                covariance_36,
                SHORT_ROLLING_WINDOW_MONTHS,
                np.nan,
                np.nan,
                np.nan,
            ))
            rolling_36_available_dates.add(rebalance_date)
        else:
            exclusion_rows.append({
                **metadata,
                "estimator": ROLLING_36_ESTIMATOR,
                "reason": (
                    f"fewer than {SHORT_ROLLING_WINDOW_MONTHS} consecutive "
                    "complete prior monthly returns"
                ),
            })

        # The 60-month estimators use the same completed monthly window.
        lookback_60_months = pd.period_range(
            end=forecast_month - 1,
            periods=ROLLING_WINDOW_MONTHS,
            freq="M",
        )
        lookback_60 = monthly_history.reindex(lookback_60_months)

        enough_60_history = (
            len(lookback_60) == ROLLING_WINDOW_MONTHS
            and not lookback_60.isna().to_numpy().any()
        )

        if enough_60_history:
            lookback_values = lookback_60.to_numpy(dtype=float)

            covariance_60 = np.cov(
                lookback_values,
                rowvar=False,
                ddof=1,
            )
            candidates.append((
                ROLLING_60_ESTIMATOR,
                covariance_60,
                ROLLING_WINDOW_MONTHS,
                np.nan,
                np.nan,
                np.nan,
            ))
            rolling_60_available_dates.add(rebalance_date)

            try:
                lw_covariance, lw_shrinkage = (
                    ledoit_wolf_correlation_covariance(lookback_values)
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                exclusion_rows.append({
                    **metadata,
                    "estimator": LEDOIT_WOLF_ESTIMATOR,
                    "reason": f"could not estimate covariance: {exc}",
                })
            else:
                candidates.append((
                    LEDOIT_WOLF_ESTIMATOR,
                    lw_covariance,
                    ROLLING_WINDOW_MONTHS,
                    lw_shrinkage,
                    np.nan,
                    np.nan,
                ))
                ledoit_wolf_available_dates.add(rebalance_date)

            try:
                constant_corr_cov = constant_correlation_covariance(
                    lookback_values
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                exclusion_rows.append({
                    **metadata,
                    "estimator": CONSTANT_CORRELATION_ESTIMATOR,
                    "reason": f"could not estimate covariance: {exc}",
                })
            else:
                candidates.append((
                    CONSTANT_CORRELATION_ESTIMATOR,
                    constant_corr_cov,
                    ROLLING_WINDOW_MONTHS,
                    CONSTANT_CORRELATION_SHRINKAGE,
                    np.nan,
                    np.nan,
                ))
                constant_corr_available_dates.add(rebalance_date)

            for decay in EWMA_LAMBDA_GRID:
                estimator_name = EWMA_ESTIMATOR_BY_LAMBDA[decay]

                try:
                    ewma_covariance, ewma_effective_months = (
                        ewma_sample_covariance(
                            lookback_values,
                            decay=decay,
                        )
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    exclusion_rows.append({
                        **metadata,
                        "estimator": estimator_name,
                        "reason": f"could not estimate covariance: {exc}",
                    })
                else:
                    candidates.append((
                        estimator_name,
                        ewma_covariance,
                        ROLLING_WINDOW_MONTHS,
                        np.nan,
                        decay,
                        ewma_effective_months,
                    ))
                    ewma_available_dates[estimator_name].add(rebalance_date)

        else:
            reason = (
                f"fewer than {ROLLING_WINDOW_MONTHS} consecutive complete "
                "prior monthly returns"
            )
            for estimator in (
                ROLLING_60_ESTIMATOR,
                LEDOIT_WOLF_ESTIMATOR,
                CONSTANT_CORRELATION_ESTIMATOR,
                *EWMA_ESTIMATORS,
            ):
                exclusion_rows.append({
                    **metadata,
                    "estimator": estimator,
                    "reason": reason,
                })

        for (
            estimator,
            matrix,
            n_estimation_months,
            shrinkage_intensity,
            decay_factor,
            effective_sample_months,
        ) in candidates:
            structure, symmetric_matrix, positive_definite = inspect_matrix(
                matrix
            )

            matrix_diagnostic_rows.append({
                **metadata,
                "estimator": estimator,
                "n_estimation_months": n_estimation_months,
                "shrinkage_intensity": shrinkage_intensity,
                "decay_factor": decay_factor,
                "effective_sample_months": effective_sample_months,
                **structure,
            })

            score_row = score_covariance(
                metadata=metadata,
                estimator=estimator,
                matrix=matrix,
                symmetric_matrix=symmetric_matrix,
                is_positive_definite=positive_definite,
                n_estimation_months=n_estimation_months,
                common_mean=common_mean,
                realized_return=realized_return,
            )
            score_row.update(structure)
            score_row["shrinkage_intensity"] = shrinkage_intensity
            score_row["decay_factor"] = decay_factor
            score_row["effective_sample_months"] = effective_sample_months
            score_rows.append(score_row)

    # Report HMM horizon-1 groups not on the validation first-session schedule.
    for record in hmm_groups:
        key = (record["rebalance_date"], record["forecast_period"])
        if key not in expected_keys:
            exclusion_rows.append({
                "rebalance_date": record["rebalance_date"].strftime("%Y-%m-%d"),
                "information_date": "",
                "forecast_month": str(record["forecast_period"]),
                "estimator": "hmm_regime",
                "reason": (
                    "HMM group is not a first-XNYS-session validation rebalance"
                ),
            })

    scores = pd.DataFrame(score_rows)
    matrix_diagnostics = pd.DataFrame(matrix_diagnostic_rows)
    exclusions = pd.DataFrame(exclusion_rows)

    summary = summarize_scores(scores)
    paired = paired_comparison(
        scores,
        pair_specs=validation_pair_specs(selected_ewma_estimator),
    )
    ewma_validation_comparison = (
        build_ewma_lambda_validation_comparison(
            ewma_training_selection,
            scores,
        )
    )

    # Count validation months where every estimator, including every EWMA
    # grid value, has a valid multivariate score.
    required_estimators = [
        "hmm_regime",
        ROLLING_36_ESTIMATOR,
        ROLLING_60_ESTIMATOR,
        LEDOIT_WOLF_ESTIMATOR,
        CONSTANT_CORRELATION_ESTIMATOR,
        *EWMA_ESTIMATORS,
    ]

    if not scores.empty:
        multivariate_by_date = scores.pivot(
            index=["rebalance_date", "forecast_month"],
            columns="estimator",
            values="multivariate_gaussian_nll_common_mean",
        )
        if set(required_estimators).issubset(
            set(multivariate_by_date.columns)
        ):
            n_paired_multivariate = int(
                multivariate_by_date[required_estimators].dropna().shape[0]
            )
        else:
            n_paired_multivariate = 0
    else:
        n_paired_multivariate = 0

    matched_hmm_groups = sum(
        1
        for date in expected_rebalances
        if (
            (date, date.to_period("M")) in hmm_by_key
            and not hmm_by_key[
                (date, date.to_period("M"))
            ]["issue"]
            and hmm_by_key[
                (date, date.to_period("M"))
            ]["matrix"] is not None
        )
    )

    coverage_record = {
        "split_directory": str(data_dir),
        "train_first_date": train.index.min().strftime("%Y-%m-%d"),
        "train_last_date": train.index.max().strftime("%Y-%m-%d"),
        "validation_first_date": validation.index.min().strftime("%Y-%m-%d"),
        "validation_last_date": validation.index.max().strftime("%Y-%m-%d"),
        "complete_joint_months_in_train_validation": int(
            len(monthly_history)
        ),
        "first_complete_joint_month": (
            str(monthly_history.index.min())
            if len(monthly_history) else ""
        ),
        "last_complete_joint_month": (
            str(monthly_history.index.max())
            if len(monthly_history) else ""
        ),
        "incomplete_train_validation_months_excluded": ";".join(
            incomplete_history_months
        ),
        "complete_validation_actual_months": int(
            len(validation_actuals)
        ),
        "incomplete_validation_months_excluded": ";".join(
            incomplete_validation_months
        ),
        "expected_validation_rebalances": int(
            len(expected_rebalances)
        ),
        "hmm_horizon1_matrix_groups": int(len(hmm_groups)),
        "matched_valid_hmm_horizon1_matrices": int(matched_hmm_groups),
        "rolling36_window_months": SHORT_ROLLING_WINDOW_MONTHS,
        "rebalances_with_rolling36_covariance": int(
            len(rolling_36_available_dates)
        ),
        "rolling_window_months": ROLLING_WINDOW_MONTHS,
        "rebalances_with_rolling60_covariance": int(
            len(rolling_60_available_dates)
        ),
        "rebalances_with_ledoit_wolf_corr_60m": int(
            len(ledoit_wolf_available_dates)
        ),
        "rebalances_with_constant_corr_60m": int(
            len(constant_corr_available_dates)
        ),
        "constant_correlation_target_weight": (
            CONSTANT_CORRELATION_SHRINKAGE
        ),
        "ewma_lambda_grid": ";".join(
            f"{decay:.2f}" for decay in EWMA_LAMBDA_GRID
        ),
        "training_selected_ewma_lambda": selected_ewma_lambda,
        "training_selected_ewma_estimator": selected_ewma_estimator,
        "ewma_selection_criterion": (
            "Lowest mean multivariate Gaussian NLL on common "
            "training-only rolling-origin forecasts."
        ),
        "rebalances_with_ewma_covariance_lambda_0_97": int(
            len(ewma_available_dates[EWMA_ESTIMATOR])
        ),
        "paired_multivariate_scored_months_all_estimators": int(
            n_paired_multivariate
        ),
        "covariance_score_center": (
            "Expanding mean of complete joint monthly returns available "
            "before each forecast month; common to all covariance estimators."
        ),
        "test_file_read": "no",
    }

    for estimator_name in EWMA_ESTIMATORS:
        coverage_record[
            f"rebalances_with_{estimator_name}"
        ] = int(len(ewma_available_dates[estimator_name]))

    coverage = pd.DataFrame([coverage_record])

    # Training-only coverage and common-origin count.
    training_counts = (
        training_scores.groupby("estimator")["forecast_month"]
        .nunique()
        .to_dict()
        if not training_scores.empty else {}
    )

    training_estimators = [
        ROLLING_36_ESTIMATOR,
        ROLLING_60_ESTIMATOR,
        LEDOIT_WOLF_ESTIMATOR,
        CONSTANT_CORRELATION_ESTIMATOR,
        *EWMA_ESTIMATORS,
    ]

    training_common_months_all_static = 0
    if not training_scores.empty:
        training_multivariate_by_date = training_scores.pivot(
            index=["rebalance_date", "forecast_month"],
            columns="estimator",
            values="multivariate_gaussian_nll_common_mean",
        )
        if set(training_estimators).issubset(
            set(training_multivariate_by_date.columns)
        ):
            training_common_months_all_static = int(
                training_multivariate_by_date[
                    training_estimators
                ].dropna().shape[0]
            )

    training_coverage_record = {
        "train_first_date": train.index.min().strftime("%Y-%m-%d"),
        "train_last_date": train.index.max().strftime("%Y-%m-%d"),
        "first_complete_train_month": (
            str(train_monthly_history.index.min())
            if len(train_monthly_history) else ""
        ),
        "last_complete_train_month": (
            str(train_monthly_history.index.max())
            if len(train_monthly_history) else ""
        ),
        "complete_train_months": int(len(train_monthly_history)),
        "incomplete_train_months_excluded": ";".join(
            incomplete_training_months
        ),
        "rolling36_training_forecasts": int(
            training_counts.get(ROLLING_36_ESTIMATOR, 0)
        ),
        "rolling60_training_forecasts": int(
            training_counts.get(ROLLING_60_ESTIMATOR, 0)
        ),
        "ledoit_wolf_60m_training_forecasts": int(
            training_counts.get(LEDOIT_WOLF_ESTIMATOR, 0)
        ),
        "constant_corr_60m_training_forecasts": int(
            training_counts.get(CONSTANT_CORRELATION_ESTIMATOR, 0)
        ),
        "ewma_lambda_grid": ";".join(
            f"{decay:.2f}" for decay in EWMA_LAMBDA_GRID
        ),
        "ewma_60m_training_forecasts_lambda_0_97": int(
            training_counts.get(EWMA_ESTIMATOR, 0)
        ),
        "training_selected_ewma_lambda": selected_ewma_lambda,
        "training_selected_ewma_estimator": selected_ewma_estimator,
        "ewma_selection_common_training_forecasts": int(
            ewma_training_selection[
                "n_common_training_forecasts"
            ].iloc[0]
        ),
        "training_months_scored_for_all_static_estimators": int(
            training_common_months_all_static
        ),
        "constant_correlation_target_weight": (
            CONSTANT_CORRELATION_SHRINKAGE
        ),
        "test_file_read": "no",
    }

    for estimator_name in EWMA_ESTIMATORS:
        training_coverage_record[
            f"training_forecasts_{estimator_name}"
        ] = int(training_counts.get(estimator_name, 0))

    training_coverage = pd.DataFrame([training_coverage_record])

    # Write validation, selection, and training diagnostic files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores.to_csv(
        OUTPUT_DIR / "covariance_validation_monthly_scores.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "covariance_validation_summary.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "covariance_validation_paired_comparison.csv",
        index=False,
    )
    ewma_validation_comparison.to_csv(
        OUTPUT_DIR / "ewma_lambda_validation_comparison.csv",
        index=False,
    )
    matrix_diagnostics.to_csv(
        OUTPUT_DIR / "covariance_matrix_diagnostics.csv",
        index=False,
    )
    coverage.to_csv(
        OUTPUT_DIR / "covariance_validation_coverage.csv",
        index=False,
    )
    exclusions.to_csv(
        OUTPUT_DIR / "covariance_validation_exclusions.csv",
        index=False,
    )

    ewma_training_selection.to_csv(
        OUTPUT_DIR / "ewma_lambda_training_selection.csv",
        index=False,
    )
    training_scores.to_csv(
        OUTPUT_DIR / "covariance_training_walk_forward_scores.csv",
        index=False,
    )
    training_summary.to_csv(
        OUTPUT_DIR / "covariance_training_walk_forward_summary.csv",
        index=False,
    )
    training_paired.to_csv(
        OUTPUT_DIR / "covariance_training_walk_forward_paired_comparison.csv",
        index=False,
    )
    training_coverage.to_csv(
        OUTPUT_DIR / "covariance_training_walk_forward_coverage.csv",
        index=False,
    )

    print(f"Train/validation data read from: {data_dir}")
    print(f"HMM covariance file read from: {HMM_COVARIANCE_FILE}")
    print(f"EWMA lambda grid: {EWMA_LAMBDA_GRID}")
    print(
        "Test file read: no. "
        "Validation scores are not used to select the EWMA lambda."
    )

    print("\nTRAINING-ONLY EWMA LAMBDA SELECTION:")
    print(
        "Selected lambda: "
        f"{selected_ewma_lambda:.2f} "
        f"({selected_ewma_estimator})"
    )
    print(ewma_training_selection.to_string(index=False))

    print(
        "\nVALIDATION EWMA GRID COMPARISON "
        "(training-selected lambda is flagged; validation is diagnostic):"
    )
    if ewma_validation_comparison.empty:
        print("No EWMA validation comparisons are available.")
    else:
        print(ewma_validation_comparison.to_string(index=False))

    print("\nVALIDATION COVARIANCE SUMMARY:")
    if summary.empty:
        print("No validation covariance forecasts could be scored.")
    else:
        print(summary.to_string(index=False))

    print("\nPAIRED VALIDATION COMPARISONS:")
    if paired.empty:
        print("No paired validation comparisons are available.")
    else:
        print(paired.to_string(index=False))

    print("\nTRAINING-ONLY WALK-FORWARD SUMMARY:")
    if training_summary.empty:
        print("No training covariance forecasts could be scored.")
    else:
        print(training_summary.to_string(index=False))

    print("\nTRAINING-ONLY PAIRED COMPARISONS:")
    if training_paired.empty:
        print("No paired training comparisons are available.")
    else:
        print(training_paired.to_string(index=False))

    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
