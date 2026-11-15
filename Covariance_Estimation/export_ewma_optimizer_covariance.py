"""Export EWMA lambda=0.93 covariance forecasts in HMM-compatible format.

Run from any directory:
    python Covariance_Estimation/export_ewma_optimizer_covariance.py

Reads train and validation returns only. It does not read portfolio_test.csv
or modify the teammate's HMM script.

Writes:
  - A monthly walk-forward forecast file for validation rebalance origins,
    with horizons 1, 2, and 3.
  - A separate latest-available-origin forecast file for horizons 1, 2, 3.

The long-format covariance CSV columns match the teammate's HMM covariance
CSV:
    rebalance_date, month_ahead, forecast_month, asset, Equity, Bond, BTC

For each origin, covariance is estimated from the trailing 60 complete
monthly return vectors strictly before that origin. The same EWMA covariance
matrix is carried forward across the three horizons: this is an explicit
flat-covariance forecast assumption, not a dynamic volatility model.
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
        "python -m pip install numpy pandas exchange_calendars scikit-learn"
    ) from exc

import covariance_diagnostics as diagnostics


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

ASSETS = tuple(diagnostics.ASSETS)
CALENDAR_NAME = diagnostics.CALENDAR_NAME

EWMA_LAMBDA = 0.93
ESTIMATION_WINDOW_MONTHS = 60
FORECAST_HORIZONS = (1, 2, 3)

OUTPUT_COLUMNS = [
    "rebalance_date",
    "month_ahead",
    "forecast_month",
    "asset",
    *ASSETS,
]

SYMMETRY_ATOL = 1e-10


# ---------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------

def build_export_calendar(daily: pd.DataFrame):
    """Build a calendar covering observed data and the forecast months."""
    first_date = daily.index.min() - pd.Timedelta(days=10)
    last_observed_month = daily.index.max().to_period("M")

    # Extend far enough to cover the next rebalance date and the forecast
    # months. The future calendar is used only for dates, never for returns.
    final_calendar_month = (
        last_observed_month + max(FORECAST_HORIZONS) + 1
    )
    last_date = final_calendar_month.end_time.normalize()

    return xcals.get_calendar(
        CALENDAR_NAME,
        start=str(first_date.date()),
        end=str(last_date.date()),
    )


# ---------------------------------------------------------------------
# Walk-forward origin validation and covariance estimation
# ---------------------------------------------------------------------

def information_date_for_origin(
    *,
    rebalance_date: pd.Timestamp,
    origin_month: pd.Period,
    daily: pd.DataFrame,
    calendar,
) -> pd.Timestamp:
    """Require data through the previous month's final XNYS session."""
    origin_sessions = diagnostics.sessions_for_month(calendar, origin_month)
    if not len(origin_sessions):
        raise ValueError(f"No XNYS sessions found for {origin_month}.")

    if rebalance_date != origin_sessions[0]:
        raise ValueError(
            f"{rebalance_date:%Y-%m-%d} is not the first XNYS session "
            f"of {origin_month}."
        )

    prior_dates = daily.index[daily.index < rebalance_date]
    if not len(prior_dates):
        raise ValueError(
            f"No observed return data before rebalance "
            f"{rebalance_date:%Y-%m-%d}."
        )

    information_date = prior_dates[-1]
    prior_month_sessions = diagnostics.sessions_for_month(
        calendar,
        origin_month - 1,
    )

    if not len(prior_month_sessions):
        raise ValueError(
            f"No XNYS sessions found for the month before {origin_month}."
        )

    expected_information_date = prior_month_sessions[-1]
    if information_date != expected_information_date:
        raise ValueError(
            f"Data before the {origin_month} rebalance end on "
            f"{information_date:%Y-%m-%d}, not on the previous month's "
            f"final XNYS session ({expected_information_date:%Y-%m-%d})."
        )

    return information_date


def trailing_monthly_window(
    monthly_history: pd.DataFrame,
    origin_month: pd.Period,
) -> pd.DataFrame:
    """Return the exact trailing window strictly before an origin month."""
    required_months = pd.period_range(
        end=origin_month - 1,
        periods=ESTIMATION_WINDOW_MONTHS,
        freq="M",
    )
    window = monthly_history.reindex(required_months)

    missing = window.index[window.isna().any(axis=1)]
    if len(missing):
        raise ValueError(
            f"Missing complete monthly returns in the trailing "
            f"{ESTIMATION_WINDOW_MONTHS}-month window: "
            f"{[str(month) for month in missing]}"
        )

    if not (window.index < origin_month).all():
        raise ValueError(
            f"Look-ahead check failed for origin {origin_month}."
        )

    if window.index.max() != origin_month - 1:
        raise ValueError(
            f"The estimation window for {origin_month} does not end "
            "in the immediately preceding month."
        )

    return window.loc[:, list(ASSETS)]


def estimate_origin_covariance(
    *,
    rebalance_date: pd.Timestamp,
    daily: pd.DataFrame,
    monthly_history: pd.DataFrame,
    calendar,
    origin_type: str,
) -> tuple[list[dict], dict]:
    """Create HMM-format rows for one walk-forward rebalance origin."""
    origin_month = rebalance_date.to_period("M")

    information_date = information_date_for_origin(
        rebalance_date=rebalance_date,
        origin_month=origin_month,
        daily=daily,
        calendar=calendar,
    )

    window = trailing_monthly_window(monthly_history, origin_month)
    window_values = window.to_numpy(dtype=float)

    covariance, effective_months = diagnostics.ewma_sample_covariance(
        window_values,
        decay=EWMA_LAMBDA,
    )

    structure, symmetric_covariance, positive_definite = (
        diagnostics.inspect_matrix(covariance)
    )
    if not positive_definite or symmetric_covariance is None:
        raise ValueError(
            f"EWMA covariance for {origin_month} is not positive definite."
        )

    if not np.isfinite(symmetric_covariance).all():
        raise ValueError(
            f"EWMA covariance for {origin_month} contains non-finite values."
        )

    rows = []
    matrices_by_horizon = {}

    for horizon in FORECAST_HORIZONS:
        forecast_month = origin_month + (horizon - 1)
        matrices_by_horizon[horizon] = symmetric_covariance

        for row_index, row_asset in enumerate(ASSETS):
            row = {
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "month_ahead": int(horizon),
                "forecast_month": str(forecast_month),
                "asset": row_asset,
            }

            for column_index, column_asset in enumerate(ASSETS):
                row[column_asset] = float(
                    symmetric_covariance[row_index, column_index]
                )

            rows.append(row)

    origin_metadata = {
        "origin_type": origin_type,
        "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
        "information_date": information_date.strftime("%Y-%m-%d"),
        "origin_month": str(origin_month),
        "forecast_months": [
            str(origin_month + (horizon - 1))
            for horizon in FORECAST_HORIZONS
        ],
        "estimation_start_month": str(window.index.min()),
        "estimation_end_month": str(window.index.max()),
        "n_estimation_months": int(len(window)),
        "ewma_lambda": float(EWMA_LAMBDA),
        "effective_sample_months": float(effective_months),
        "minimum_eigenvalue": float(
            structure["minimum_eigenvalue"]
        ),
        "condition_number": float(structure["condition_number"]),
        "positive_definite": bool(positive_definite),
        "horizon_covariance_assumption": (
            "Same origin covariance carried forward for horizons 1-3."
        ),
    }

    # Defensive check: horizons should be flat/persistent under this method.
    reference = matrices_by_horizon[FORECAST_HORIZONS[0]]
    for horizon in FORECAST_HORIZONS[1:]:
        if not np.allclose(
            reference,
            matrices_by_horizon[horizon],
            rtol=0.0,
            atol=SYMMETRY_ATOL,
        ):
            raise ValueError(
                f"Unexpected horizon-dependent covariance for "
                f"{rebalance_date:%Y-%m-%d}."
            )

    return rows, origin_metadata


def build_origin_forecasts(
    *,
    origins: list[pd.Timestamp],
    daily: pd.DataFrame,
    monthly_history: pd.DataFrame,
    calendar,
    origin_type: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build forecasts for each origin and report any excluded origins."""
    rows = []
    origin_metadata = []
    exclusions = []

    for rebalance_date in origins:
        try:
            origin_rows, metadata = estimate_origin_covariance(
                rebalance_date=rebalance_date,
                daily=daily,
                monthly_history=monthly_history,
                calendar=calendar,
                origin_type=origin_type,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            exclusions.append({
                "origin_type": origin_type,
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "origin_month": str(rebalance_date.to_period("M")),
                "reason": str(exc),
            })
            continue

        rows.extend(origin_rows)
        origin_metadata.append(metadata)

    return rows, origin_metadata, exclusions


# ---------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------

def validate_long_covariance_file(
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Validate schema, horizon labels, symmetry, and positive definiteness."""
    if frame.empty:
        raise ValueError(f"{label}: no covariance rows were produced.")

    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            f"{label}: unexpected columns {list(frame.columns)}; "
            f"expected {OUTPUT_COLUMNS}."
        )

    duplicate_keys = frame.duplicated(
        ["rebalance_date", "month_ahead", "asset"],
        keep=False,
    )
    if duplicate_keys.any():
        raise ValueError(f"{label}: duplicate rebalance/horizon/asset rows.")

    matrices_by_origin = {}

    for (rebalance_date, horizon), group in frame.groupby(
        ["rebalance_date", "month_ahead"],
        sort=False,
    ):
        horizon = int(horizon)
        if horizon not in FORECAST_HORIZONS:
            raise ValueError(
                f"{label}: unexpected month_ahead value {horizon}."
            )

        if len(group) != len(ASSETS) or set(group["asset"]) != set(ASSETS):
            raise ValueError(
                f"{label}: expected exactly one row for each asset at "
                f"{rebalance_date}, horizon {horizon}."
            )

        rebalance_month = pd.Timestamp(rebalance_date).to_period("M")
        expected_forecast_month = rebalance_month + (horizon - 1)
        forecast_months = {
            pd.Period(str(value), freq="M")
            for value in group["forecast_month"]
        }

        if forecast_months != {expected_forecast_month}:
            raise ValueError(
                f"{label}: incorrect forecast_month for "
                f"{rebalance_date}, horizon {horizon}."
            )

        ordered = group.set_index("asset").loc[
            list(ASSETS),
            list(ASSETS),
        ]
        matrix = ordered.to_numpy(dtype=float)

        _, symmetric_matrix, positive_definite = diagnostics.inspect_matrix(
            matrix
        )
        if symmetric_matrix is None:
            raise ValueError(
                f"{label}: invalid covariance matrix at "
                f"{rebalance_date}, horizon {horizon}."
            )
        if not positive_definite:
            raise ValueError(
                f"{label}: covariance is not positive definite at "
                f"{rebalance_date}, horizon {horizon}."
            )

        matrices_by_origin.setdefault(rebalance_date, {})[horizon] = (
            symmetric_matrix
        )

    for rebalance_date, horizon_matrices in matrices_by_origin.items():
        if set(horizon_matrices) != set(FORECAST_HORIZONS):
            raise ValueError(
                f"{label}: origin {rebalance_date} does not have all "
                f"horizons {FORECAST_HORIZONS}."
            )

        reference = horizon_matrices[FORECAST_HORIZONS[0]]
        for horizon in FORECAST_HORIZONS[1:]:
            if not np.allclose(
                reference,
                horizon_matrices[horizon],
                rtol=0.0,
                atol=SYMMETRY_ATOL,
            ):
                raise ValueError(
                    f"{label}: expected the flat covariance forecast "
                    f"to be identical across horizons for {rebalance_date}."
                )


# ---------------------------------------------------------------------
# Training-only EWMA lambda audit
# ---------------------------------------------------------------------

def get_training_lambda_selection(
    *,
    train: pd.DataFrame,
    train_monthly_history: pd.DataFrame,
    calendar,
) -> tuple[pd.DataFrame, str, float]:
    """Recheck the lambda choice using training data only."""
    training_scores = diagnostics.build_training_walk_forward_scores(
        train,
        train_monthly_history,
        calendar,
    )

    selection, selected_estimator, selected_lambda = (
        diagnostics.select_ewma_lambda_from_training(training_scores)
    )

    return selection, str(selected_estimator), float(selected_lambda)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    data_dir = diagnostics.find_split_dir()
    train, validation, daily = diagnostics.load_train_validation(data_dir)
    calendar = build_export_calendar(daily)

    train_monthly_history, incomplete_training_months = (
        diagnostics.complete_monthly_returns(train, calendar)
    )
    monthly_history, incomplete_history_months = (
        diagnostics.complete_monthly_returns(daily, calendar)
    )

    if incomplete_training_months:
        warnings.warn(
            "Incomplete training months excluded: "
            f"{incomplete_training_months}",
            stacklevel=2,
        )

    if incomplete_history_months:
        warnings.warn(
            "Incomplete train/validation months excluded: "
            f"{incomplete_history_months}",
            stacklevel=2,
        )

    # Audit lambda selection using training only. The exported value remains
    # fixed at 0.93; validation scores are never used to choose it.
    training_selection, selected_estimator, selected_lambda = (
        get_training_lambda_selection(
            train=train,
            train_monthly_history=train_monthly_history,
            calendar=calendar,
        )
    )

    if not np.isclose(selected_lambda, EWMA_LAMBDA):
        warnings.warn(
            f"The current training-only selection is lambda={selected_lambda:.2f}, "
            f"but this exporter is deliberately producing the requested "
            f"lambda={EWMA_LAMBDA:.2f} candidate.",
            stacklevel=2,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_selection_path = (
        OUTPUT_DIR / "ewma_lambda_training_selection_export_check.csv"
    )
    training_selection.to_csv(training_selection_path, index=False)

    # -----------------------------------------------------------------
    # Historical monthly validation-origin forecasts
    # -----------------------------------------------------------------

    validation_origins = diagnostics.expected_validation_rebalances(
        validation,
        calendar,
    )
    if not validation_origins:
        raise ValueError(
            "No first-XNYS-session validation rebalance dates were found."
        )

    walk_forward_rows, walk_forward_details, walk_forward_exclusions = (
        build_origin_forecasts(
            origins=validation_origins,
            daily=daily,
            monthly_history=monthly_history,
            calendar=calendar,
            origin_type="validation_walk_forward",
        )
    )

    walk_forward_frame = pd.DataFrame(
        walk_forward_rows,
        columns=OUTPUT_COLUMNS,
    )
    if not walk_forward_frame.empty:
        validate_long_covariance_file(
            walk_forward_frame,
            label="validation walk-forward EWMA output",
        )

    if walk_forward_frame.empty:
        raise ValueError(
            "No validation-origin EWMA forecasts could be exported. "
            "See the exclusions output for details."
        )

    # -----------------------------------------------------------------
    # Latest available monthly origin
    # -----------------------------------------------------------------

    latest_complete_month = monthly_history.index.max()
    latest_data_month = daily.index.max().to_period("M")

    if latest_complete_month != latest_data_month:
        raise ValueError(
            "The latest observed month is not complete. Refusing to label "
            "a forecast as the next-month origin while partial-month data "
            "are present."
        )

    latest_month_sessions = diagnostics.sessions_for_month(
        calendar,
        latest_complete_month,
    )
    if (
        not len(latest_month_sessions)
        or daily.index.max() != latest_month_sessions[-1]
    ):
        raise ValueError(
            "The last observed date is not the final XNYS session of the "
            "latest complete month."
        )

    latest_forecast_month = latest_complete_month + 1
    latest_origin_sessions = diagnostics.sessions_for_month(
        calendar,
        latest_forecast_month,
    )
    if not len(latest_origin_sessions):
        raise ValueError(
            f"No XNYS sessions found for latest forecast month "
            f"{latest_forecast_month}."
        )

    latest_rebalance_date = latest_origin_sessions[0]

    latest_rows, latest_details, latest_exclusions = build_origin_forecasts(
        origins=[latest_rebalance_date],
        daily=daily,
        monthly_history=monthly_history,
        calendar=calendar,
        origin_type="latest_available_origin",
    )

    if latest_exclusions or not latest_rows:
        reasons = "; ".join(
            item["reason"] for item in latest_exclusions
        )
        raise ValueError(
            "Could not create the latest-available-origin forecast. "
            f"{reasons}"
        )

    latest_frame = pd.DataFrame(
        latest_rows,
        columns=OUTPUT_COLUMNS,
    )
    validate_long_covariance_file(
        latest_frame,
        label="latest-origin EWMA output",
    )

    # -----------------------------------------------------------------
    # Write CSVs with the same long-format schema as the HMM file
    # -----------------------------------------------------------------

    walk_forward_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_walk_forward_validation.csv"
    )
    latest_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_latest_available_origin.csv"
    )
    origin_details_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_walk_forward_origin_details.csv"
    )
    coverage_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_coverage.csv"
    )
    exclusions_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_exclusions.csv"
    )
    metadata_path = (
        OUTPUT_DIR
        / "covariances_ewma_lambda_0_93_metadata.json"
    )

    walk_forward_frame.to_csv(
        walk_forward_path,
        index=False,
        float_format="%.16g",
    )
    latest_frame.to_csv(
        latest_path,
        index=False,
        float_format="%.16g",
    )

    pd.DataFrame(walk_forward_details).to_csv(
        origin_details_path,
        index=False,
    )

    exclusions = pd.DataFrame(
        walk_forward_exclusions + latest_exclusions,
        columns=[
            "origin_type",
            "rebalance_date",
            "origin_month",
            "reason",
        ],
    )
    exclusions.to_csv(exclusions_path, index=False)

    coverage = pd.DataFrame([{
        "train_first_date": train.index.min().strftime("%Y-%m-%d"),
        "train_last_date": train.index.max().strftime("%Y-%m-%d"),
        "validation_first_date": validation.index.min().strftime("%Y-%m-%d"),
        "validation_last_date": validation.index.max().strftime("%Y-%m-%d"),
        "validation_origins_requested": int(len(validation_origins)),
        "validation_origins_exported": int(len(walk_forward_details)),
        "validation_origins_excluded": int(
            len(walk_forward_exclusions)
        ),
        "forecast_horizons": ";".join(
            str(horizon) for horizon in FORECAST_HORIZONS
        ),
        "validation_covariance_matrices_exported": int(
            len(walk_forward_details) * len(FORECAST_HORIZONS)
        ),
        "validation_long_format_rows": int(len(walk_forward_frame)),
        "latest_rebalance_date": latest_rebalance_date.strftime("%Y-%m-%d"),
        "latest_information_date": latest_details[0]["information_date"],
        "latest_forecast_months": ";".join(
            latest_details[0]["forecast_months"]
        ),
        "latest_long_format_rows": int(len(latest_frame)),
        "ewma_lambda_exported": EWMA_LAMBDA,
        "training_selected_lambda": selected_lambda,
        "training_selected_estimator": selected_estimator,
        "estimation_window_months": ESTIMATION_WINDOW_MONTHS,
        "horizon_covariance_assumption": (
            "At each origin, the same covariance matrix is carried forward "
            "for horizons 1-3."
        ),
        "test_file_read": "no",
        "hmm_script_modified": "no",
    }])
    coverage.to_csv(coverage_path, index=False)

    metadata = {
        "estimator": (
            f"ewma_cov_60m_lambda_{EWMA_LAMBDA:.2f}".replace(".", "_")
        ),
        "ewma_lambda": float(EWMA_LAMBDA),
        "estimation_window_months": int(ESTIMATION_WINDOW_MONTHS),
        "forecast_horizons": list(FORECAST_HORIZONS),
        "asset_order": list(ASSETS),
        "monthly_return_definition": (
            "Compounded monthly simple returns for complete common XNYS "
            "trading months."
        ),
        "covariance_units": "monthly decimal-return squared",
        "estimation_method": (
            "Demeaned exponentially weighted sample covariance using the "
            "same ewma_sample_covariance implementation as "
            "covariance_diagnostics.py."
        ),
        "walk_forward_rule": (
            "At each monthly rebalance, use only the trailing 60 complete "
            "monthly return vectors strictly before the rebalance month."
        ),
        "month_ahead_convention": (
            "month_ahead=1 is the rebalance month; 2 and 3 are the next "
            "two calendar months."
        ),
        "horizon_covariance_assumption": (
            "A static EWMA estimate does not model future changes in "
            "volatility or correlation. The origin covariance matrix is "
            "therefore carried forward unchanged for horizons 1-3."
        ),
        "not_cumulative_covariance": (
            "Each matrix is the covariance of an individual forecast "
            "month's return vector. It is not the covariance of cumulative "
            "returns over 2 or 3 months."
        ),
        "cross_month_covariance_included": False,
        "training_lambda_selection": {
            "selected_lambda": float(selected_lambda),
            "selected_estimator": selected_estimator,
            "selection_criterion": (
                "Lowest mean multivariate Gaussian NLL on common "
                "training-only rolling-origin forecasts."
            ),
            "validation_used_to_select_lambda": False,
        },
        "walk_forward_validation_file": walk_forward_path.name,
        "latest_available_origin_file": latest_path.name,
        "latest_available_origin": latest_details[0],
        "test_file_read": False,
        "hmm_script_modified": False,
        "hmm_covariance_file_read": False,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Train/validation returns read from: {data_dir}")
    print(
        f"Training-only lambda selection: {selected_lambda:.2f} "
        f"({selected_estimator})"
    )
    print(f"EWMA lambda exported: {EWMA_LAMBDA:.2f}")
    print(
        f"Validation origins exported: {len(walk_forward_details)} "
        f"of {len(validation_origins)}"
    )
    print(
        "Validation covariance matrices: "
        f"{len(walk_forward_details) * len(FORECAST_HORIZONS)}"
    )
    print(f"Validation output rows: {len(walk_forward_frame)}")
    print(
        "Latest available origin: "
        f"{latest_rebalance_date:%Y-%m-%d}, information through "
        f"{latest_details[0]['information_date']}"
    )
    print(
        "Latest forecast months: "
        f"{', '.join(latest_details[0]['forecast_months'])}"
    )

    print("\nLATEST-ORIGIN OUTPUT:")
    print(latest_frame.to_string(index=False))

    print("\nFILES WRITTEN:")
    print(f"  Walk-forward validation: {walk_forward_path}")
    print(f"  Latest available origin: {latest_path}")
    print(f"  Origin details: {origin_details_path}")
    print(f"  Coverage: {coverage_path}")
    print(f"  Exclusions: {exclusions_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Training lambda audit: {training_selection_path}")
    print("\nTest file read: no. HMM script modified: no.")


if __name__ == "__main__":
    main()