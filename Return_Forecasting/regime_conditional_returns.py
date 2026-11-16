"""Create HMM state-conditional monthly expected-return forecasts.

This is a separate script. It imports, but does not modify, the team's
hmm_regime_model.py. It reads training and validation data only—never test.

Bull/bear mean the HMM's lower/higher-equity-variance states.
Returns are compounded monthly simple returns, in decimals.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HMM_DIR = ROOT / "HMM_regime_forecasts"

# Import the team's functions without editing or running its main script.
sys.path.insert(0, str(HMM_DIR))
import hmm_regime_model as hmm  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def conditional_monthly_means(model, p_asof, day_counts):
    """Return expected monthly returns conditional on each month's end state.

    The HMM may change state on every trading day. For each forecast month,
    this calculates E[compounded monthly return | bull at month-end] and
    E[compounded monthly return | bear at month-end], plus the probabilities
    of those ending states.
    """
    transition = np.asarray(model.transmat_, dtype=float)
    daily_gross_means = 1.0 + np.asarray(model.means_, dtype=float) / hmm.SCALE

    p = np.asarray(p_asof, dtype=float).copy()
    results = []

    for n_days in day_counts:
        if n_days < 1:
            raise ValueError("Every forecast month must have at least one session.")

        # Before the first future trading day, cumulative gross return is 1.
        # Rows correspond to the current state; columns correspond to assets.
        weighted_gross = p[:, None].copy()

        for _ in range(n_days):
            # Move to the next day's state, then multiply by its expected
            # daily gross return.
            weighted_gross = (transition.T @ weighted_gross) * daily_gross_means
            p = p @ transition

        # weighted_gross[s, a] is E[gross return of asset a * 1(end state=s)].
        # Divide by P(end state=s) to get the conditional expected gross return.
        conditional_gross = np.ones_like(weighted_gross)
        np.divide(
            weighted_gross,
            p[:, None],
            out=conditional_gross,
            where=p[:, None] > 1e-14,
        )
        conditional_returns = conditional_gross - 1.0

        # Mixing the conditional forecasts by their ending-state probabilities
        # should recover the ordinary, probability-weighted HMM forecast.
        probability_weighted_mean = p @ conditional_returns

        results.append((
            p.copy(),
            conditional_returns.copy(),
            probability_weighted_mean.copy(),
        ))

    return results


def main():
    # The team's loader reads train and validation only.
    _, returns, validation, assets = hmm.load_returns(HMM_DIR)

    # Extend the exchange calendar far enough to forecast three months after
    # the final validation month. This uses calendar dates, not future returns.
    next_month = validation.index[-1].to_period("M") + 1
    final_forecast_month = next_month + (hmm.HORIZON - 1)
    calendar = hmm.xcals.get_calendar(
        hmm.CALENDAR,
        start=str(returns.index.min().date()),
        end=str(final_forecast_month.end_time.normalize().date()),
    )

    # Recreate the HMM validation rebalance dates.
    validation_months = validation.index.to_period("M").unique()
    origins = []
    for month in validation_months:
        date = validation.index[validation.index.to_period("M") == month][0]
        origins.append((date, "validation"))

    # Also add one forecast origin immediately after validation. Its model
    # history ends with validation; no test returns are read or used.
    next_month_sessions = calendar.sessions_in_range(
        next_month.start_time.normalize(),
        next_month.end_time.normalize(),
    )
    if not len(next_month_sessions):
        raise ValueError(f"No exchange sessions found for {next_month}.")

    forward_date = pd.Timestamp(next_month_sessions[0])
    if forward_date.tz is not None:
        forward_date = forward_date.tz_localize(None)
    forward_date = forward_date.normalize()
    origins.append((forward_date, "post_validation_forward"))

    rows = []

    for origin_number, (rebalance_date, origin_type) in enumerate(origins):
        # Strictly exclude the rebalance day's return and everything after it.
        history = returns.loc[returns.index < rebalance_date]
        if history.empty:
            raise ValueError(f"No past returns available for {rebalance_date.date()}.")

        model = hmm.fit_hmm(history.to_numpy(), seed=origin_number)
        p_asof = hmm.filtered_state_probabilities(
            model, history.to_numpy()
        )[-1]

        sessions_by_month = hmm.forecast_sessions(calendar, rebalance_date)
        day_counts = [len(sessions) for sessions in sessions_by_month]

        # The original HMM's ordinary forecasts provide a consistency check.
        _, unconditional_means, _ = hmm.forecast_moments_exact(
            model, p_asof, day_counts
        )
        conditional = conditional_monthly_means(
            model, p_asof, day_counts
        )

        for k, (sessions, state_result) in enumerate(
            zip(sessions_by_month, conditional)
        ):
            p_end, state_means, mixture = state_result

            if not np.allclose(
                mixture,
                unconditional_means[k],
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ArithmeticError(
                    "State-conditional means do not reproduce the "
                    "probability-weighted HMM forecast."
                )

            row = {
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "information_date": history.index[-1].strftime("%Y-%m-%d"),
                "origin_type": origin_type,
                "month_ahead": k + 1,
                "forecast_month": str(sessions[0].to_period("M")),
                "p_bull_month_end": float(p_end[0]),
                "p_bear_month_end": float(p_end[1]),
            }

            for j, asset in enumerate(assets):
                row[f"{asset}_conditional_on_bull_end"] = float(state_means[0, j])
                row[f"{asset}_conditional_on_bear_end"] = float(state_means[1, j])
                row[f"{asset}_probability_weighted"] = float(mixture[j])

            rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "hmm_regime_conditional_returns.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Returns are monthly simple returns in decimals.")
    print("Bull/bear are lower/higher equity-variance states.")
    print("Test data was not read.")


if __name__ == "__main__":
    main()