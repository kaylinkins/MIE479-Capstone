"""Training-tuned jump forecasts. Run: python jump_regime_model_tuned.py.

Refresh expected_returns.csv in jump_tuned_outputs/. Preserve the saved risk
and regime inputs. Select settings using training returns only.
"""
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import jump_regime_model as j

h, np = j.h, j.np
PENALTIES = (10., 50., 150.)
TUNING_MONTHS = 24
OUTPUT = h.HERE / "jump_tuned_outputs"


def fit_jump(history, seed=0, *, penalty=50.):
    """Fit state returns for probability-weighted forecasts."""
    return j.fit_jump(history, seed, jump_penalty=penalty, mean_prior_days=0.)


def select_settings(train, calendar):
    """Score 1-3 month forecasts on chronological folds entirely within training."""
    scores = {penalty: [] for penalty in PENALTIES}
    months = train.index.to_period("M").unique()[-TUNING_MONTHS:]
    for i, month in enumerate(months):
        date = train.index[train.index.to_period("M") == month][0]
        history = train.loc[train.index < date]
        sessions = h.forecast_sessions(calendar, date)
        lengths = [len(dates) for dates in sessions]
        for penalty in PENALTIES:
            model, p = fit_jump(history, seed=i, penalty=penalty)
            _, means, covs = h.forecast_moments_exact(model, p, lengths)
            for k, dates in enumerate(sessions):
                # Exclude horizons extending beyond the training split.
                if not dates.isin(train.index).all():
                    continue
                actual = (1+train.loc[dates]).prod().to_numpy()-1
                variance = np.diag(covs[k])
                reference = history.to_numpy().var(axis=0)*lengths[k]
                score = .5*np.mean(np.log(variance/reference)
                                   + (actual-means[k])**2/variance)
                scores[penalty].append(score)
        print(f"Training fold {i+1}/{len(months)}: {date.date()}", flush=True)
    best = min(scores, key=lambda settings: np.mean(scores[settings]))
    print(f"Selected penalty={best:g}; "
          f"training score={np.mean(scores[best]):.4f}; "
          f"penalty 50 score={np.mean(scores[50.]):.4f}")
    return best


def main():
    train, returns, validation, _ = h.load_returns(h.HERE)
    calendar = h.make_calendar(returns, validation)
    penalty = select_settings(train, calendar)
    # Freeze settings before validation; refit using only history at each origin.
    OUTPUT.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".jump-forecast-", dir=OUTPUT.parent) as directory:
        staged = Path(directory)
        j.main(fit_model=partial(fit_jump, penalty=penalty), output=staged)
        (staged / "expected_returns.csv").replace(OUTPUT / "expected_returns.csv")
    print("Updated expected returns in", OUTPUT)


if __name__ == "__main__":
    main()
