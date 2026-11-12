"""Training-tuned jump forecasts. Run: python jump_regime_model_tuned.py.

Keep the original jump model as a reference. Write five optimizer inputs to
jump_tuned_outputs/. Select settings using training returns only; never test data.
"""
from functools import partial
import jump_regime_model as j

h, np = j.h, j.np
PENALTIES = (10., 50., 150.)
MEAN_POOLING = (0., .5, 1.)
TUNING_MONTHS = 24
OUTPUT = h.HERE / "jump_tuned_outputs"


def fit_jump(history, seed=0, *, penalty=50., pooling=0.):
    """Pool noisy regime means; keep regime-specific covariance and transitions."""
    model, p = j.fit_jump(history, seed, jump_penalty=penalty)
    target = history.mean().to_numpy() * h.SCALE
    model.means_ = (1-pooling)*model.means_ + pooling*target
    return model, p


def select_settings(train, calendar):
    """Score 1-3 month forecasts on chronological folds entirely within training."""
    scores = {(penalty, pooling): [] for penalty in PENALTIES for pooling in MEAN_POOLING}
    months = train.index.to_period("M").unique()[-TUNING_MONTHS:]
    for i, month in enumerate(months):
        date = train.index[train.index.to_period("M") == month][0]
        history = train.loc[train.index < date]
        sessions = h.forecast_sessions(calendar, date)
        lengths = [len(dates) for dates in sessions]
        target = history.mean().to_numpy() * h.SCALE
        for penalty in PENALTIES:
            model, p = j.fit_jump(history, seed=i, jump_penalty=penalty)
            state_means = model.means_.copy()
            for pooling in MEAN_POOLING:
                model.means_ = (1-pooling)*state_means + pooling*target
                _, means, covs = h.forecast_moments_exact(model, p, lengths)
                for k, dates in enumerate(sessions):
                    # Exclude horizons extending beyond the training split.
                    if not dates.isin(train.index).all():
                        continue
                    actual = (1+train.loc[dates]).prod().to_numpy()-1
                    variance = np.diag(covs[k])
                    reference = history.to_numpy().var(axis=0)*lengths[k]
                    # Marginal Gaussian quasi-score balances mean errors and risk:
                    # too-small variances are penalized when actual errors are large.
                    score = .5*np.mean(np.log(variance/reference)
                                       + (actual-means[k])**2/variance)
                    scores[penalty, pooling].append(score)
        print(f"Training fold {i+1}/{len(months)}: {date.date()}", flush=True)
    best = min(scores, key=lambda settings: np.mean(scores[settings]))
    print(f"Selected penalty={best[0]:g}, mean pooling={best[1]:g}; "
          f"training score={np.mean(scores[best]):.4f}; "
          f"original={np.mean(scores[50., 0.]):.4f}")
    return best


def main():
    train, returns, validation, _ = h.load_returns(h.HERE)
    calendar = h.make_calendar(returns, validation)
    penalty, pooling = select_settings(train, calendar)
    # Freeze settings before validation; refit using only history at each origin.
    j.main(fit_model=partial(fit_jump, penalty=penalty, pooling=pooling), output=OUTPUT)


if __name__ == "__main__":
    main()
