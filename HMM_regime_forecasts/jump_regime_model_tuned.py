"""Training-tuned jump forecasts. Run: python jump_regime_model_tuned.py.

Refresh expected_returns.csv in jump_tuned_outputs/. Preserve the saved risk
and regime inputs. Select settings using training returns only.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import jump_regime_model as j

h, np = j.h, j.np
PENALTIES = (10., 50., 150.)
TUNING_MONTHS = 24
OUTPUT = h.HERE / "jump_tuned_outputs"


def fit_jump(history, seed=0, *, penalty=50.):
    """Fit the shared Jump features with unpooled regime returns."""
    raw = j.features(history)
    center = raw.mean()
    scale = raw.std().clip(lower=1e-8)
    clipped = raw.clip(center - 3 * scale, center + 3 * scale, axis=1)
    features = ((clipped - clipped.mean()) / clipped.std().clip(lower=1e-8)).to_numpy()
    jump = j.JumpModel(n_components=2, jump_penalty=penalty, cont=False,
                       n_init=5, max_iter=300, random_state=h.SEED + seed)
    jump.fit(features, sort_by=None)
    labels = np.asarray(jump.labels_, dtype=int)
    current = int(np.asarray(jump.predict_online(features))[-1])
    returns = history.loc[raw.index].to_numpy()

    # State 1 has the higher equity variance, matching the HMM convention.
    variances = [returns[labels == s, 0].var() if np.any(labels == s)
                 else np.inf for s in range(2)]
    mapping = np.argsort(np.argsort(variances))
    labels, current = mapping[labels], mapping[current]

    prior = np.diag(history.to_numpy().var(axis=0))
    means, covariances = [], []
    for state in range(2):
        state_returns = returns[labels == state]
        n = len(state_returns)
        # Borrow the fitted state's mean if the other state has no observations.
        mean = (state_returns if n else returns).mean(axis=0)
        centered = state_returns - mean
        covariance = (centered.T @ centered + h.SHRINK * prior) / (n + h.SHRINK)
        means.append(mean)
        covariances.append(covariance)

    counts = np.ones((2, 2))
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    model = SimpleNamespace(
        transmat_=counts / counts.sum(axis=1, keepdims=True),
        means_=np.array(means) * h.SCALE,
        covars_=np.array(covariances) * h.SCALE**2,
    )
    return model, np.eye(2)[current]


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
    train, returns, validation, assets = h.load_returns(h.HERE)
    calendar = h.make_calendar(returns, validation)
    penalty = select_settings(train, calendar)
    rows = []
    origins = validation.groupby(validation.index.to_period("M")).head(1).index
    for i, date in enumerate(origins):
        history = returns.loc[returns.index < date]
        model, p = fit_jump(history, seed=i, penalty=penalty)
        sessions = h.forecast_sessions(calendar, date)
        lengths = [len(dates) for dates in sessions]
        _, means, _ = h.forecast_moments_exact(model, p, lengths)
        for k, dates in enumerate(sessions):
            rows.append(dict(rebalance_date=str(date.date()), month_ahead=k + 1,
                             forecast_month=str(dates[0].to_period("M")),
                             **dict(zip(assets, means[k]))))
        print(f"{i + 1}/{len(origins)} {date.date()}: forecasts complete", flush=True)

    OUTPUT.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".jump-forecast-", dir=OUTPUT.parent) as directory:
        staged = Path(directory) / "expected_returns.csv"
        j.pd.DataFrame(rows).to_csv(staged, index=False)
        staged.replace(OUTPUT / "expected_returns.csv")
    print("Updated expected returns in", OUTPUT)


if __name__ == "__main__":
    main()
