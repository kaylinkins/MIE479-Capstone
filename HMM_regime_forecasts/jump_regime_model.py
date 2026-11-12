"""Statistical jump-model alternative. Run: python jump_regime_model.py.

Paper-inspired common regime, NOT the paper's asset-specific JM + classifier
or sparse JM + Black-Litterman. No test data or diagnostic files are used.
Outputs have the HMM's five-file schema, in jump_outputs/.
"""
from types import SimpleNamespace
import hmm_regime_model as h
from jumpmodels.jump import JumpModel # from Yizhan Shu's reposittory, rememebr to site!
import numpy as np
import pandas as pd

OUTPUT = h.HERE / "jump_outputs"
JUMP_PENALTY = 50.0  # fixed baseline; not selected using validation outcomes


def features(returns):
    """Causal return/downside-risk features, inspired by Shu, Yu and Mulvey."""
    columns = []
    for asset in returns:
        r = returns[asset]
        for half_life in (5, 20, 60):
            mean = r.ewm(halflife=half_life).mean()
            downside = r.clip(upper=0).pow(2).ewm(halflife=half_life).mean()
            downside = downside.pow(.5).clip(lower=1e-8)
            columns.extend([mean, np.log(downside), mean / downside])
    # Discard initialization; no centered windows or future observations.
    return pd.concat(columns, axis=1).iloc[60:]


def fit_jump(history, seed=0):
    """Fit two persistent states to past features; estimate return moments separately."""
    raw = features(history)
    # Fit clipping/scaling again using ONLY this rebalance's available history.
    center, scale = raw.mean(), raw.std().clip(lower=1e-8)
    clipped = raw.clip(center - 3*scale, center + 3*scale, axis=1)
    X = ((clipped - clipped.mean()) / clipped.std().clip(lower=1e-8)).to_numpy()
    jm = JumpModel(n_components=2, jump_penalty=JUMP_PENALTY,
                   cont=False, n_init=5, max_iter=300, random_state=h.SEED+seed)
    # Minimize squared feature distances plus a penalty for each state change.
    # Historical fitted labels can use all available PAST data, never future returns.
    jm.fit(X, sort_by=None)
    labels = np.asarray(jm.labels_, dtype=int)
    current = int(np.asarray(jm.predict_online(X))[-1])
    R = history.loc[raw.index].to_numpy()
    # Match HMM naming: state 0 lower equity variance; state 1 higher variance.
    order = np.argsort([R[labels == s, 0].var() for s in range(2)])
    mapping = np.argsort(order)
    labels, current = mapping[labels], mapping[current]
    target = history.to_numpy().mean(axis=0)
    prior = np.diag(history.to_numpy().var(axis=0))
    means, covariances = [], []
    for s in range(2):
        state_returns = R[labels == s]
        n = len(state_returns)
        mean = (state_returns.sum(axis=0) + h.MEAN_PRIOR_DAYS*target) / (n+h.MEAN_PRIOR_DAYS)
        centered = state_returns - mean
        # Same historical-mean and diagonal-covariance shrinkage targets as HMM.
        covariance = (centered.T@centered + h.SHRINK*prior
                      + h.MEAN_PRIOR_DAYS*np.outer(mean-target, mean-target)) / (n+h.SHRINK)
        means.append(mean)
        covariances.append(covariance)
    # JM assignments are NOT probabilities. An explicit empirical Markov extension
    # supplies future regime probabilities; one count per cell prevents zero rates.
    counts = np.ones((2, 2))
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    model = SimpleNamespace(transmat_=counts/counts.sum(axis=1, keepdims=True),
                            means_=np.array(means)*h.SCALE,
                            covars_=np.array(covariances)*h.SCALE**2)
    return model, np.eye(2)[current]


def main():
    _, returns, validation, assets = h.load_returns(h.HERE)
    calendar = h.make_calendar(returns, validation)
    probabilities, means_out, cov_out, schedule, scenarios = [], [], [], [], {}
    months = validation.index.to_period("M").unique()
    for i, month in enumerate(months):
        date = validation.index[validation.index.to_period("M") == month][0]
        history = returns.loc[returns.index < date]
        model, p = fit_jump(history, seed=i)
        sessions = h.forecast_sessions(calendar, date)
        lengths = [len(dates) for dates in sessions]
        # Reuse identical compounding, calendar, daily switching and Gaussian
        # scenario machinery so the change being compared is regime estimation.
        bear, means, covariances = h.forecast_moments_exact(model, p, lengths)
        paths = h.simulate_scenarios(model, p, lengths, np.random.default_rng(h.SEED+i))
        keydate = str(date.date())
        scenarios[keydate] = paths.astype(np.float32)
        probabilities.append(dict(rebalance_date=keydate,
            information_date=str(history.index[-1].date()), p_bear_today=p[1],
            regime_today="bear" if p[1] > .5 else "bull",
            **{f"p_bear_month{k+1}": bear[k] for k in range(h.HORIZON)},
            **{f"p_bear_month{k+1}_end":
               (p @ np.linalg.matrix_power(model.transmat_, sum(lengths[:k+1])))[1]
               for k in range(h.HORIZON)}))
        for k, dates in enumerate(sessions):
            key = dict(rebalance_date=keydate, month_ahead=k+1,
                       forecast_month=str(dates[0].to_period("M")))
            means_out.append({**key, **dict(zip(assets, means[k]))})
            for j, asset in enumerate(assets):
                cov_out.append({**key, "asset": asset, **dict(zip(assets, covariances[k][j]))})
            schedule.append({**key, "first_trading_date": str(dates[0].date()),
                "last_trading_date": str(dates[-1].date()), "trading_days": len(dates)})
        print(f"{i+1}/{len(months)} {keydate}: forecasts complete", flush=True)
    OUTPUT.mkdir(exist_ok=True)
    pd.DataFrame(probabilities).to_csv(OUTPUT/"regime_probabilities.csv", index=False)
    pd.DataFrame(means_out).to_csv(OUTPUT/"expected_returns.csv", index=False)
    pd.DataFrame(cov_out).to_csv(OUTPUT/"covariances.csv", index=False)
    pd.DataFrame(schedule).to_csv(OUTPUT/"forecast_trading_days.csv", index=False)
    np.savez_compressed(OUTPUT/"cvar_scenarios.npz", **scenarios)
    print("Saved five team handoff files to", OUTPUT)


if __name__ == "__main__":
    main()
