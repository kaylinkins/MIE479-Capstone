"""Gaussian HMM forecasts for the team's MVO and CVaR MPO models.

Run: python hmm_regime_model.py
Read portfolio_train.csv and portfolio_validation.csv beside this file.
Write ONLY these files in hmm_outputs/:
  regime_probabilities.csv   current and 1-3 month regime probabilities
  expected_returns.csv      monthly decimal expected returns
  covariances.csv           monthly covariance matrices
  cvar_scenarios.npz         joint paths (scenario, month, asset), keyed by date
  forecast_trading_days.csv  exact NYSE dates covered by each forecast

Asset order: Equity, Bond, BTC. Fit monthly with strictly prior observations.
Regularized means/covariances and daily Markov transitions define both MVO
moments and CVaR scenarios. No portfolio optimization, saved diagnostics or plots.
Mean regularization reduces estimation noise; it does not guarantee accuracy.
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
if (HERE/'.diagnostic_dependencies').exists():
    sys.path.insert(0,str(HERE/'.diagnostic_dependencies'))
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

CALENDAR="XNYS"
ASSETS_EXPECTED=("Equity","Bond","BTC")
N_STARTS=10
SHRINK=250
STICKY=500
MEAN_PRIOR_DAYS=252  # stabilize means with one year of historical-mean prior weight
HORIZON=3
N_SCENARIOS=10000
SCALE=100
SEED=42
OUTPUT=HERE/'hmm_outputs'
logging.getLogger('hmmlearn').setLevel(logging.ERROR)


# 1. Observations: daily simple returns on the same dates for all three assets.
def load_returns(folder: Path):
    """Read the fixed training/validation splits; the test split is never opened."""
    splits = []
    for name in ("train", "validation"):
        frame = pd.read_csv(folder / f"portfolio_{name}.csv", index_col=0,
                            parse_dates=True)
        if frame.empty or set(frame.columns) != set(ASSETS_EXPECTED):
            raise ValueError(f"{name}: expected nonempty Equity, Bond, BTC returns")
        frame = frame.loc[:, list(ASSETS_EXPECTED)].astype(float)
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame = frame.sort_index()
        if frame.index.hasnans or not frame.index.is_unique:
            raise ValueError(f"{name}: invalid or duplicated dates")
        if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy() <= -1).any():
            raise ValueError(f"{name}: returns must be finite and greater than -100%")
        if splits and splits[-1].index[-1] >= frame.index[0]:
            raise ValueError("Training and validation dates overlap")
        splits.append(frame)
    return splits[0], pd.concat(splits), splits[-1], list(ASSETS_EXPECTED)


# A transition represents one trading session. NYSE dates, rather than a fixed
# 21-day approximation, determine how many transitions belong to each month.
def make_calendar(returns: pd.DataFrame, oos: pd.DataFrame):
    first_day = returns.index.min()
    last_forecast_month = oos.index[-1].to_period("M") + (HORIZON - 1)
    end = last_forecast_month.end_time.normalize()
    calendar = xcals.get_calendar(
        CALENDAR,
        start=str(first_day.date()),
        end=str(end.date()),
    )
    # Avoid using future price observations to determine the month lengths.
    expected_past = calendar.sessions_in_range(first_day, returns.index.max())
    unexpected = returns.index.difference(expected_past)
    if len(unexpected):
        raise ValueError(f"Return data contains dates outside the {CALENDAR} calendar")
    missing = expected_past.difference(returns.index)
    if len(missing):
        warnings.warn(
            f"Missing {len(missing)} {CALENDAR} sessions: model transitions follow observed rows.",
            stacklevel=2,
        )
    return calendar

def forecast_sessions(calendar, rebalance_date: pd.Timestamp) -> list[pd.DatetimeIndex]:
    """NYSE dates for each month, starting with the first tradable decision day."""
    periods = [rebalance_date.to_period("M") + k for k in range(HORIZON)]
    out = []
    for k, month in enumerate(periods):
        dates = calendar.sessions_in_range(month.start_time, min(month.end_time.normalize(), calendar.last_session))
        if k == 0:
            dates = dates[dates >= rebalance_date.normalize()]
        out.append(dates)
    return out

# 2. Latent-state model: S_t follows a two-state Markov chain, and
# r_t | S_t=s ~ Normal(mu_s, Sigma_s). Full covariance captures asset dependence.
def fit_hmm(R: np.ndarray, *, seed: int = 0) -> GaussianHMM:
    """Fit regularized state means/covariances; state 0 has lower equity variance."""
    # Fit in percentage points for numerical conditioning; forecasts revert to decimals.
    X = np.asarray(R, dtype=float) * SCALE
    # Mean prior: mu_s is pulled toward the historical mean, especially when
    # few observations occupy state s. It does not constrain the sign of returns.
    mean_target = X.mean(axis=0)
    # Covariance prior shrinks toward historical variances with zero correlations.
    # This stabilizes estimation; fitted off-diagonal covariances remain possible.
    prior = SHRINK * np.diag(X.var(axis=0))
    best, best_score = None, -np.inf
    # EM alternates state-membership estimation and parameter updates. Its
    # solution depends on initialization, so compare several random starts.
    for start in range(N_STARTS):
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            means_prior=mean_target,
            means_weight=MEAN_PRIOR_DAYS,
            covars_prior=np.array([prior, prior]),
            covars_weight=SHRINK + 3,
            # Diagonal pseudo-counts favor persistent regimes over rapid switching.
            transmat_prior=1 + STICKY * np.eye(2),
            min_covar=1e-6,
            n_iter=1000,
            random_state=seed * 100 + start,
        )
        model.fit(X)
        # Rank starts by likelihood plus mean/covariance/transition penalties.
        # This regularized score is not a normalized Bayesian model probability.
        try:
            score = model.score(X) + sum(
                STICKY * np.log(max(model.transmat_[s, s], 1e-300))
                - 0.5 * SHRINK * np.linalg.slogdet(model.covars_[s])[1]
                - 0.5 * np.trace(prior @ np.linalg.inv(model.covars_[s]))
                - 0.5 * MEAN_PRIOR_DAYS * ((model.means_[s]-mean_target)
                    @ np.linalg.solve(model.covars_[s], model.means_[s]-mean_target))
                for s in range(2)
            )
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(score) and (best is None or score > best_score):
            best, best_score = model, score
    if best is None:
        raise RuntimeError("Every HMM start failed; check conditioning and priors")

    # State labels are arbitrary after EM. Reorder every state-dependent array
    # together: bull = lower equity variance, bear = higher equity variance.
    # These names do not guarantee positive bull or negative bear mean returns.
    order = np.argsort(best.covars_[:, 0, 0])
    best.startprob_ = best.startprob_[order]
    best.transmat_ = best.transmat_[np.ix_(order, order)]
    best.means_ = best.means_[order]
    best.covars_ = best.covars_[order]
    return best

# 3. Bayesian filtering: use past returns to infer the current hidden state.
def filtered_state_probabilities(model, R: np.ndarray) -> np.ndarray:
    """Forward-filter state posteriors P(S_t|r_1,...,r_t), NOT smoothed.

    The last posterior equals hmmlearn.predict_proba(R)[-1] up to numerical
    precision. Intermediate posteriors are calculated without future data.
    """
    X = np.asarray(R, dtype=float) * SCALE
    # Emission likelihood: how compatible is each observed vector with each state?
    log_lik = np.column_stack([
        multivariate_normal.logpdf(
            X, mean=model.means_[s], cov=model.covars_[s], allow_singular=False
        )
        for s in range(2)
    ])
    # Log-space arithmetic avoids underflow when multiplying many probabilities.
    log_P = np.log(np.maximum(model.transmat_, 1e-300))
    log_state = np.log(np.maximum(model.startprob_, 1e-300))
    result = np.empty((len(X), 2), dtype=float)
    for t in range(len(X)):
        if t:
            # Prediction: p(S_t | data through t-1) = p_{t-1} @ P.
            log_state = logsumexp(log_state[:, None] + log_P, axis=0)
        # Update with today's likelihood, then normalize using Bayes' rule.
        log_state = log_state + log_lik[t]
        log_state -= logsumexp(log_state)
        result[t] = np.exp(log_state)
    return result

# 4. Forecast monthly compounded returns while allowing daily state changes.
def forecast_moments_exact(
    model, p_asof: np.ndarray, day_counts: list[int] | tuple[int, ...]
):
    """Exact conditional monthly moments under the fitted HMM assumptions.

    Gross return is G_a = product_d(1 + r_da); monthly return is G_a - 1.
    Track E[G_a * 1(S=s)] and E[G_a G_b * 1(S=s)] for each ending state.
    This includes uncertainty in both daily shocks and future regime paths.
    """
    P = np.asarray(model.transmat_, dtype=float)
    mu = np.asarray(model.means_, dtype=float) / SCALE
    cov = np.asarray(model.covars_, dtype=float) / SCALE**2
    gross = 1.0 + mu                                      # (state, asset)
    gross2 = cov + gross[:, :, None] * gross[:, None, :]  # (state, a, b)
    n_assets = gross.shape[1]
    p = np.asarray(p_asof, dtype=float).copy()
    result_p, result_mu, result_cov = [], [], []

    for n_days in day_counts:
        # Before this month's first day, the cumulative gross return is 1.
        first = np.broadcast_to(p[:, None], (2, n_assets)).copy()
        second = np.broadcast_to(p[:, None, None], (2, n_assets, n_assets)).copy()
        bear_sum = 0.0

        for _ in range(n_days):
            # Move probability/moments through P before applying each state's
            # daily gross-return moments. Conditional shocks are independent by day.
            p = p @ P
            first = (P.T @ first) * gross
            second = np.einsum("ij,iab->jab", P, second) * gross2
            bear_sum += p[1]

        e_gross = first.sum(axis=0)
        e_gross_outer = second.sum(axis=0)
        month_mu = e_gross - 1.0
        # Cov(G) = E[G G'] - E[G] E[G]'; subtracting 1 leaves covariance unchanged.
        month_cov = e_gross_outer - np.outer(e_gross, e_gross)
        month_cov = (month_cov + month_cov.T) / 2
        if np.linalg.eigvalsh(month_cov).min() < -1e-10:
            raise ArithmeticError("Month covariance is not positive semidefinite")

        result_p.append(bear_sum / n_days)
        result_mu.append(month_mu)
        result_cov.append(month_cov)

    return result_p, result_mu, result_cov

# 5. Monte Carlo approximation of the same distribution, for CVaR tail losses.
def simulate_scenarios(model, p_asof, day_counts, rng, n_scenarios=N_SCENARIOS):
    """Joint monthly paths with continuous daily Gaussian regime switching."""
    n_days = sum(day_counts)
    n_assets = model.means_.shape[1]
    P = np.asarray(model.transmat_)
    state = np.empty((n_scenarios, n_days), dtype=np.int8)
    # First future day uses p_asof @ P, not the last observed day's state directly.
    state[:, 0] = rng.random(n_scenarios) < (np.asarray(p_asof) @ P)[1]
    for d in range(1, n_days):
        state[:, d] = rng.random(n_scenarios) < P[state[:, d - 1], 1]

    daily = np.empty((n_scenarios, n_days, n_assets), dtype=float)
    for s in range(2):
        # Draw only as many conditional returns as occupy this state.
        mask = state == s
        n = int(mask.sum())
        if n:
            daily[mask] = rng.multivariate_normal(
                model.means_[s] / SCALE, model.covars_[s] / SCALE**2, size=n
            )
    if (daily <= -1).any():
        raise RuntimeError("Simulated simple return is <= -100%; scenarios were not clipped")

    # One continuous state path spans all months, preserving cross-month
    # dependence. Compound daily simple returns separately within each month.
    monthly = np.empty((n_scenarios, len(day_counts), n_assets), dtype=float)
    left = 0
    for k, count in enumerate(day_counts):
        monthly[:, k, :] = np.prod(1.0 + daily[:, left:left + count, :], axis=1) - 1.0
        left += count
    return monthly

# 6. Walk forward monthly: fit using data before the decision, then forecast.
def main():
    _,returns,validation,assets=load_returns(HERE)
    calendar=make_calendar(returns,validation)
    months=validation.index.to_period('M').unique()
    probabilities = []
    means_out = []
    covariances_out = []
    schedule = []
    scenarios = {}
    for i,month in enumerate(months):
        date=validation.index[validation.index.to_period('M')==month][0]
        # Expanding window: no return on or after this rebalance enters the fit.
        history=returns.loc[returns.index<date]
        model=fit_hmm(history.to_numpy(),seed=i)
        # Last filtered posterior is the state estimate as of the prior session.
        p=filtered_state_probabilities(model,history.to_numpy())[-1]
        sessions=forecast_sessions(calendar,date)
        lengths=[len(block) for block in sessions]
        bear,means,covariances=forecast_moments_exact(model,p,lengths)
        paths=simulate_scenarios(model,p,lengths,np.random.default_rng(SEED+i),n_scenarios=N_SCENARIOS)
        keydate=str(date.date())
        scenarios[keydate]=paths.astype(np.float32)
        # Monthly probabilities describe average regime occupancy; *_end fields
        # describe the state on the month's final session. They are not identical.
        probabilities.append(dict(rebalance_date=keydate,information_date=str(history.index[-1].date()),
            p_bear_today=p[1],regime_today='bear' if p[1]>.5 else 'bull',
            **{f'p_bear_month{k+1}':bear[k] for k in range(HORIZON)},
            **{f'p_bear_month{k+1}_end':(p@np.linalg.matrix_power(model.transmat_,sum(lengths[:k+1])))[1]
               for k in range(HORIZON)}))
        for k,dates in enumerate(sessions):
            key=dict(rebalance_date=keydate,month_ahead=k+1,forecast_month=str(dates[0].to_period('M')))
            means_out.append({**key,**dict(zip(assets,means[k]))})
            for j,asset in enumerate(assets):
                covariances_out.append({**key,'asset':asset,**dict(zip(assets,covariances[k][j]))})
            schedule.append({**key,'first_trading_date':str(dates[0].date()),
                'last_trading_date':str(dates[-1].date()),'trading_days':len(dates)})
        print(f'{i+1}/{len(months)} {keydate}: forecasts complete',flush=True)
    # Keep the handoff limited to the five agreed files. Analytic MVO moments
    # and simulated CVaR moments can differ slightly due to Monte Carlo error.
    OUTPUT.mkdir(exist_ok=True)
    pd.DataFrame(probabilities).to_csv(OUTPUT/'regime_probabilities.csv',index=False)
    pd.DataFrame(means_out).to_csv(OUTPUT/'expected_returns.csv',index=False)
    pd.DataFrame(covariances_out).to_csv(OUTPUT/'covariances.csv',index=False)
    np.savez_compressed(OUTPUT/'cvar_scenarios.npz',**scenarios)
    pd.DataFrame(schedule).to_csv(OUTPUT/'forecast_trading_days.csv',index=False)
    print('Saved the five team handoff files to',OUTPUT)


if __name__=='__main__':
    main()

