"""Two-state, daily regime-switching HMM forecasts for MVO and CVaR MPO.

Requires: numpy, pandas, scipy, hmmlearn, exchange_calendars.
    pip install numpy pandas scipy hmmlearn exchange_calendars

Reads portfolio_train.csv and portfolio_validation.csv from ../Data. In test mode,
portfolio_test.csv is also required.

A monthly forecast is the simple *compounded* return from the rebalancing
session through that month's final NYSE session. Following months use all
NYSE sessions. BTC, bond, and equity data MUST be aligned to this calendar.

MVO: expected_returns.csv / covariances.csv contain EXACT conditional
moments (within floating-point precision) under the fitted daily Markov-
switching Gaussian simple-return model. We propagate first and second
moments through every trading session, allowing the regime to change daily.

CVaR: cvar_scenarios.npz contains JOINT 3-month simulated scenario paths
from the SAME fitted model. Sampling error means scenario sample moments
will be close to, not exactly equal to, the analytic MVO moments.

Both monthly moments and scenarios use the true number of NYSE sessions,
not 21 or the number of rows of future (unobserved) returns.

Outputs are under hmm_outputs/ or hmm_outputs_test/:
  regime_probabilities.csv, expected_returns.csv,
  expected_returns_historical.csv, covariances.csv,
  covariances_historical.csv, horizon_joint_covariances.csv,
  cvar_scenarios.npz,
  forecast_trading_days.csv, moment_consistency.csv,
  daily_bear_probability.csv, regime_summary.csv,
  forecast_evaluation.csv.

Note: the HMM operates on daily SIMPLE returns. It is not a log-return HMM.
'Bull'/'bear' denote low/high equity *variance*, not necessarily positive/
negative equity expected returns. Training-period daily state estimates are
in-sample (parameters estimated on all training data); validation/test
state forecasts are out-of-sample, with expanding-window model refits.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# ----------------------------- SETTINGS -----------------------------
PERIOD = "validation"       # "test" only after all settings are frozen
CALENDAR = "XNYS"           # NYSE sessions; all 3 asset returns must align
N_STARTS = 10
SHRINK = 250                # covariance prior strength (original settings)
STICKY = 500                # extra diagonal transition-prior counts
HORIZON = 3                 # forecast months
N_SCENARIOS = 5000
SCALE = 100                 # model daily returns in percent
SEED = 42
ASSETS_EXPECTED = ("Equity", "Bond", "BTC")

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# ------------------------------ INPUT -------------------------------
def load_returns(folder: Path):
    if PERIOD not in ("validation", "test"):
        raise ValueError("PERIOD must be 'validation' or 'test'")

    def read(filename: str) -> pd.DataFrame:
        frame = pd.read_csv(folder / filename, index_col=0, parse_dates=True)
        if frame.empty:
            raise ValueError(f"{filename}: empty data")
        if set(frame.columns) != set(ASSETS_EXPECTED):
            raise ValueError(
                f"{filename}: expected columns {ASSETS_EXPECTED}, got {tuple(frame.columns)}"
            )
        frame = frame.loc[:, ASSETS_EXPECTED]
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame = frame.sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"{filename}: duplicated dates")
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: missing/non-finite returns")
        if (frame.to_numpy(dtype=float) <= -1).any():
            raise ValueError(f"{filename}: simple returns must be greater than -100%")
        return frame.astype(float)

    train = read("portfolio_train.csv")
    validation = read("portfolio_validation.csv")
    chunks = [train, validation]
    oos = validation
    if PERIOD == "test":
        test = read("portfolio_test.csv")
        chunks.append(test)
        oos = test
    if not train.index.max() < validation.index.min():
        raise ValueError("Training and validation dates overlap or are misordered")
    if PERIOD == "test" and not validation.index.max() < oos.index.min():
        raise ValueError("Validation and test dates overlap or are misordered")

    returns = pd.concat(chunks).sort_index()
    if not returns.index.is_unique:
        raise ValueError("Overlapping split dates")
    return train, returns, oos, list(ASSETS_EXPECTED)


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
        raise ValueError(
            f"Found {len(unexpected)} return dates outside {CALENDAR} sessions "
            f"(e.g. {list(unexpected[:5].date)}). Align the asset calendar first."
        )
    missing = expected_past.difference(returns.index)
    if len(missing):
        warnings.warn(
            f"Missing {len(missing)} {CALENDAR} sessions in combined return "
            f"data (e.g. {list(missing[:5].date)}). An HMM transition "
            "currently occurs per observed return row; investigate missing "
            "sessions before relying on estimates.",
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
        if not len(dates):
            raise ValueError(f"No forecast sessions for {month} as of {rebalance_date}")
        out.append(dates)
    if out[0][0] != rebalance_date.normalize():
        raise ValueError("Rebalance date is not an exchange trading session")
    return out


# ------------------------- HMM FIT/FILTER --------------------------
def fit_hmm(R: np.ndarray, *, seed: int = 0) -> GaussianHMM:
    """Retain original regularization and 10-start heuristic; states re-ordered."""
    X = np.asarray(R, dtype=float) * SCALE
    prior = SHRINK * np.diag(X.var(axis=0))
    best, best_score = None, -np.inf
    for start in range(N_STARTS):
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            covars_prior=np.array([prior, prior]),
            covars_weight=SHRINK + 3,
            transmat_prior=1 + STICKY * np.eye(2),
            min_covar=1e-6,
            n_iter=1000,
            random_state=seed * 100 + start,
        )
        model.fit(X)
        # Same heuristic prior-aware start ranking as the original code.
        # Do not interpret this as a fully normalized marginal likelihood.
        try:
            score = model.score(X) + sum(
                STICKY * np.log(max(model.transmat_[s, s], 1e-300))
                - 0.5 * SHRINK * np.linalg.slogdet(model.covars_[s])[1]
                - 0.5 * np.trace(prior @ np.linalg.inv(model.covars_[s]))
                for s in range(2)
            )
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(score) and (best is None or score > best_score):
            best, best_score = model, score
    if best is None:
        raise RuntimeError("Every HMM start failed; check conditioning and priors")

    # State 0 = lower equity variance; 1 = higher equity variance.
    order = np.argsort(best.covars_[:, 0, 0])
    best.startprob_ = best.startprob_[order]
    best.transmat_ = best.transmat_[np.ix_(order, order)]
    best.means_ = best.means_[order]
    best.covars_ = best.covars_[order]
    return best


def filtered_state_probabilities(model, R: np.ndarray) -> np.ndarray:
    """Forward-filter state posteriors P(S_t|r_1,...,r_t), NOT smoothed.

    The last posterior equals hmmlearn.predict_proba(R)[-1] up to numerical
    precision. Intermediate posteriors are calculated without future data.
    """
    X = np.asarray(R, dtype=float) * SCALE
    log_lik = np.column_stack([
        multivariate_normal.logpdf(
            X, mean=model.means_[s], cov=model.covars_[s], allow_singular=False
        )
        for s in range(2)
    ])
    log_P = np.log(np.maximum(model.transmat_, 1e-300))
    log_state = np.log(np.maximum(model.startprob_, 1e-300))
    result = np.empty((len(X), 2), dtype=float)
    for t in range(len(X)):
        if t:
            log_state = logsumexp(log_state[:, None] + log_P, axis=0)
        log_state = log_state + log_lik[t]
        log_state -= logsumexp(log_state)
        result[t] = np.exp(log_state)
    return result


def regime_summary(model, assets: list[str]) -> pd.DataFrame:
    result = {}
    for s, name in enumerate(("bull", "bear")):
        daily_mu = model.means_[s] / SCALE
        covariance = model.covars_[s] / SCALE**2
        vol = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(vol, vol)
        stay = model.transmat_[s, s]
        result[name] = {
            **{f"{a} return (ann. arithmetic)": daily_mu[j] * 252
               for j, a in enumerate(assets)},
            **{f"{a} volatility (ann.)": vol[j] * np.sqrt(252)
               for j, a in enumerate(assets)},
            "Corr Equity-Bond": corr[0, 1],
            "Corr Equity-BTC": corr[0, 2],
            "Corr Bond-BTC": corr[1, 2],
            "Daily prob. of staying": stay,
            "Expected length (sessions)": 1 / (1 - stay) if stay < 1 else np.inf,
        }
    return pd.DataFrame(result)


# ------------------------ EXACT MVO MOMENTS ------------------------
def forecast_moments_exact(
    model, p_asof: np.ndarray, day_counts: list[int] | tuple[int, ...]
):
    """Exact conditional moments of compounded monthly simple returns.

    Each day: S_d follows model.transmat_; conditional on S_d=s,
    r_d ~ N(mu_s, Sigma_s), independent of other days conditional on states.
    Recursively propagate state-weighted 1st/2nd moments of gross return
    products; thus covariance includes BOTH within-day shocks and
    cross-day dependence induced by persistent/changing regimes.

    Returns: avg daily bear probs, mean vectors, covariance matrices.
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
        if n_days < 1:
            raise ValueError("Every forecast month must contain >= 1 session")
        # Before this month's first day, the cumulative gross return is 1.
        first = np.broadcast_to(p[:, None], (2, n_assets)).copy()
        second = np.broadcast_to(p[:, None, None], (2, n_assets, n_assets)).copy()
        bear_sum = 0.0

        for _ in range(n_days):
            p = p @ P
            first = (P.T @ first) * gross
            second = np.einsum("ij,iab->jab", P, second) * gross2
            bear_sum += p[1]

        e_gross = first.sum(axis=0)
        e_gross_outer = second.sum(axis=0)
        month_mu = e_gross - 1.0
        month_cov = e_gross_outer - np.outer(e_gross, e_gross)
        month_cov = (month_cov + month_cov.T) / 2
        if np.linalg.eigvalsh(month_cov).min() < -1e-10:
            raise ArithmeticError("Month covariance is not positive semidefinite")

        result_p.append(bear_sum / n_days)
        result_mu.append(month_mu)
        result_cov.append(month_cov)

    return result_p, result_mu, result_cov


def horizon_joint_covariance_exact(model, p_asof, day_counts, monthly_means, monthly_covs):
    """Exact covariance of the stacked HORIZON x assets monthly returns.

    Diagonal blocks are the exact within-month covariance matrices. Off-
    diagonal blocks retain persistence of the hidden Markov state *across*
    month boundaries. Use this for multi-period MVO objectives involving
    terminal/cumulative risk; standalone monthly matrices alone omit it.

    Order is month1: Equity,Bond,BTC; month2: ...; month3: ... .
    """
    P = np.asarray(model.transmat_, dtype=float)
    g = 1 + np.asarray(model.means_, dtype=float) / SCALE
    n_assets = g.shape[1]
    n_months = len(day_counts)
    month_transition = [np.linalg.matrix_power(P, n) for n in day_counts]

    # p_start[k]: state probabilities just BEFORE the first day of month k.
    p_start = []
    p = np.asarray(p_asof, dtype=float).copy()
    for T in month_transition:
        p_start.append(p.copy())
        p = p @ T

    # B[k][a][s,t] = E[monthly gross return of asset a * indicator(S_end=t)
    #                  | S_start=s]; one daily step is P @ diag(1+mu_a).
    B = [
        [np.linalg.matrix_power(P @ np.diag(g[:, a]), n)
         for a in range(n_assets)]
        for n in day_counts
    ]
    one = np.ones(P.shape[0])
    joint = np.empty((n_months * n_assets, n_months * n_assets))
    for i in range(n_months):
        ix = slice(i * n_assets, (i + 1) * n_assets)
        joint[ix, ix] = monthly_covs[i]
        for j in range(i + 1, n_months):
            # Unweighted regime evolution between months i and j.
            inter = np.eye(P.shape[0])
            for k in range(i + 1, j):
                inter = inter @ month_transition[k]
            block = np.empty((n_assets, n_assets))
            for a in range(n_assets):
                for b in range(n_assets):
                    cross_gross = p_start[i] @ B[i][a] @ inter @ B[j][b] @ one
                    block[a, b] = cross_gross - (
                        (1 + monthly_means[i][a]) * (1 + monthly_means[j][b])
                    )
            jx = slice(j * n_assets, (j + 1) * n_assets)
            joint[ix, jx] = block
            joint[jx, ix] = block.T
    joint = (joint + joint.T) / 2
    if np.linalg.eigvalsh(joint).min() < -1e-10:
        raise ArithmeticError("Joint horizon covariance is not positive semidefinite")
    return joint


def historical_iid_moments(history: np.ndarray, n_days: int):
    """Exactly compounded IID daily empirical-mean/cov historical baseline."""
    mu_d = np.asarray(history).mean(axis=0)
    cov_d = np.cov(history, rowvar=False, bias=True)
    gross = 1 + mu_d
    month_mu = gross**n_days - 1
    second_gross = np.outer(gross, gross) + cov_d
    # Per-day cross-asset dependence retained; days treated as IID.
    month_cov = second_gross**n_days - np.outer(gross**n_days, gross**n_days)
    return month_mu, (month_cov + month_cov.T) / 2


# ------------------------ SHARED CVaR PATHS ------------------------
def simulate_scenarios(model, p_asof, day_counts, rng, n_scenarios=N_SCENARIOS):
    """Daily state changes plus conditional multivariate-Gaussian daily shocks.

    Returns joint array (scenarios, months, assets); state paths remain
    continuous across month boundaries (not sampled independently by month).
    """
    n_days = sum(day_counts)
    n_assets = model.means_.shape[1]
    P = np.asarray(model.transmat_)
    state = np.empty((n_scenarios, n_days), dtype=np.int8)
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
        raise RuntimeError(
            "The Gaussian simple-return HMM generated a return <= -100%; "
            "consider fitting a log-return HMM instead of clipping scenarios."
        )

    monthly = np.empty((n_scenarios, len(day_counts), n_assets), dtype=float)
    left = 0
    for k, count in enumerate(day_counts):
        monthly[:, k, :] = np.prod(1.0 + daily[:, left:left + count, :], axis=1) - 1.0
        left += count
    return monthly


def moment_diagnostic(sim, means, covs, rebalance_date, forecast_months):
    rows = []
    for k, month in enumerate(forecast_months):
        sample_mu = sim[:, k, :].mean(axis=0)
        sample_cov = np.cov(sim[:, k, :], rowvar=False)
        mc_se_mu = np.sqrt(np.maximum(np.diag(covs[k]), 0) / len(sim))
        rows.append({
            "rebalance_date": str(rebalance_date.date()),
            "month_ahead": k + 1,
            "forecast_month": str(month),
            "max_abs_mean_difference": float(np.max(np.abs(sample_mu - means[k]))),
            "max_mean_z_score": float(np.max(
                np.abs(sample_mu - means[k]) / np.maximum(mc_se_mu, 1e-12)
            )),
            "max_abs_cov_difference": float(np.max(np.abs(sample_cov - covs[k]))),
        })
    return rows


def gaussian_nll(realized, mean, covariance):
    """Joint Gaussian moment-score for an observed compounded return vector.

    This is a diagnostic of first/second moment calibration, NOT the true
    log-likelihood of the non-Gaussian HMM return mixture.
    """
    covariance = (covariance + covariance.T) / 2
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Forecast covariance is not positive definite")
    error = realized - mean
    quad = float(error @ np.linalg.solve(covariance, error))
    return 0.5 * (len(error) * np.log(2 * np.pi) + logdet + quad)


# ------------------------------ MAIN -------------------------------
def main():
    folder = Path(__file__).resolve().parent
    output = folder / ("hmm_outputs" if PERIOD == "validation" else "hmm_outputs_test")
    output.mkdir(exist_ok=True)
    train, returns, oos, assets = load_returns(folder.parent / "Data")
    calendar = make_calendar(returns, oos)

    # In-sample depiction: filtered by date, but HMM parameters fitted on
    # the full training set. Do not treat early train dates as OOS estimates.
    train_model = fit_hmm(train.to_numpy())
    summary = regime_summary(train_model, assets)
    train_filtered = filtered_state_probabilities(train_model, train.to_numpy())
    daily_bear = {
        dt: float(p) for dt, p in zip(train.index, train_filtered[:, 1])
    }
    daily_source = {dt: "training_in_sample" for dt in train.index}

    oos_periods = oos.index.to_period("M").unique()
    rebalance_dates = [oos.index[oos.index.to_period("M") == month][0]
                       for month in oos_periods]
    prob_rows, mean_rows, hist_rows, cov_rows, hist_cov_rows = [], [], [], [], []
    schedule_rows, moment_rows, eval_rows, joint_cov_rows = [], [], [], []
    scenarios = {}
    rng = np.random.default_rng(SEED)

    for i, date in enumerate(rebalance_dates):
        history = returns.loc[returns.index < date]
        if history.empty:
            raise ValueError("Cannot fit HMM without past returns")
        model = fit_hmm(history.to_numpy(), seed=i)
        p_bear = float(filtered_state_probabilities(model, history.to_numpy())[-1, 1])
        p_asof = np.array([1 - p_bear, p_bear])

        sessions = forecast_sessions(calendar, date)
        day_counts = [len(x) for x in sessions]
        months = [x[0].to_period("M") for x in sessions]

        # Analytical first/second compounded monthly moments for MVO.
        probs, means, covs = forecast_moments_exact(model, p_asof, day_counts)

        # Exact cross-month covariance, if MVO optimizes *horizon-wide* risk.
        joint_cov = horizon_joint_covariance_exact(
            model, p_asof, day_counts, means, covs
        )
        for r in range(HORIZON * len(assets)):
            for c in range(HORIZON * len(assets)):
                joint_cov_rows.append({
                    "rebalance_date": str(date.date()),
                    "row_month_ahead": r // len(assets) + 1,
                    "row_asset": assets[r % len(assets)],
                    "col_month_ahead": c // len(assets) + 1,
                    "col_asset": assets[c % len(assets)],
                    "covariance": float(joint_cov[r, c]),
                })

        # Continuous daily regimes and shocks for CVaR; SAME fitted model.
        sim = simulate_scenarios(model, p_asof, day_counts, rng)
        scenarios[str(date.date())] = sim.astype(np.float32)
        moment_rows.extend(moment_diagnostic(sim, means, covs, date, months))

        P = model.transmat_
        denom = P[0, 1] + P[1, 0]
        long_run_bear = float(P[0, 1] / denom) if denom > 0 else np.nan

        prob_rows.append({
            "rebalance_date": str(date.date()),
            "information_date": str(history.index[-1].date()),
            "p_bear_today": p_bear,  # as of information_date, BEFORE rebalance
            "regime_today": "bear" if p_bear > 0.5 else "bull",
            **{f"p_bear_month{k + 1}": probs[k] for k in range(HORIZON)},
        })

        for k, (month, dates, n_days) in enumerate(zip(months, sessions, day_counts)):
            key = {
                "rebalance_date": str(date.date()),
                "month_ahead": k + 1,
                "forecast_month": str(month),
            }
            schedule_rows.append({
                **key,
                "first_trading_date": str(dates[0].date()),
                "last_trading_date": str(dates[-1].date()),
                "trading_days": n_days,
                "partial_first_month": bool(k == 0 and dates[0] >
                    calendar.sessions_in_range(month.start_time, min(month.end_time.normalize(), calendar.last_session))[0]),
            })

            mu_hist, cov_hist = historical_iid_moments(history.to_numpy(), n_days)
            mean_rows.append({**key, **dict(zip(assets, means[k]))})
            hist_rows.append({**key, **dict(zip(assets, mu_hist))})
            for name, row in zip(assets, covs[k]):
                cov_rows.append({**key, "asset": name, **dict(zip(assets, row))})
            for name, row in zip(assets, cov_hist):
                hist_cov_rows.append({**key, "asset": name, **dict(zip(assets, row))})

            # Evaluation uses future returns only AFTER forecasting; never
            # included in the refit/forecast at this rebalance date.
            observed_dates = oos.index[oos.index.isin(dates)]
            if len(observed_dates) != len(dates) or not observed_dates.equals(dates):
                continue  # incomplete future month; no misleading score
            realized_month = (1 + oos.loc[dates].to_numpy()).prod(axis=0) - 1
            eval_rows.append({
                **key,
                "trading_days": n_days,
                "mean_mse_hmm": float(np.mean((realized_month - means[k])**2)),
                "mean_mse_historical": float(np.mean((realized_month - mu_hist)**2)),
                "joint_gaussian_nll_hmm": gaussian_nll(realized_month, means[k], covs[k]),
                "joint_gaussian_nll_historical": gaussian_nll(realized_month, mu_hist, cov_hist),
                **{f"squared_error_hmm_{a}": float((realized_month[j]-means[k][j])**2)
                   for j, a in enumerate(assets)},
                **{f"squared_error_historical_{a}": float((realized_month[j]-mu_hist[j])**2)
                   for j, a in enumerate(assets)},
            })

        # Daily out-of-sample filtered state posterior: freeze this month's
        # parameters; each day's probability uses returns through THAT day.
        month_dates = oos.index[oos.index.to_period("M") == date.to_period("M")]
        through = returns.loc[returns.index <= month_dates[-1]]
        filtered = filtered_state_probabilities(model, through.to_numpy())
        where = through.index.get_indexer(month_dates)
        for day, p in zip(month_dates, filtered[where, 1]):
            daily_bear[day] = float(p)
            daily_source[day] = "out_of_sample_walk_forward"

        print(f"{date.date()}  bear(as-of)={p_bear:.3f}  "
              f"day_counts={day_counts}  next 3 P(bear)={np.round(probs, 3)}")

    # Write output in compatible layouts, with separate calendar/diagnostics.
    pd.DataFrame(prob_rows).to_csv(output / "regime_probabilities.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(output / "expected_returns.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(output / "expected_returns_historical.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(output / "covariances.csv", index=False)
    pd.DataFrame(hist_cov_rows).to_csv(output / "covariances_historical.csv", index=False)
    pd.DataFrame(joint_cov_rows).to_csv(output / "horizon_joint_covariances.csv", index=False)
    np.savez_compressed(output / "cvar_scenarios.npz", **scenarios)
    pd.DataFrame(schedule_rows).to_csv(output / "forecast_trading_days.csv", index=False)
    pd.DataFrame(moment_rows).to_csv(output / "moment_consistency.csv", index=False)
    pd.DataFrame({
        "date": sorted(daily_bear),
        "p_bear": [daily_bear[d] for d in sorted(daily_bear)],
        "source": [daily_source[d] for d in sorted(daily_bear)],
    }).to_csv(output / "daily_bear_probability.csv", index=False)
    summary.to_csv(output / "regime_summary.csv")
    evaluation = pd.DataFrame(eval_rows)
    evaluation.to_csv(output / "forecast_evaluation.csv", index=False)

    print("\nREGIMES ON TRAINING DATA:\n", summary.round(4).to_string())
    if len(evaluation):
        cols = ["mean_mse_hmm", "mean_mse_historical",
                "joint_gaussian_nll_hmm", "joint_gaussian_nll_historical"]
        print("\nOUT-OF-SAMPLE MONTHLY SCORES (lower is better):\n",
              evaluation.groupby("month_ahead")[cols].mean().round(6).to_string())
    print(f"\nSaved {len(rebalance_dates)} rebalance dates to {output}")


if __name__ == "__main__":
    main()
