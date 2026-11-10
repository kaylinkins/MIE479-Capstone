# ============================================================
# CAPSTONE PROJECT: HMM REGIME DETECTION AND FORECASTING (WP 2.1)
# Equities, Fixed Income, and Bitcoin
# ============================================================
#
# Fits a two-state Gaussian HMM (calm "bull" and turbulent "bear") to the daily
# returns of the three assets. At the start of every validation month it
#   1. refits the HMM on all data before that month (expanding window),
#   2. finds today's probability of the bear regime,
#   3. forecasts regime probabilities, expected returns and the covariance
#      matrix for each of the next 3 months,
#   4. simulates regime-switching return scenarios for the CVaR MPO.
#
# Inputs:  portfolio_train.csv, portfolio_validation.csv (from "Splitting data.py")
# Outputs (hmm_outputs/):
#   regime_probabilities.csv         P(bear) today and for months 1-3, per rebalance date
#   expected_returns.csv             expected monthly return per asset, per date and month ahead
#   expected_returns_historical.csv  same layout, historical mean (alternative for WP 2.4)
#   covariances.csv                  3x3 monthly covariance (3 rows), per date and month ahead
#   cvar_scenarios.npz               per date: array (scenarios, 3 months, 3 assets)
#   daily_bear_probability.csv       bear probability on every day (filtered, no look-ahead)
#   regime_summary.csv               what the two regimes look like on the training data
#   forecast_evaluation.csv          forecast errors vs simple benchmarks (validation)
# Graphs are made separately by make_figures.py.
#
# Notes:
#   - Forecasts are monthly decimal returns, assets in the order Equity, Bond, BTC.
#   - The test set is only read when PERIOD = "test" (outputs go to hmm_outputs_test/).
#   - The transition matrix P is daily, so the regime probability j days ahead
#     is p_today @ P^j, and a month is 21 trading days.
#   - Stabilization (TA's recommendation): covariance shrinkage toward the
#     diagonal of the sample covariance, 10 random starts, and a "sticky" prior
#     that favours staying in the same regime. SHRINK and STICKY were chosen on
#     validation forecast performance.
#   - The state with the higher equity volatility is always "bear".
#   - "Today's" probability uses only data up to yesterday. On the last day of
#     the data, hmmlearn's predict_proba equals the filtered probability.
# ============================================================

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# SETTINGS
PERIOD = "validation" # "validation" for model selection; set to "test" ONCE at the
                      # end, after every setting is frozen (needs portfolio_test.csv)
N_STARTS = 10         # random starting points per fit
SHRINK = 250          # covariance shrinkage strength (pseudo-days)
STICKY = 500          # extra "stay in the same regime" counts
HORIZON = 3           # forecast months
DAYS = 21             # trading days per month
N_SCENARIOS = 5000    # CVaR scenarios per date
SCALE = 100           # fit in percent returns (EM is more stable)
SEED = 42

folder = Path(__file__).resolve().parent
output = folder / ("hmm_outputs" if PERIOD == "validation" else "hmm_outputs_test")
output.mkdir(exist_ok=True)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# STEP 1: LOAD DATA
train = pd.read_csv(folder / "portfolio_train.csv", index_col=0, parse_dates=True)
validation = pd.read_csv(folder / "portfolio_validation.csv", index_col=0, parse_dates=True)
assets = list(train.columns)
if PERIOD == "test":
    test = pd.read_csv(folder / "portfolio_test.csv", index_col=0, parse_dates=True)
    returns = pd.concat([train, validation, test]).sort_index()
    oos = test                                         # out-of-sample period
else:
    returns = pd.concat([train, validation]).sort_index()
    oos = validation


# STEP 2: FIT THE HMM
def fit_hmm(R, seed=0):
    """Best of N_STARTS fits; states sorted so 0 = bull, 1 = bear."""
    X = R * SCALE
    prior = SHRINK * np.diag(X.var(axis=0))           # shrinkage target
    best, best_score = None, -np.inf
    for start in range(N_STARTS):
        model = GaussianHMM(n_components=2,
                            covariance_type="full",
                            covars_prior=np.array([prior, prior]), 
                            covars_weight=SHRINK + 3,
                            transmat_prior=1 + STICKY * np.eye(2),
                            min_covar=1e-6, n_iter=1000, 
                            random_state=seed * 100 + start)
        model.fit(X)
        # Pick the start with the best fit INCLUDING the priors (what EM maximizes);
        # plain likelihood would prefer less sticky, less shrunk solutions.
        score = model.score(X) + sum(
            STICKY * np.log(model.transmat_[s, s])
            - 0.5 * SHRINK * np.linalg.slogdet(model.covars_[s])[1]
            - 0.5 * np.trace(prior @ np.linalg.inv(model.covars_[s])) for s in range(2))
        if best is None or score > best_score:
            best, best_score = model, score
    order = np.argsort(best.covars_[:, 0, 0])          # low equity variance first
    best.startprob_ = best.startprob_[order]
    best.transmat_ = best.transmat_[np.ix_(order, order)]
    best.means_ = best.means_[order]
    best.covars_ = best.covars_[order]
    return best


def bear_prob(model, R):
    """Filtered bear probability on the last day of R. P(bear today) ∝ P(bear | yesterday's info, via the transition matrix) 
    × likelihood of today's returns under bear"""
    return model.predict_proba(R * SCALE)[-1, 1]


# STEP 3: FORECASTS
def forecast(model, p_today):
    """P(bear), mean vector and covariance for each of the next HORIZON months."""
    P = model.transmat_
    mu = model.means_ / SCALE                          # daily, decimal
    cov = model.covars_ / SCALE ** 2
    probs, means, covs = [], [], []
    p = p_today
    for k in range(HORIZON):
        daily_p = []
        for day in range(DAYS):
            p = p @ P
            daily_p.append(p)
        w = np.mean(daily_p, axis=0)                   # share of the month in each regime
        m = DAYS * (w[0] * mu[0] + w[1] * mu[1])
        c = sum(w[s] * (DAYS * cov[s] + DAYS ** 2 * np.outer(mu[s], mu[s])) for s in range(2))
        probs.append(w[1])
        means.append(m)
        covs.append(c - np.outer(m, m))
    return probs, means, covs


def simulate_scenarios(model, p_today, rng):
    """Daily regime paths drawn with P, Gaussian returns in each regime,
    compounded into monthly returns. Shape (N_SCENARIOS, HORIZON, assets)."""
    P = model.transmat_
    n_days = HORIZON * DAYS
    state = np.zeros((N_SCENARIOS, n_days), dtype=int)
    state[:, 0] = rng.random(N_SCENARIOS) < (p_today @ P)[1]
    for t in range(1, n_days):
        state[:, t] = rng.random(N_SCENARIOS) < P[state[:, t - 1], 1]
    daily = np.zeros((N_SCENARIOS, n_days, len(assets)))
    for s in range(2):
        draws = rng.multivariate_normal(model.means_[s] / SCALE, model.covars_[s] / SCALE ** 2,
                                        size=(N_SCENARIOS, n_days))
        daily[state == s] = draws[state == s]
    daily = daily.reshape(N_SCENARIOS, HORIZON, DAYS, len(assets))
    return (1 + daily).prod(axis=2) - 1


# STEP 4: FIT ON THE TRAINING DATA (what the two regimes look like)
model_train = fit_hmm(train.values)
daily_bear = {}                                        # bear probability on every day
for t in range(len(train)):
    daily_bear[train.index[t]] = bear_prob(model_train, train.values[:t + 1])

summary = {}
for s, name in enumerate(["bull", "bear"]):
    mu = model_train.means_[s] / SCALE
    cov = model_train.covars_[s] / SCALE ** 2
    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)
    stay = model_train.transmat_[s, s]
    summary[name] = {**{f"{a} return (ann.)": mu[j] * 252 for j, a in enumerate(assets)},
                     **{f"{a} volatility (ann.)": vol[j] * np.sqrt(252) for j, a in enumerate(assets)},
                     "Corr Equity-Bond": corr[0, 1], 
                     "Corr Equity-BTC": corr[0, 2],
                     "Corr Bond-BTC": corr[1, 2], 
                     "Daily prob. of staying": stay,
                     "Expected length (trading days)": 1 / (1 - stay)}
summary = pd.DataFrame(summary)


# STEP 5: WALK FORWARD OVER THE OUT-OF-SAMPLE PERIOD
# Validation: the first refit (Feb 2022) uses only the training data. Each later
# month adds the months already seen (expanding window). Test: the same, starting
# from training + validation.
months = returns.index.to_period("M")
rebalance_dates = [returns.index[months == m][0]
                   for m in months[returns.index >= oos.index[0]].unique()]

prob_rows, mean_rows, hist_rows, cov_rows, eval_rows = [], [], [], [], []
scenarios = {}
rng = np.random.default_rng(SEED)

for i, date in enumerate(rebalance_dates):
    history = returns[returns.index < date]            # only data before the decision
    model = fit_hmm(history.values, seed=i)
    p_bear = bear_prob(model, history.values)
    p_today = np.array([1 - p_bear, p_bear])
    probs, means, covs = forecast(model, p_today)
    hist_mean = history.values.mean(axis=0) * DAYS
    hist_var = history.values.var(axis=0) * DAYS          # static benchmark for risk
    P = model.transmat_
    long_run_bear = P[0, 1] / (P[0, 1] + P[1, 0])          # long-run share of bear days
    scenarios[str(date.date())] = simulate_scenarios(model, p_today, rng).astype(np.float32)

    prob_rows.append({"rebalance_date": date.date(), 
                      "p_bear_today": p_bear,
                      "regime_today": "bear" if p_bear > 0.5 else "bull",
                      "p_bear_month1": probs[0], 
                      "p_bear_month2": probs[1],
                      "p_bear_month3": probs[2]})
    for k in range(HORIZON):
        key = {"rebalance_date": date.date(), "month_ahead": k + 1,
               "forecast_month": str(date.to_period("M") + k)}
        mean_rows.append({**key, **dict(zip(assets, means[k]))})
        hist_rows.append({**key, **dict(zip(assets, hist_mean))})
        for a, row in zip(assets, covs[k]):
            cov_rows.append({**key, "asset": a, **dict(zip(assets, row))})
        eval_rows.append({"month_ahead": k + 1, "month": date.to_period("M") + k,
                          "p_hmm": probs[k], "p_no_change": p_bear, "p_long_run": long_run_bear,
                          "var_hmm": np.diag(covs[k]), "var_static": hist_var})

    # Bear probability on each day of this month with this month's model,
    # using returns up to and including that day.
    for day in returns.index[months == date.to_period("M")]:
        daily_bear[day] = bear_prob(model, returns[returns.index <= day].values)

    print(f"{date.date()}  P(bear) today {p_bear:.2f}  months 1-3: {np.round(probs, 2)}")


# STEP 6: EVALUATE THE FORECASTS (out-of-sample months only)
# Regime: forecast P(bear) vs the realized share of bear days in that month
#   (average filtered bear probability), compared with "no change" and the
#   long-run share. Risk: QLIKE of forecast vs realized variance per asset
#   (lower is better), compared with the static full-history variance.
daily_bear_s = pd.Series(daily_bear).sort_index()
def qlike(realized, forecast):
    return np.mean(realized / forecast - np.log(realized / forecast) - 1)

scores = []
for row in eval_rows:
    in_month = oos.index.to_period("M") == row["month"]
    if not in_month.any():
        continue                                           # month is after the period
    month_returns = oos[in_month].values
    realized_bear = daily_bear_s[oos.index[in_month]].mean()
    realized_var = (month_returns ** 2).sum(axis=0)
    scale = len(month_returns) / DAYS                      # real month length
    scores.append({"month_ahead": row["month_ahead"],
                   "bear_error_hmm": (row["p_hmm"] - realized_bear) ** 2,
                   "bear_error_no_change": (row["p_no_change"] - realized_bear) ** 2,
                   "bear_error_long_run": (row["p_long_run"] - realized_bear) ** 2,
                   "qlike_hmm": qlike(realized_var, row["var_hmm"] * scale),
                   "qlike_static": qlike(realized_var, row["var_static"] * scale)})
evaluation = pd.DataFrame(scores).groupby("month_ahead").mean()


# STEP 7: SAVE
pd.DataFrame(prob_rows).to_csv(output / "regime_probabilities.csv", index=False)
pd.DataFrame(mean_rows).to_csv(output / "expected_returns.csv", index=False)
pd.DataFrame(hist_rows).to_csv(output / "expected_returns_historical.csv", index=False)
pd.DataFrame(cov_rows).to_csv(output / "covariances.csv", index=False)
np.savez_compressed(output / "cvar_scenarios.npz", **scenarios)   # np.load(file)["2022-06-01"]
pd.Series(daily_bear, name="p_bear").sort_index().to_csv(
    output / "daily_bear_probability.csv", index_label="date")
summary.to_csv(output / "regime_summary.csv")
evaluation.to_csv(output / "forecast_evaluation.csv")

print("\nREGIMES ON THE TRAINING DATA:")
print(summary.round(3).to_string())
print(f"\nFORECAST EVALUATION (averages over {PERIOD} months, lower is better):")
print(evaluation.round(3).to_string())
print(f"\n{len(rebalance_dates)} rebalance dates saved to {output}")
