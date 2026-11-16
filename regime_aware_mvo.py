"""Model 1: Regime-aware MVO MPO (HMM-MVO) (Bonnie).

The MVO MPO planner fed with regime-dependent forecasts: every planned month k
has its own expected returns mu_k and covariance S_k (from the HMM, the jump
model or Eliza's regime-conditional forecasts). The optimization rule is the
SAME in every month; only the inputs change with the regime outlook.

    weights, details = solve_plan(current_weights, forecast, horizon, policy, caps)

Policies (the regime-aware model is Max Sharpe MVO; Min Var MVO is kept as an
optional reference, and as the bear rule Model 2 uses)
  Max Sharpe MVO   every month: min variance + cost subject to
                   mu_k'w_k >= mu_k'w_k^MS - 1e-6 x days_k, (just under) the
                   return of that month's max-Sharpe portfolio (with H = 1
                   and no cost this returns about the max-Sharpe weights)
  Min Var MVO      every month: min variance + cost

Objective (receding horizon: plan H months, execute month 1, re-plan):
    min  sum_k [ gamma * w_k' S_k w_k + c * ||w_k - w_{k-1}||_1 ]
    s.t. 1'w_k = 1,  0 <= w_k <= caps

Inputs: forecast.mu (months, 3), forecast.cov (months, 3, 3), assets ordered
Equity, Bond, BTC, monthly decimal returns; forecast.days (months,) trading
days per month. The model reads no files and sees no realized returns;
regime_mvo_backtest.py supplies forecasts and scores it.

solve_stages() is the shared engine: it accepts one rule per planned month
('MS' or 'MV'). Model 2 (regime_switching_mvo.py) reuses it with rules that
change by forecast regime, so the two models differ ONLY in which months get
the return floor; inputs, costs, caps, solver and floor definition are shared.

Matched with Kaylin's CVaR MPO (cvar_mpo.matched_return_targets /
optimize_matched_cvar and regime_switching_cvar.optimize_switching_cvar):
  * Max-Sharpe floor of month k = mu_k' w_k^MS - 1e-6 x (trading days in
    month k), with w_k^MS the long-only, capped max-Sharpe portfolio of that
    month's own mean and risk forecast (zero risk-free rate).
  * One joint plan over all H months; only month 1 is executed.
  * Objective = risk weight x risk + 10 bps x one-way trading, no extra
    return reward; long-only, fully invested, caps (1, 1, 0.2).
Different by design: risk is variance (gamma = 1.25), not CVaR95 (0.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.optimize import brentq

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

RISK_AVERSION = 1.25     # gamma on monthly variance (lambda/2, lambda = 2.5)
COST_RATE = 0.001        # 10 bps per dollar traded, used inside the objective
RISK_FREE = 0.0          # monthly, for the max-Sharpe target
FLOOR_ALLOWANCE_PER_DAY = 1e-6   # floor = max-Sharpe return - 1e-6 per trading day (as in the CVaR MPO)

POLICIES = ("Max Sharpe MVO", "Min Var MVO")
HORIZONS = (1, 2, 3)     # planning horizons supported: the forecasts cover months 1-3


def check_horizon(horizon, forecast):
    """H must be 1, 2 or 3 and the forecast must cover all H months."""
    if isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError(f"Horizon must be one of {HORIZONS}, got {horizon}")
    if len(forecast.mu) < horizon or len(forecast.cov) < horizon:
        raise ValueError(f"Forecast covers {len(forecast.mu)} months; H = {horizon} needs {horizon}")


@dataclass(frozen=True)
class Forecast:
    mu: np.ndarray              # (months, 3) expected monthly returns
    cov: np.ndarray             # (months, 3, 3) monthly covariances
    p_bear: np.ndarray | None   # (months,) P(bear), or None if no regimes
    days: np.ndarray | None = None   # (months,) trading days per month; None = no floor allowance


# ------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------

def best_feasible_return(mu, caps):
    """Highest mu'w over the capped long-only simplex (fill best assets first)."""
    remaining, best = 1.0, 0.0
    for j in np.argsort(-mu):
        take = min(remaining, caps[j])
        best += take * mu[j]
        remaining -= take
    return best


def project_capped_simplex(values, caps):
    """Remove solver noise: nearest point with 0 <= w <= caps, sum(w) = 1."""
    shift = brentq(lambda t: np.clip(values - t, 0, caps).sum() - 1,
                   float(np.min(values - caps)) - 1, float(np.max(values)) + 1, xtol=1e-15)
    return np.clip(values - shift, 0, caps)


def max_sharpe_weights(mu, cov, caps, risk_free=RISK_FREE):
    """Long-only, capped maximum-Sharpe weights as a convex QP.

    Homogenized form: min y'Sy s.t. (mu - rf)'y = 1, sum(y) = kappa,
    0 <= y <= kappa * caps, kappa >= 0; then w = y / kappa.
    Returns None when no feasible portfolio has a positive excess return.
    """
    excess = mu - risk_free
    if best_feasible_return(excess, caps) <= 1e-12:
        return None
    y, kappa = cp.Variable(3), cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))),
                         [excess @ y == 1, cp.sum(y) == kappa, y >= 0, y <= kappa * caps])
    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal" or kappa.value is None or kappa.value <= 0:
        raise RuntimeError(f"Max-Sharpe QP failed: {problem.status}")
    return project_capped_simplex(y.value / kappa.value, caps)


# ------------------------------------------------------------
# Shared MPO engine: one rule per planned month
# ------------------------------------------------------------

def solve_stages(current_weights, forecast, stages, caps,
                 cost_rate=COST_RATE, gamma=RISK_AVERSION):
    """Plan weights for len(stages) months; stages[k] is 'MS' or 'MV'.

    Returns (weights to hold now, {"stages": rules used, "floors": return
    floor per month (None for MV months), "plan": planned path}).
    An 'MS' month with no positive-excess portfolio falls back to 'MV'.
    """
    horizon = len(stages)
    mus, covs = forecast.mu[:horizon], forecast.cov[:horizon]
    if forecast.days is not None and (len(forecast.days) < horizon or np.any(np.asarray(forecast.days) < 1)):
        raise ValueError("Provide a positive trading-day count for every planned month")
    W = cp.Variable((horizon, 3))
    constraints = [cp.sum(W, axis=1) == 1, W >= 0, W <= np.tile(caps, (horizon, 1))]
    objective, previous, used, floors = 0, current_weights, [], []
    for k in range(horizon):
        objective += gamma * cp.quad_form(W[k], cp.psd_wrap(covs[k]))
        objective += cost_rate * cp.sum(cp.abs(W[k] - previous))
        stage, floor = stages[k], None
        if stage == "MS":
            target = max_sharpe_weights(mus[k], covs[k], caps)
            if target is None:
                stage = "MV"
            else:
                allowance = 0.0 if forecast.days is None else FLOOR_ALLOWANCE_PER_DAY * forecast.days[k]
                floor = float(mus[k] @ target - max(allowance, 1e-9))
                constraints.append(mus[k] @ W[k] >= floor)
        elif stage != "MV":
            raise ValueError(f"Unknown stage rule: {stage}")
        used.append(stage)
        floors.append(floor)
        previous = W[k]
    problem = cp.Problem(cp.Minimize(100 * objective), constraints)   # x100: solver scaling only
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in ("optimal", "optimal_inaccurate") or W.value is None:
        raise RuntimeError(f"MPO problem status: {problem.status}")
    first = project_capped_simplex(W.value[0], caps)
    if np.abs(first - W.value[0]).max() > 1e-5:
        raise AssertionError("Weight clean-up moved the solver solution materially")
    return first, {"stages": used, "floors": floors, "plan": W.value}


# ------------------------------------------------------------
# Model 1 decision
# ------------------------------------------------------------

def solve_plan(current_weights, forecast, horizon, policy, caps,
               cost_rate=COST_RATE, gamma=RISK_AVERSION):
    """Regime-aware MVO MPO: the same rule every month, regime-dependent inputs.
    horizon: H = 1, 2 or 3 planned months."""
    check_horizon(horizon, forecast)
    if policy == "Max Sharpe MVO":
        stages = ["MS"] * horizon
    elif policy == "Min Var MVO":
        stages = ["MV"] * horizon
    else:
        raise ValueError(f"Unknown policy: {policy}; choose from {POLICIES}")
    return solve_stages(current_weights, forecast, stages, caps, cost_rate, gamma)