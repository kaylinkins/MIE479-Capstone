"""One convex mean-variance/trading-cost planner for H=1 and H>1, covering the
baseline models. Shares execution accounting with CVaR_MPO/cvar_mpo.py.

The objective is a sum of stage expected returns minus stage variance penalties
and proportional trade-cost estimates. Planned weights are a schedule; they do
not anticipate future outcomes. Future weight transitions ignore return drift
in the planning problem. Execution applies exact realized drift and
self-financing fees to ONLY the first planned allocation, using the same
execute_first_period / self_financing_fee as the CVaR planner.

Baseline models (see BASELINES):
  MVO (mean-variance)        max  sum mu'w - risk_aversion * w'Sw - costs
  MVO (min variance)         same with return_reward = 0
  Max Sharpe (matched)       min-variance-plus-cost with a per-period return
                             target equal to the stage max-Sharpe expected
                             return (the MVO analogue of CVaR matched return)
  Max Sharpe / Risk parity / Equal weight / 60/40 (tracking)
                             min  sum risk_aversion * (w-t)'S(w-t) + costs,
                             trading toward each model's single-period target t

This is a quadratic program, so it is solved with cvxpy (CLARABEL) rather than
scipy linprog. Risk is the sum of per-period variances (or active variances
against a target); it is not terminal-wealth variance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
from pathlib import Path
import sys

import numpy as np
import importlib
from scipy.optimize import brentq, minimize

# Resolve the consolidated library relative to this file, including direct runs
# from the Portfolio Optimization folder. No backtest runner is imported.
_CVAR_DIRECTORY = str(Path(__file__).resolve().parents[1] / "CVaR_MPO")
if _CVAR_DIRECTORY not in sys.path:
    sys.path.insert(0, _CVAR_DIRECTORY)
from cvar_mpo import (ASSETS, DEFAULT_UPPER, _costs, _limits, _portfolio,
                      execute_first_period, loss_var_cvar, self_financing_fee)

# Default risk aversion multiplies the variance directly (as risk_aversion
# multiplies CVaR in the CVaR planner). 1.25 = lambda/2 with lambda = 2.5,
# the risk-aversion coefficient used in the Black-Litterman prior.
DEFAULT_RISK_AVERSION = 1.25
_SCALE = 100.         # objective scaling for the solver only; reported values are unscaled

def _load_cvxpy():
    """Import the solver dependency when a mean-variance plan is requested."""
    try:
        return importlib.import_module("cvxpy")
    except ImportError as exc:
        raise ImportError(
            "Mean-variance planning requires cvxpy; install it with `pip install cvxpy`."
        ) from exc

__all__ = ["MVPlan", "solve_mv_plan", "optimize_forecast", "moments_from_scenarios",
           "max_sharpe_weights", "risk_parity_weights", "BASELINES",
           "execute_first_period", "self_financing_fee"]


def _covariances(values, horizon):
    values = np.asarray(values, float)
    if values.ndim == 2:
        values = np.repeat(values[None], horizon, axis=0)
    if values.shape != (horizon, 3, 3) or not np.isfinite(values).all():
        raise ValueError("Covariances must be finite (period, 3, 3) or one (3, 3) matrix")
    values = (values+np.transpose(values, (0, 2, 1)))/2
    for k, matrix in enumerate(values):
        if np.linalg.eigvalsh(matrix).min() < -1e-12*max(1., np.trace(matrix)):
            raise ValueError(f"Covariance in planning period {k} is not positive semidefinite")
    return values


def _targets(values, horizon, caps):
    if values is None:
        return None
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = np.repeat(values[None], horizon, axis=0)
    if values.shape != (horizon, 3):
        raise ValueError("Target weights must be one (3,) vector or one per planning period")
    for k, row in enumerate(values):
        _portfolio(row, f"Target weights in planning period {k}")
        if (row > caps+1e-10).any():
            raise ValueError(f"Target weights in planning period {k} violate the allocation caps")
    return values


def _clean_weights(weights, caps):
    """Remove numerical solver noise without renormalizing beyond a cap.

    Project onto the capped simplex: clipping and then dividing by the sum
    can raise a BTC allocation above its cap. The caller still checks that
    this numerical cleanup has not materially moved the solver solution.
    """
    values = np.asarray(weights, float)
    if caps.sum() < 1.:
        # Input validation permits a tiny floating-point shortfall from one.
        return caps.copy()
    shift = brentq(lambda t: np.clip(values-t, 0., caps).sum()-1.,
                   float(np.min(values-caps)), float(np.max(values)), xtol=5e-16)
    clean = np.clip(values-shift, 0., caps)
    residual = 1.-clean.sum()
    if residual > 0.:
        index = int(np.argmax(caps-clean))
        clean[index] += min(residual, caps[index]-clean[index])
    elif residual < 0.:
        index = int(np.argmax(clean))
        clean[index] += max(residual, -clean[index])
    return clean


@dataclass(frozen=True)
class MVPlan:
    weights: np.ndarray
    trades: np.ndarray
    pretrade_weights: np.ndarray
    expected_returns: np.ndarray
    forecast_volatility: np.ndarray
    forecast_active_volatility: np.ndarray | None
    forecast_var: np.ndarray | None
    forecast_cvar: np.ndarray | None
    estimated_costs: np.ndarray
    cost_rates: np.ndarray
    objective: float
    solver_optimum: float
    metadata: dict

    @property
    def first_weights(self):
        return self.weights[0].copy()


def solve_mv_plan(covariances, pretrade_weights, *, expected_returns=None,
                  risk_aversion=DEFAULT_RISK_AVERSION, return_reward=1., cost_rate=.001,
                  upper=DEFAULT_UPPER, min_return=None, risk_budget=None,
                  target_weights=None, terminal_weights=None):
    """Maximize sum(return_reward*mu*w - risk_aversion*risk - estimated trading costs).

    ``covariances`` has shape (period, 3, 3) (or one (3, 3) matrix used for
    every period), for the same holding-period simple returns as the
    expected returns. ``expected_returns`` has shape (period, 3); it is
    required when return_reward > 0 or a minimum-return target is set.

    Risk in period k is w'S_k w, or (w - t_k)'S_k(w - t_k) when
    ``target_weights`` is given (tracking a single-period model's target).
    ``return_reward=0`` gives minimum-risk-plus-cost optimization, optionally
    with gross per-period minimum-return targets. Risk budgets are per-period
    limits on volatility (or active volatility with a target). Terminal
    weights are optional and default to no liquidation constraint.
    """
    cp = _load_cvxpy()
    prior = _portfolio(pretrade_weights, "Pretrade weights", allow_cash=True)
    costs = _costs(cost_rate)
    caps = np.asarray(upper, float)
    if (caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any() or
            (caps > 1).any() or caps.sum() < 1-1e-12):
        raise ValueError("Weight caps must admit a fully invested long-only portfolio")
    if (not np.isfinite(risk_aversion) or risk_aversion <= 0 or
            not np.isfinite(return_reward) or return_reward < 0):
        raise ValueError("Require positive risk aversion and nonnegative return reward")
    raw = np.asarray(covariances, float)
    if raw.ndim == 3:
        horizon = raw.shape[0]
    elif expected_returns is not None:
        horizon = np.asarray(expected_returns, float).shape[0]
    elif target_weights is not None and np.ndim(target_weights) == 2:
        horizon = np.shape(target_weights)[0]
    else:
        horizon = 1
    if horizon < 1:
        raise ValueError("Planning horizon must be at least one period")
    covs = _covariances(raw, horizon)
    if expected_returns is None:
        if return_reward > 0 or min_return is not None:
            raise ValueError("Expected returns are required for a return reward or return target")
        means = np.zeros((horizon, 3))
    else:
        means = np.asarray(expected_returns, float)
    if means.shape != (horizon, 3) or not np.isfinite(means).all() or (means <= -1).any():
        raise ValueError("Expected returns must be finite (period, 3) decimal simple returns")
    targets = _limits(min_return, horizon, "Minimum-return targets")
    budgets = _limits(risk_budget, horizon, "Volatility budgets")
    if budgets is not None and (budgets <= 0).any():
        raise ValueError("Volatility budgets must be positive")
    tracked = _targets(target_weights, horizon, caps)
    terminal = _portfolio(terminal_weights, "Terminal weights") if terminal_weights is not None else None
    if terminal is not None and (terminal > caps+1e-10).any():
        raise ValueError("Terminal weights violate the allocation caps")
    if targets is not None:
        for k, mean in enumerate(means):
            remaining, maximum = 1., 0.
            for j in np.argsort(-mean):
                allocated = min(remaining, caps[j])
                maximum += allocated*mean[j]
                remaining -= allocated
            if targets[k] > maximum+1e-10:
                raise ValueError(f"Minimum-return target infeasible in planning period {k}")

    # Variables: planned weights W (H x 3); trades are W_k - W_{k-1}, with W_{-1} = prior.
    W = cp.Variable((horizon, 3))
    reward, risk, cost = 0, 0, 0
    constraints = [cp.sum(W, axis=1) == 1, W >= 0, W <= np.tile(caps, (horizon, 1))]
    previous = prior
    for k in range(horizon):
        S = cp.psd_wrap(covs[k])
        deviation = W[k]-tracked[k] if tracked is not None else W[k]
        stage_risk = cp.quad_form(deviation, S)
        reward += return_reward*(means[k] @ W[k])
        risk += stage_risk
        cost += costs @ cp.abs(W[k]-previous)
        if targets is not None:
            constraints.append(means[k] @ W[k] >= targets[k])
        if budgets is not None:
            constraints.append(stage_risk <= budgets[k]**2)
        previous = W[k]
    if terminal is not None:
        constraints.append(W[horizon-1] == terminal)
    problem = cp.Problem(cp.Maximize(_SCALE*(reward-risk_aversion*risk-cost)), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal":
        raise RuntimeError(f"Mean-variance planning failed: {problem.status}")

    weights = np.asarray([_clean_weights(row, caps) for row in W.value])
    trades = np.diff(np.vstack([prior, weights]), axis=0)
    expected = (means*weights).sum(axis=1)
    volatility = np.sqrt([w @ S @ w for w, S in zip(weights, covs)])
    active = None if tracked is None else np.sqrt(
        [np.maximum((w-t) @ S @ (w-t), 0.) for w, t, S in zip(weights, tracked, covs)])
    stage_risk = (active if tracked is not None else volatility)**2
    estimated_costs = (np.abs(trades)*costs).sum(axis=1)
    if (not np.allclose(weights.sum(axis=1), 1., atol=1e-7, rtol=0) or
            (weights < 0.).any() or (weights > caps).any() or
            not np.allclose(weights, W.value, atol=1e-6, rtol=0)):
        raise AssertionError("Solution violates the weight constraints beyond solver tolerance")
    if targets is not None and (expected < targets-2e-7).any():
        raise AssertionError("Solution violates a return target")
    if budgets is not None and (np.sqrt(stage_risk) > budgets+2e-7).any():
        raise AssertionError("Solution violates a volatility budget")
    value = float(return_reward*expected.sum()-risk_aversion*stage_risk.sum()-estimated_costs.sum())
    if not np.isclose(value, problem.value/_SCALE, atol=5e-7, rtol=1e-6):
        raise AssertionError("Reported objective differs from the solved objective")
    metadata = {"horizon": horizon, "assets": list(ASSETS),
        "risk_measure": ("sum of per-period active variances against target weights" if tracked is not None
                         else "sum of per-period portfolio variances") + "; not terminal-wealth variance",
        "risk_aversion": float(risk_aversion), "return_reward": float(return_reward),
        "planning_dynamics": "future trade = next planned weights minus previous planned weights; return drift ignored only in planning",
        "execution": "execute first allocation only; apply exact drift and self-financing fees, then replan",
        "terminal_policy": "fixed last-stage allocation" if terminal is not None else "no forced liquidation or terminal fee",
        "solver": "cvxpy (CLARABEL)", "solver_status": problem.status,
        "upper": caps.tolist(), "gross_return_targets": None if targets is None else targets.tolist(),
        "volatility_budgets": None if budgets is None else budgets.tolist(),
        "target_weights": None if tracked is None else tracked.tolist()}
    return MVPlan(weights.copy(), trades.copy(), prior, expected, volatility, active, None, None,
                  estimated_costs, costs.copy(), value, float(problem.value/_SCALE), metadata)


# ------------------------------------------------------------
# Single-period targets used by the baseline models
# ------------------------------------------------------------

def max_sharpe_weights(mean, covariance, upper=DEFAULT_UPPER, risk_free=0.):
    """Long-only maximum-Sharpe weights for one period (same scale for all inputs)."""
    mean, covariance, caps = (np.asarray(x, float) for x in (mean, covariance, upper))
    start = np.full(3, 1/3)
    solved = minimize(lambda w: -(w @ mean-risk_free)/np.sqrt(w @ covariance @ w), start,
                      bounds=[(0., float(c)) for c in caps],
                      constraints=[{"type": "eq", "fun": lambda w: w.sum()-1.}], method="SLSQP")
    if not solved.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {solved.message}")
    return _clean_weights(solved.x, caps)


def risk_parity_weights(covariance):
    """Equal-risk-contribution weights (convex formulation, rescaled to sum to 1)."""
    covariance = np.asarray(covariance, float)
    solved = minimize(lambda y: .5*y @ covariance @ y-np.log(y).sum()/3,
                      1/np.sqrt(np.diag(covariance)), bounds=[(1e-9, None)]*3, method="L-BFGS-B")
    if not solved.success:
        raise RuntimeError(f"Risk-parity optimization failed: {solved.message}")
    return solved.x/solved.x.sum()


def _per_period(function, *arrays):
    return np.asarray([function(*items) for items in zip(*arrays)])


def _fixed(weights):
    return lambda means, covs, caps: np.repeat(np.asarray(weights, float)[None], len(means), axis=0)


# name -> (uses BTC cap, settings builder(means, covs, caps) -> solve_mv_plan keyword arguments)
BASELINES = {
    "MVO (mean-variance)": (True, lambda means, covs, caps: {"return_reward": 1.}),
    "MVO (min variance)": (True, lambda means, covs, caps: {"return_reward": 0.}),
    "Max Sharpe (matched return)": (True, lambda means, covs, caps: {
        "return_reward": 0.,
        "min_return": (means*_per_period(lambda m, S: max_sharpe_weights(m, S, caps),
                                         means, covs)).sum(axis=1)-1e-9}),
    "Max Sharpe (tracking)": (True, lambda means, covs, caps: {
        "return_reward": 0.,
        "target_weights": _per_period(lambda m, S: max_sharpe_weights(m, S, caps), means, covs)}),
    "Risk parity (tracking)": (False, lambda means, covs, caps: {
        "return_reward": 0., "target_weights": _per_period(risk_parity_weights, covs)}),
    "Equal weight (tracking)": (False, lambda means, covs, caps: {
        "return_reward": 0., "target_weights": _fixed([1/3, 1/3, 1/3])(means, covs, caps)}),
    "60/40 (tracking)": (False, lambda means, covs, caps: {
        "return_reward": 0., "target_weights": _fixed([.6, .4, 0.])(means, covs, caps)}),
}


# ------------------------------------------------------------
# Connection to the shared forecast bundle (same as the CVaR planner)
# ------------------------------------------------------------

def moments_from_scenarios(scenarios):
    """Per-period scenario covariances, shape (period, 3, 3), from (scenario, period, 3) returns."""
    values = np.asarray(scenarios, float)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Scenarios must be finite (scenario, period, 3) with at least two scenarios")
    return np.asarray([np.cov(values[:, k, :], rowvar=False) for k in range(values.shape[1])])


def optimize_forecast(bundle, model, pretrade_weights, *, baseline="MVO (mean-variance)",
                      horizon=None, btc_cap=.20, alpha=.95, **settings):
    """Connect the common forecast bundle to the same planner at any prefix H.

    Uses the bundle's shared expected returns and the per-period covariance of
    its native scenarios, so mean-variance and CVaR plans are built from the
    same forecasts. The CVaR of the chosen weights on those scenarios is
    reported (forecast_var / forecast_cvar) for side-by-side comparison.
    """
    if tuple(bundle.assets) != ASSETS:
        raise ValueError("Optimizer requires Equity/Bond/BTC forecast order")
    if model not in bundle.returns:
        raise ValueError(f"Unknown forecast model: {model}")
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline}; choose from {list(BASELINES)}")
    maximum = bundle.returns[model].shape[1]
    horizon = maximum if horizon is None else horizon
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= maximum:
        raise ValueError("Requested horizon must be an available positive forecast prefix")
    if "expected_returns" in settings or "upper" in settings:
        raise ValueError("Bundle optimization sets expected returns and caps itself")
    capped, builder = BASELINES[baseline]
    caps = np.array([1., 1., btc_cap if (capped and btc_cap is not None) else 1.])
    scenarios = bundle.returns[model][:, :horizon]
    means = np.asarray(bundle.expected_returns[:horizon], float)
    covs = moments_from_scenarios(scenarios)
    model_settings = builder(means, covs, caps)
    clash = set(model_settings) & set(settings)
    if clash:
        raise ValueError(f"Settings {sorted(clash)} are fixed by the {baseline} baseline")
    plan = solve_mv_plan(covs, pretrade_weights, expected_returns=means, upper=caps,
                         **model_settings, **settings)
    var, es = np.asarray([loss_var_cvar(scenarios[:, k] @ plan.weights[k], alpha)
                          for k in range(horizon)]).T
    plan.metadata.update({"baseline": baseline, "forecast_model": model,
                          "forecast_origin": str(bundle.origin.date()),
                          "period_dates": [[str(day.date()) for day in period] for period in bundle.periods[:horizon]],
                          "forecast_fingerprint": bundle.metadata.get("forecast_fingerprint"),
                          "covariance_source": "per-period covariance of the bundle's native scenarios",
                          "cvar_diagnostic_alpha": alpha})
    return replace(plan, forecast_var=var, forecast_cvar=es)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
