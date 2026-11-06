"""One convex CVaR/trading-cost planner for H=1 and H>1.

The objective is a sum of stage expected returns minus stage CVaR penalties
and proportional trade-cost estimates. Planned weights are common across all
scenarios; they do not anticipate future outcomes. Future weight transitions
ignore return drift in the planning LP. Execution below applies exact realized
drift and self-financing fees to ONLY the first planned allocation.

This is not terminal-wealth CVaR, nested CVaR, or a scenario-tree policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from itertools import combinations, product

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.sparse import block_diag, csr_matrix, diags, hstack, identity, kron, vstack

ASSETS = ("Equity", "Bond", "BTC")
DEFAULT_UPPER = (1., 1., 1.)


def _portfolio(values, label, *, allow_cash=False):
    values = np.asarray(values, float)
    if (values.shape != (3,) or not np.isfinite(values).all() or (values < -1e-10).any() or
            not (np.isclose(values.sum(), 1., atol=1e-9, rtol=0) or
                 (allow_cash and np.all(values == 0)))):
        raise ValueError(f"{label} must be a long-only fully invested three-asset vector" +
                         (" or zero holdings for initial cash" if allow_cash else ""))
    return values.copy()


def _costs(values):
    values = np.asarray(values, float)
    if values.ndim == 0:
        values = np.full(3, float(values))
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or (values >= .1).any():
        raise ValueError("Buy/sell cost rates must be finite, nonnegative and below 10%")
    return values


def _limits(values, horizon, label):
    if values is None:
        return None
    values = np.asarray(values, float)
    if values.ndim == 0:
        values = np.full(horizon, float(values))
    if values.shape != (horizon,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite scalar or one value per planning period")
    return values


def loss_var_cvar(returns, alpha=.95):
    """Interpolated loss VaR and fractional-tail CVaR consistent with the LP."""
    returns = np.asarray(returns, float)
    if returns.ndim != 1 or not len(returns) or not np.isfinite(returns).all() or not 0 < alpha < 1:
        raise ValueError("Require finite return observations and confidence in (0,1)")
    losses = np.sort(-returns)[::-1]
    mass = (1-alpha)*len(losses)
    whole = int(np.floor(mass))
    fraction = mass-whole
    es = (losses[:whole].sum()+(fraction*losses[whole] if fraction > 1e-12 else 0))/mass
    return float(np.quantile(-returns, alpha)), float(es)


@dataclass(frozen=True)
class CVaRPlan:
    weights: np.ndarray
    trades: np.ndarray
    pretrade_weights: np.ndarray
    expected_returns: np.ndarray
    forecast_var: np.ndarray
    forecast_cvar: np.ndarray
    estimated_costs: np.ndarray
    cost_rates: np.ndarray
    objective: float
    solver_minimum: float
    alpha: float
    metadata: dict

    @property
    def first_weights(self):
        return self.weights[0].copy()


def solve_cvar_plan(scenarios, pretrade_weights, *, expected_returns=None,
                    alpha=.95, risk_aversion=1., return_reward=1., cost_rate=.001,
                    upper=DEFAULT_UPPER, min_return=None, risk_budget=None,
                    terminal_weights=None):
    """Maximize sum(mu*w - risk_aversion*CVaR - estimated trading costs).

    ``scenarios`` has shape (scenario, period, asset), in decimal incremental
    simple returns. Every scenario has probability 1/N. ``expected_returns``
    supplies a common ex-ante objective/target forecast across model comparisons;
    omitting it explicitly uses each stage's scenario mean. CVaR always values
    the supplied native scenario distribution; there is no mean translation.

    ``return_reward=0`` gives minimum-CVaR-plus-cost optimization, optionally
    with gross per-period minimum-return targets. Risk budgets are gross
    per-period CVaR limits. Terminal weights are optional and default to no
    liquidation constraint or terminal fee. Future allocations are a planning
    schedule, not independent scenario-contingent decisions.
    """
    values = np.asarray(scenarios, float)
    if (values.ndim != 3 or min(values.shape) < 1 or values.shape[2] != 3 or
            not np.isfinite(values).all() or (values <= -1).any()):
        raise ValueError("Scenarios must be finite (scenario, period, 3) simple returns greater than -1")
    n, horizon, assets = values.shape
    prior = _portfolio(pretrade_weights, "Pretrade weights", allow_cash=True)
    costs = _costs(cost_rate)
    caps = np.asarray(upper, float)
    if (caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any() or
            (caps > 1).any() or caps.sum() < 1-1e-12):
        raise ValueError("Weight caps must admit a fully invested long-only portfolio")
    if (not np.isfinite(alpha) or not 0 < alpha < 1 or not np.isfinite(risk_aversion) or
            risk_aversion <= 0 or not np.isfinite(return_reward) or return_reward < 0):
        raise ValueError("Require confidence in (0,1), positive risk aversion and nonnegative return reward")
    means = values.mean(axis=0) if expected_returns is None else np.asarray(expected_returns, float)
    if means.shape != (horizon, 3) or not np.isfinite(means).all() or (means <= -1).any():
        raise ValueError("Expected returns must be finite (period, 3) decimal simple returns")
    targets = _limits(min_return, horizon, "Minimum-return targets")
    budgets = _limits(risk_budget, horizon, "CVaR budgets")
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

    # Variables: [w(H*3), eta(H), tail_slack(H*N), abs_trade(H*3)].
    nw, nu = horizon*assets, horizon*n
    variables = 2*nw+horizon+nu
    objective = np.r_[-return_reward*means.ravel(), np.full(horizon, risk_aversion),
                       np.full(nu, risk_aversion/((1-alpha)*n)), np.tile(costs, horizon)]
    tail = hstack([block_diag([csr_matrix(-values[:, k, :]) for k in range(horizon)], format="csr"),
                   -kron(identity(horizon), np.ones((n, 1)), format="csr"),
                   -identity(nu, format="csr"), csr_matrix((nu, nw))], format="csr")
    difference = kron(diags([np.ones(horizon), -np.ones(max(0, horizon-1))],
                            [0, -1], shape=(horizon, horizon)), identity(assets), format="csr")
    forward = hstack([difference, csr_matrix((nw, horizon+nu)), -identity(nw)], format="csr")
    reverse = hstack([-difference, csr_matrix((nw, horizon+nu)), -identity(nw)], format="csr")
    offset = np.r_[prior, np.zeros(nw-assets)]
    inequalities = [tail, forward, reverse]
    bounds_rhs = [np.zeros(nu), offset, -offset]
    if targets is not None:
        mean_rows = block_diag([csr_matrix(-means[k:k+1]) for k in range(horizon)], format="csr")
        inequalities.append(hstack([mean_rows, csr_matrix((horizon, variables-nw))], format="csr"))
        bounds_rhs.append(-targets)
    if budgets is not None:
        risk_rows = hstack([csr_matrix((horizon, nw)), identity(horizon),
                            kron(identity(horizon), np.full((1, n), 1/((1-alpha)*n)), format="csr"),
                            csr_matrix((horizon, nw))], format="csr")
        inequalities.append(risk_rows)
        bounds_rhs.append(budgets)
    weight_sum = kron(identity(horizon), np.ones((1, assets)), format="csr")
    equality = hstack([weight_sum, csr_matrix((horizon, variables-nw))], format="csr")
    weights_bounds = [(0., float(cap)) for _ in range(horizon) for cap in caps]
    if terminal is not None:
        weights_bounds[-assets:] = [(float(w), float(w)) for w in terminal]
    bounds = weights_bounds+[(None, None)]*horizon+[(0., None)]*(nu+nw)
    solved = linprog(objective, A_ub=vstack(inequalities, format="csr"),
                     b_ub=np.concatenate(bounds_rhs), A_eq=equality, b_eq=np.ones(horizon),
                     bounds=bounds, method="highs")
    if not solved.success:
        raise RuntimeError(f"CVaR planning failed: {solved.message}")
    weights = solved.x[:nw].reshape(horizon, assets)
    trades = np.diff(np.vstack([prior, weights]), axis=0)
    expected = (means*weights).sum(axis=1)
    var, es = np.asarray([loss_var_cvar(values[:, k] @ weights[k], alpha) for k in range(horizon)]).T
    estimated_costs = (np.abs(trades)*costs).sum(axis=1)
    eta = solved.x[nw:nw+horizon]
    slack = solved.x[nw+horizon:nw+horizon+nu].reshape(horizon, n)
    lp_es = eta+slack.mean(axis=1)/(1-alpha)
    if (not np.allclose(weights.sum(axis=1), 1., atol=1e-7, rtol=0) or
            (weights < -1e-7).any() or (weights > caps+1e-7).any() or
            not np.allclose(es, lp_es, atol=2e-7, rtol=1e-7)):
        raise AssertionError("LP solution violates weights or empirical CVaR consistency")
    if targets is not None and (expected < targets-2e-7).any():
        raise AssertionError("LP solution violates a return target")
    if budgets is not None and (es > budgets+2e-7).any():
        raise AssertionError("LP solution violates a CVaR budget")
    value = float(return_reward*expected.sum()-risk_aversion*es.sum()-estimated_costs.sum())
    if not np.isclose(value, -solved.fun, atol=5e-7, rtol=1e-7):
        raise AssertionError("Reported objective differs from the solved objective")
    metadata = {"horizon": horizon, "scenario_count": n, "assets": list(ASSETS),
        "risk_measure": "sum of gross per-period CVaRs; not terminal-wealth or nested CVaR",
        "risk_aversion": float(risk_aversion), "return_reward": float(return_reward),
        "expected_return_source": "explicit common forecasts" if expected_returns is not None else "native scenario means",
        "planning_dynamics": "future trade = next planned weights minus previous planned weights; return drift ignored only in planning",
        "execution": "execute first allocation only; apply exact drift and self-financing fees, then replan",
        "terminal_policy": "fixed last-stage allocation" if terminal is not None else "no forced liquidation or terminal fee",
        "solver": "scipy.optimize.linprog(method='highs')", "solver_message": solved.message,
        "upper": caps.tolist(), "gross_return_targets": None if targets is None else targets.tolist(),
        "gross_cvar_budgets": None if budgets is None else budgets.tolist()}
    return CVaRPlan(weights.copy(), trades.copy(), prior, expected, var, es,
                    estimated_costs, costs.copy(), value, float(solved.fun), alpha, metadata)


def optimize_forecast(bundle, model, pretrade_weights, *, horizon=None, **settings):
    """Connect the common forecast bundle to the same planner at any prefix H."""
    if tuple(bundle.assets) != ASSETS:
        raise ValueError("Optimizer requires Equity/Bond/BTC forecast order")
    if model not in bundle.returns:
        raise ValueError(f"Unknown forecast model: {model}")
    maximum = bundle.returns[model].shape[1]
    horizon = maximum if horizon is None else horizon
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= maximum:
        raise ValueError("Requested horizon must be an available positive forecast prefix")
    if "expected_returns" in settings:
        raise ValueError("Bundle optimization uses its explicitly shared expected-return forecasts")
    plan = solve_cvar_plan(bundle.returns[model][:, :horizon], pretrade_weights,
                          expected_returns=bundle.expected_returns[:horizon], **settings)
    plan.metadata.update({"forecast_model": model, "forecast_origin": str(bundle.origin.date()),
                          "period_dates": [[str(day.date()) for day in period] for period in bundle.periods[:horizon]],
                          "forecast_fingerprint": bundle.metadata.get("forecast_fingerprint")})
    return plan


@dataclass(frozen=True)
class MaxSharpeReturnTargets:
    """Ex-ante, stage-specific gross return floors and their reference portfolios.

    The reference maximizes monthly expected excess return divided by monthly
    volatility. It uses native scenario sample covariance and the explicitly
    supplied common mean forecast. It does not modify the scenario distribution.
    """

    weights: np.ndarray
    covariances: np.ndarray
    targets: np.ndarray
    sharpe_ratios: np.ndarray
    risk_free: np.ndarray
    upper: np.ndarray
    metadata: dict


def _capped_simplex_vertices(caps):
    """Enumerate vertices of the three-asset, fully invested allocation polygon."""
    vertices = []
    for fixed in combinations(range(3), 2):
        free = next(j for j in range(3) if j not in fixed)
        for at_upper in product((False, True), repeat=2):
            weight = np.zeros(3)
            for j, use_upper in zip(fixed, at_upper):
                weight[j] = caps[j] if use_upper else 0.
            weight[free] = 1.-weight.sum()
            if (weight >= -1e-12).all() and (weight <= caps+1e-12).all():
                weight = np.clip(weight, 0., caps)
                if not any(np.allclose(weight, old, atol=1e-12, rtol=0) for old in vertices):
                    vertices.append(weight)
    if not vertices:
        raise ValueError("Weight caps do not admit a fully invested reference portfolio")
    return vertices


def _maximum_sharpe_three_assets(mean, covariance, caps, risk_free):
    """Global analytic candidate enumeration on a capped three-asset simplex.

    On an edge p+t*d the Sharpe derivative has a linear numerator, giving at
    most one nonconstant interior stationary point. At a full-dimensional
    interior optimum homogeneity implies Sigma*w is proportional to excess
    mean. Enumerating vertices, edge stationary points and that tangency point
    therefore includes every global maximum for positive-definite covariance.
    All vertex-pair segments are considered; extra diagonals are feasible and
    cannot improve beyond the true global maximum. No nonlinear local solver
    or covariance ridge is used. Accuracy is subject to floating-point arithmetic.
    """
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= max(0., eigenvalues[-1]*1e-12):
        raise ValueError("Maximum-Sharpe reference requires positive-definite native covariance; "
                         "zero-volatility or numerically singular cases are not silently regularized")
    excess = mean-risk_free
    vertices = _capped_simplex_vertices(caps)
    candidates = list(vertices)
    for first, second in combinations(vertices, 2):
        direction = second-first
        numerator, slope = float(excess @ first), float(excess @ direction)
        variance = float(first @ covariance @ first)
        cross = float(first @ covariance @ direction)
        curvature = float(direction @ covariance @ direction)
        constant = slope*variance-numerator*cross
        coefficient = slope*cross-numerator*curvature
        if coefficient != 0.:
            t = -constant/coefficient
            if 0. < t < 1.:
                candidates.append(first+t*direction)
    tangency = np.linalg.solve(covariance, excess)
    if abs(tangency.sum()) > 1e-14*np.linalg.norm(tangency, ord=1):
        candidate = tangency/tangency.sum()
        # Strict feasibility avoids moving a tangency candidate outside caps
        # during normalization. Boundary optima are already included above.
        if (candidate >= 0).all() and (candidate <= caps).all():
            candidates.append(candidate)
    matrix = np.asarray(candidates)
    variances = np.einsum("ni,ij,nj->n", matrix, covariance, matrix)
    ratios = (matrix @ excess)/np.sqrt(variances)
    index = int(np.argmax(ratios))
    chosen = matrix[index].copy()
    violation = max(float(np.max(-chosen)), float(np.max(chosen-caps)),
                    abs(float(chosen.sum())-1.), 0.)
    if violation > 1e-10 or not np.isfinite(ratios[index]):
        raise AssertionError("Maximum-Sharpe candidate failed feasibility or finite-risk checks")
    gradient = excess-float(excess @ chosen)/variances[index]*(covariance @ chosen)
    supporting_violation = max(float(gradient @ (vertex-chosen)) for vertex in vertices)
    return chosen, float(ratios[index]), {
        "candidate_count": len(candidates),
        "vertex_count": len(vertices),
        "minimum_covariance_eigenvalue": float(eigenvalues[0]),
        "covariance_condition_number": float(eigenvalues[-1]/eigenvalues[0]),
        "maximum_allocation_violation": violation,
        "positive_feasible_excess_return": bool(max(float(excess @ v) for v in vertices) > 0.),
        "positive_sharpe_supporting_plane_violation": max(0., supporting_violation)
            if ratios[index] > 0 else None,
    }


def maximum_sharpe_return_targets(scenarios, expected_returns, *, upper=DEFAULT_UPPER,
                                 risk_free=0.):
    """Construct maximum-Sharpe-matched *gross monthly* return floors.

    All stages use the forecasts available at the current origin, not the
    observations that will later realize in those months. Covariance is the
    unbiased sample covariance (ddof=1) of each native incremental-return stage;
    means are the supplied common forecast, not those sample means. ``risk_free``
    is a monthly simple return, scalar or one value per stage, and only affects
    the reference's Sharpe numerator. Floors remain gross ``mean @ weights``.

    Reference and CVaR allocation caps must be identical. No return floor is
    relaxed. Singular covariance is explicitly rejected because a positive
    zero-volatility excess return makes the usual Sharpe ratio undefined.
    Nonpositive feasible excess returns are supported by the same global
    candidate enumeration; the resulting floor may be zero or negative.
    """
    values = np.asarray(scenarios, float)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1 or
            values.shape[2] != 3 or not np.isfinite(values).all() or (values <= -1).any()):
        raise ValueError("Reference scenarios require at least two finite (scenario, period, 3) "
                         "simple returns greater than -1")
    horizon = values.shape[1]
    means = np.asarray(expected_returns, float)
    if means.shape != (horizon, 3) or not np.isfinite(means).all() or (means <= -1).any():
        raise ValueError("Common expected returns must be finite (period, 3) simple returns")
    caps = np.asarray(upper, float)
    if (caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any() or
            (caps > 1).any() or caps.sum() < 1-1e-12):
        raise ValueError("Weight caps must admit a fully invested long-only reference portfolio")
    rates = _limits(risk_free, horizon, "Monthly risk-free returns")
    if rates is None:
        raise ValueError("Monthly risk-free returns must be supplied as a scalar or stage vector")
    covariances = np.asarray([np.cov(values[:, k], rowvar=False, ddof=1) for k in range(horizon)])
    weights, ratios, diagnostics = [], [], []
    for k in range(horizon):
        try:
            weight, ratio, diagnostic = _maximum_sharpe_three_assets(means[k], covariances[k], caps, rates[k])
        except ValueError as error:
            raise ValueError(f"Planning period {k}: {error}") from error
        weights.append(weight)
        ratios.append(ratio)
        diagnostics.append(diagnostic)
    weights = np.asarray(weights)
    return MaxSharpeReturnTargets(weights, covariances, np.sum(means*weights, axis=1),
        np.asarray(ratios), rates.copy(), caps.copy(), {
            "method": "analytic capped three-asset maximum-Sharpe candidate enumeration",
            "solver_guarantee": "global candidate coverage for positive-definite covariance, "
                                "subject to floating-point arithmetic",
            "covariance_source": "native stage scenario sample covariance, ddof=1; no ridge",
            "mean_source": "explicit common stage expected-return forecasts",
            "floor_definition": "gross common mean dot same-stage maximum-Sharpe reference weights",
            "target_relaxation": "none",
            "stage_diagnostics": diagnostics,
        })


def optimize_max_sharpe_matched_forecast(bundle, model, pretrade_weights, *, horizon=None,
                                        risk_free=0., **settings):
    """Minimize stage CVaRs plus trading costs at maximum-Sharpe return floors.

    This is an opt-in policy: the existing ``optimize_forecast`` defaults do not
    change. The stage objective is ``-lambda*CVaR-estimated_cost``; expected
    return enters only through the gross per-stage lower bounds. With nonzero
    costs lambda still determines risk versus turnover. Monthly execution and
    replanning remain the caller's responsibility. This is stagewise CVaR,
    rather than the separate compounded terminal-wealth formulation.
    """
    if "min_return" in settings:
        raise ValueError("Matched policy computes min_return from maximum-Sharpe references")
    if "expected_returns" in settings:
        raise ValueError("Matched policy uses the bundle's common expected-return forecasts")
    reward = settings.pop("return_reward", 0.)
    if not np.isscalar(reward) or not np.isfinite(reward) or reward != 0:
        raise ValueError("Matched minimum-CVaR policy requires return_reward=0")
    if tuple(bundle.assets) != ASSETS or model not in bundle.returns:
        raise ValueError("Require Equity/Bond/BTC asset order and an available forecast model")
    maximum = bundle.returns[model].shape[1]
    horizon = maximum if horizon is None else horizon
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= maximum:
        raise ValueError("Requested horizon must be an available positive forecast prefix")
    reference = maximum_sharpe_return_targets(bundle.returns[model][:, :horizon],
        bundle.expected_returns[:horizon], upper=settings.get("upper", DEFAULT_UPPER), risk_free=risk_free)
    plan = optimize_forecast(bundle, model, pretrade_weights, horizon=horizon,
                             return_reward=0., min_return=reference.targets, **settings)
    plan.metadata.update({"policy": "minimum CVaR plus costs with maximum-Sharpe-matched gross return floors",
        "return_floor_reference": {**reference.metadata, "weights": reference.weights.tolist(),
            "covariances": reference.covariances.tolist(), "targets": reference.targets.tolist(),
            "sharpe_ratios": reference.sharpe_ratios.tolist(),
            "monthly_risk_free_returns": reference.risk_free.tolist(), "upper": reference.upper.tolist()}})
    return plan


def self_financing_fee(target_weights, pretrade_weights, cost_rate=.001):
    target = _portfolio(target_weights, "Target weights")
    prior = _portfolio(pretrade_weights, "Pretrade weights", allow_cash=True)
    costs = _costs(cost_rate)
    if not costs.any():
        return 0.
    return float(brentq(lambda fee: fee-np.sum(costs*np.abs((1-fee)*target-prior)), 0., .999))


@dataclass(frozen=True)
class FirstPeriodExecution:
    fee_fraction: float
    gross_daily_returns: np.ndarray
    net_daily_returns: np.ndarray
    gross_period_return: float
    net_period_return: float
    ending_weights: np.ndarray
    ending_wealth: float


def execute_first_period(plan, realized_daily_returns, *, initial_wealth=1.):
    """Apply just the first allocation to explicitly supplied realized returns.

    No file is read here. This is an accounting building block, not an automatic
    backtest. Drift may take weights outside caps between decision dates; the
    next optimization enforces caps again on its chosen post-trade allocation.
    """
    returns = np.asarray(realized_daily_returns, float)
    if (returns.ndim != 2 or not len(returns) or returns.shape[1] != 3 or
            not np.isfinite(returns).all() or (returns <= -1).any() or
            not np.isfinite(initial_wealth) or initial_wealth <= 0):
        raise ValueError("Supply finite realized daily simple returns and positive starting wealth")
    target = plan.first_weights
    fee = self_financing_fee(target, plan.pretrade_weights, plan.cost_rates)
    holdings = target.copy()
    gross = []
    for row in returns:
        value = float(row @ holdings)
        gross.append(value)
        holdings *= (1+row)/(1+value)
    gross = np.asarray(gross)
    net = gross.copy()
    net[0] = (1-fee)*(1+gross[0])-1
    gross_period = float(np.prod(1+gross)-1)
    net_period = float(np.prod(1+net)-1)
    expected_gross = float((np.prod(1+returns, axis=0)-1) @ target)
    if (not np.isclose(gross_period, expected_gross, atol=1e-12, rtol=1e-10) or
            not np.isclose(net_period, (1-fee)*(1+expected_gross)-1, atol=1e-12, rtol=1e-10)):
        raise AssertionError("Daily execution does not reconcile with self-financing period wealth")
    return FirstPeriodExecution(fee, gross, net, gross_period, net_period,
                                holdings, initial_wealth*(1+net_period))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
