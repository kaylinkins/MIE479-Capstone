"""Self-contained matched-return CVaR MPO and exact execution accounting.

Public entry: optimize_matched_cvar(scenarios, pretrade_weights, holding_days).
Scenarios are decimal incremental returns with shape (scenario, period, 3),
ordered Equity/Bond/BTC. The restored policy uses native means and original
maximum-Sharpe floors minus 1e-6 per holding day. Defaults: lambda0.5 in the
runner, CVaR95, BTC20%, buy/sell costs10bps. Only month one is executed.

Risk is the SUM of stage CVaRs, not compounded terminal-wealth or nested CVaR.
Future planned trades ignore future return drift; realized execution uses exact
self-financing fees and daily holding drift. No data or file is loaded here.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from scipy.optimize import brentq, linprog, minimize
from scipy.sparse import block_diag, csr_matrix, diags, hstack, identity, kron, vstack

ASSETS = ("Equity", "Bond", "BTC")
DEFAULT_UPPER = (1., 1., 1.)
SEPARATION_TOLERANCE = 1e-10
MAX_ITERATIONS = 200

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


def solve_full_cvar_plan(scenarios, pretrade_weights, *, expected_returns=None,
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


class CVaRCutPool:
    """Reusable valid stage cuts for the same scenarios and confidence.

    One pool may be reused across return targets, risk aversion, prior holdings
    and H=1/H=3 prefixes. Fingerprints prevent accidental reuse for a different
    model, origin or empirical ensemble. This is an in-memory acceleration;
    every solve still performs full-ensemble separation before accepting it.
    """
    def __init__(self):
        self._alpha = None
        self._fingerprints = []
        self._cuts = []

    def _prepare(self, values, alpha):
        if self._alpha is not None and self._alpha != alpha:
            raise ValueError("Reusable CVaR cuts require the same confidence")
        fingerprints = [sha256(np.ascontiguousarray(values[:, k]).tobytes()).hexdigest()
                        for k in range(values.shape[1])]
        common = min(len(fingerprints), len(self._fingerprints))
        if fingerprints[:common] != self._fingerprints[:common]:
            raise ValueError("Reusable CVaR cuts require identical scenario/stage ensembles")
        self._alpha = float(alpha)
        for fingerprint in fingerprints[len(self._fingerprints):]:
            self._fingerprints.append(fingerprint)
            self._cuts.append([])
        return self._cuts[:values.shape[1]]


def tail_average_plane(scenarios, weights, alpha=.95):
    """Return a valid supporting plane of empirical CVaR at ``weights``.

    Every plane is a feasible tail-probability-weighted average of losses:
    q>=0, sum(q)=1 and q_i<=1/((1-alpha)*N). The fractional boundary observation
    receives only its remaining mass, as in the original LP. Ties can choose
    any subset without affecting the oracle's value or the validity of a cut.
    """
    values = np.asarray(scenarios, float)
    weights = np.asarray(weights, float)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values) or
            weights.shape != (3,) or not np.isfinite(values).all() or
            not np.isfinite(weights).all() or not 0 < alpha < 1):
        raise ValueError("Require finite (scenario, 3) returns, three weights and confidence in (0,1)")
    losses = -(values @ weights)
    mass = (1-alpha)*len(values)
    whole = int(np.floor(mass))
    fraction = mass-whole
    count = whole+int(fraction > 1e-12)
    # When floating-point mass differs negligibly from an integer, retaining
    # the tiny extra mass makes the cut exactly probability-normalized.
    if fraction > 0 and count == whole:
        count += 1
    count = max(1, min(count, len(values)))
    selected = np.argpartition(losses, len(values)-count)[-count:]
    selected = selected[np.argsort(-losses[selected], kind="stable")]
    plane = -values[selected[:whole]].sum(axis=0)
    if fraction > 0:
        plane -= fraction*values[selected[whole]]
    plane /= mass
    return plane


def solve_cvar_plan(scenarios, pretrade_weights, *, expected_returns=None,
                    alpha=.95, risk_aversion=1., return_reward=1., cost_rate=.001,
                    upper=DEFAULT_UPPER, min_return=None, risk_budget=None,
                    terminal_weights=None, cut_pool=None):
    """Exact counterpart of the full scenario LP, with the same public API."""
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

    # Variables [weights(H*3), CVaR epigraph(H), absolute trades(H*3)].
    nw = horizon*assets
    variables = 2*nw+horizon
    objective = np.r_[-return_reward*means.ravel(), np.full(horizon, risk_aversion),
                       np.tile(costs, horizon)]
    rows, rhs = [], []
    for index in range(nw):
        forward = np.zeros(variables)
        forward[index] = 1.
        if index >= assets:
            forward[index-assets] = -1.
        forward[nw+horizon+index] = -1.
        reverse = -forward
        reverse[nw+horizon+index] = -1.
        rows.extend([forward, reverse])
        rhs.extend([prior[index] if index < assets else 0.,
                    -prior[index] if index < assets else 0.])
    if targets is not None:
        for k in range(horizon):
            row = np.zeros(variables)
            row[k*assets:(k+1)*assets] = -means[k]
            rows.append(row)
            rhs.append(-targets[k])
    if budgets is not None:
        for k in range(horizon):
            row = np.zeros(variables)
            row[nw+k] = 1.
            rows.append(row)
            rhs.append(budgets[k])
    equality = np.zeros((horizon, variables))
    for k in range(horizon):
        equality[k, k*assets:(k+1)*assets] = 1.
    bounds = [(0., float(cap)) for _ in range(horizon) for cap in caps]
    if terminal is not None:
        bounds[-assets:] = [(float(w), float(w)) for w in terminal]
    bounds += [(None, None)]*horizon+[(0., None)]*nw
    cut_keys = [set() for _ in range(horizon)]
    cuts_added = 0
    shared_cuts = (CVaRCutPool() if cut_pool is None else cut_pool)
    if not isinstance(shared_cuts, CVaRCutPool):
        raise ValueError("cut_pool must be a CVaRCutPool instance")
    stage_pools = shared_cuts._prepare(values, alpha)

    def add_cut(k, plane, *, save=True):
        nonlocal cuts_added
        key = tuple(np.round(plane, 14))
        if key in cut_keys[k]:
            return False
        cut_keys[k].add(key)
        row = np.zeros(variables)
        row[k*assets:(k+1)*assets] = plane
        row[nw+k] = -1.
        rows.append(row)
        rhs.append(0.)
        cuts_added += 1
        if save:
            stage_pools[k].append(plane.copy())
        return True

    warm_cuts = sum(len(cuts) for cuts in stage_pools)
    for k in range(horizon):
        for plane in stage_pools[k]:
            add_cut(k, plane, save=False)

    # Corner cuts give a bounded master and include both defensive/aggressive
    # directions, while retaining all feasible portfolios in its relaxation.
    initial = list(np.eye(assets))+[np.full(assets, 1/assets), prior]
    for k in range(horizon):
        for weights in initial:
            add_cut(k, tail_average_plane(values[:, k], weights, alpha))

    fallback_reason = None
    max_gap = np.inf
    solved = None
    for iteration in range(1, MAX_ITERATIONS+1):
        solved = linprog(objective, A_ub=csr_matrix(np.asarray(rows)), b_ub=np.asarray(rhs),
                         A_eq=csr_matrix(equality), b_eq=np.ones(horizon), bounds=bounds,
                         method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                                  "dual_feasibility_tolerance": 1e-10})
        if not solved.success:
            fallback_reason = f"Master failed: {solved.message}"
            break
        weights = solved.x[:nw].reshape(horizon, assets)
        planes = [tail_average_plane(values[:, k], weights[k], alpha) for k in range(horizon)]
        exact_es = np.array([plane @ weights[k] for k, plane in enumerate(planes)])
        gaps = exact_es-solved.x[nw:nw+horizon]
        max_gap = float(max(0., gaps.max()))
        budget_ok = budgets is None or np.all(exact_es <= budgets+SEPARATION_TOLERANCE)
        if max_gap <= SEPARATION_TOLERANCE and budget_ok:
            break
        added = 0
        for k, plane in enumerate(planes):
            if gaps[k] > SEPARATION_TOLERANCE or (budgets is not None and exact_es[k] > budgets[k]+SEPARATION_TOLERANCE):
                added += int(add_cut(k, plane))
        if not added:
            fallback_reason = "Separation stalled before certification"
            break
    else:
        fallback_reason = "Maximum separation iterations reached"

    if fallback_reason is not None:
        plan = solve_full_cvar_plan(values, prior, expected_returns=expected_returns,
            alpha=alpha, risk_aversion=risk_aversion, return_reward=return_reward,
            cost_rate=costs, upper=caps, min_return=targets, risk_budget=budgets,
            terminal_weights=terminal)
        plan.metadata.update({"backend": "full LP fallback from exact constraint generation",
            "constraint_generation_fallback": fallback_reason,
            "constraint_generation_iterations": iteration, "tail_cuts": cuts_added,
            "warm_tail_cuts": warm_cuts, "scenario_reduction": False,
            "separation_tolerance": SEPARATION_TOLERANCE,
            "last_separation_gap": max_gap,
            "minimization_lower_bound": float(plan.solver_minimum),
            "minimization_upper_bound": float(-plan.objective),
            "objective_certification_gap": max(0., float(-plan.objective-plan.solver_minimum))})
        return plan

    trades = np.diff(np.vstack([prior, weights]), axis=0)
    expected = (means*weights).sum(axis=1)
    var, es = np.asarray([loss_var_cvar(values[:, k] @ weights[k], alpha) for k in range(horizon)]).T
    estimated_costs = (np.abs(trades)*costs).sum(axis=1)
    value = float(return_reward*expected.sum()-risk_aversion*es.sum()-estimated_costs.sum())
    feasible_minimum = -value
    objective_gap = feasible_minimum-float(solved.fun)
    if (not np.allclose(weights.sum(axis=1), 1., atol=1e-8, rtol=0) or
            (weights < -1e-8).any() or (weights > caps+1e-8).any() or
            (targets is not None and (expected < targets-1e-8).any()) or
            (budgets is not None and (es > budgets+1e-8).any()) or
            objective_gap < -1e-8 or objective_gap > horizon*risk_aversion*SEPARATION_TOLERANCE+1e-8):
        raise AssertionError("Constraint-generation solution failed feasibility/objective certification")
    metadata = {"horizon": horizon, "scenario_count": n, "assets": list(ASSETS),
        "risk_measure": "sum of gross per-period CVaRs; not terminal-wealth or nested CVaR",
        "risk_aversion": float(risk_aversion), "return_reward": float(return_reward),
        "expected_return_source": "explicit common forecasts" if expected_returns is not None else "native scenario means",
        "planning_dynamics": "future trade = next planned weights minus previous planned weights; return drift ignored only in planning",
        "execution": "execute first allocation only; apply exact drift and self-financing fees, then replan",
        "terminal_policy": "fixed last-stage allocation" if terminal is not None else "no forced liquidation or terminal fee",
        "solver": "scipy.optimize.linprog(method='highs'); exact tail-average constraint generation",
        "solver_message": solved.message, "upper": caps.tolist(),
        "gross_return_targets": None if targets is None else targets.tolist(),
        "gross_cvar_budgets": None if budgets is None else budgets.tolist(),
        "backend": "exact finite-ensemble CVaR constraint generation",
        "constraint_generation_fallback": None,
        "constraint_generation_iterations": iteration, "tail_cuts": cuts_added,
        "warm_tail_cuts": warm_cuts,
        "separation_tolerance": SEPARATION_TOLERANCE,
        "max_separation_gap": max_gap,
        "minimization_lower_bound": float(solved.fun),
        "minimization_upper_bound": feasible_minimum,
        "objective_certification_gap": max(0., float(objective_gap)),
        "scenario_reduction": False}
    return CVaRPlan(weights.copy(), trades.copy(), prior, expected, var, es,
                    estimated_costs, costs.copy(), value, float(solved.fun), alpha, metadata)


def _maximum_sharpe(returns, upper=None, risk_free=0.0):
    """Maximize Sharpe using returns and risk-free rate on the same horizon."""
    returns = np.asarray(returns, dtype=float)
    mean = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False)
    upper = np.ones(returns.shape[1]) if upper is None else np.asarray(upper, dtype=float)
    if not np.isfinite(returns).all() or upper.sum() < 1:
        raise ValueError("Invalid returns or infeasible weight caps")

    def objective(weights):
        variance = float(weights @ covariance @ weights)
        return -float(mean @ weights-risk_free)/np.sqrt(max(variance, 1e-20))

    # Multiple feasible starts reduce sensitivity to the initial allocation.
    starts = [upper/upper.sum()]
    for asset in range(len(upper)):
        weights = np.zeros(len(upper))
        for index in [asset, *[j for j in range(len(upper)) if j != asset]]:
            weights[index] = min(upper[index], max(0.0, 1-weights.sum()))
        starts.append(weights)
    solutions = []
    for start in starts:
        result = minimize(objective, start, bounds=[(0, cap) for cap in upper],
                          constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        if (result.success and np.isfinite(result.fun) and
                abs(result.x.sum()-1) < 1e-7 and
                (result.x >= -1e-7).all() and (result.x <= upper+1e-7).all()):
            solutions.append(result)
    if not solutions:
        raise RuntimeError("Max-Sharpe optimization failed for every starting allocation")
    return min(solutions, key=lambda result: result.fun).x


@dataclass(frozen=True)
class MatchedReturnTargets:
    expected_returns: np.ndarray
    weights: np.ndarray
    covariances: np.ndarray
    targets: np.ndarray
    holding_days: np.ndarray
    metadata: dict


def matched_return_targets(scenarios, holding_days, *, upper=(1., 1., .2), risk_free=0., native_layout='C'):
    """Use each stage's native mean and the original maximum-Sharpe routine."""
    values = np.asarray(scenarios, float)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] != 3
            or not np.isfinite(values).all() or (values <= -1).any()):
        raise ValueError('Require finite (scenario, period, 3) simple returns greater than -1')
    days = np.asarray(holding_days)
    if (days.shape != (values.shape[1],) or not np.issubdtype(days.dtype, np.integer) or (days < 1).any()):
        raise ValueError('Provide a positive integer holding-day count for every stage')
    rates = np.asarray(risk_free, float)
    if rates.ndim == 0:
        rates = np.full(values.shape[1], float(rates))
    if rates.shape != days.shape or not np.isfinite(rates).all():
        raise ValueError('Risk-free returns must be finite monthly scalar or stage vector')
    caps = np.asarray(upper, float)
    if caps.shape != (3,) or (caps < 0).any() or (caps > 1).any() or not np.isfinite(caps).all() or caps.sum() < 1:
        raise ValueError('Allocation caps must admit a fully invested long-only portfolio')
    if native_layout not in ('C', 'F'):
        raise ValueError('Native layout must be C or F')
    # Original simulated matrices are C-contiguous; historical DataFrame.to_numpy
    # matrices are Fortran-contiguous. Preserve reduction order because finite-
    # difference SLSQP can change its final target slightly after tiny roundoff.
    stages = [np.array(values[:, k], order=native_layout, copy=True) for k in range(values.shape[1])]
    means = np.array([stage.mean(axis=0) for stage in stages])
    weights = np.array([_maximum_sharpe(stage, upper=caps, risk_free=rates[k])
                        for k, stage in enumerate(stages)])
    covariances = np.array([np.cov(stage, rowvar=False, ddof=1) for stage in stages])
    targets = np.sum(means*weights, axis=1)-1e-6*days
    return MatchedReturnTargets(means, weights, covariances, targets, days.copy(), {
        'method': 'Original maximum-Sharpe reference; multistart SLSQP',
        'mean_source': 'Native stage scenario sample means',
        'floor_definition': 'Native mean dot original maximum-Sharpe weights minus 1e-6 times holding days',
        'daily_floor_allowance': 1e-6, 'upper': caps.tolist(), 'risk_free': rates.tolist(),
        'native_array_layout': native_layout})


def optimize_matched_cvar(scenarios, pretrade_weights, holding_days, *,
                                   upper=(1., 1., .2), risk_free=0., reference=None,
                                   use_cut_backend=True, cut_pool=None, native_layout='C', **settings):
    """Minimize lambda*sum(stage CVaR)+costs using original native return floors.

    Only execution of the first allocation belongs to the caller. Costs stay in
    the planning objective unless explicitly set to zero for a replay diagnostic.
    """
    if 'min_return' in settings or 'expected_returns' in settings:
        raise ValueError('Original matched policy derives native means and return floors')
    reward = settings.pop('return_reward', 0.)
    if not np.isscalar(reward) or not np.isfinite(reward) or reward != 0:
        raise ValueError('Original matched policy requires return_reward=0')
    settings.setdefault('risk_aversion', .5)
    values = np.asarray(scenarios, float)
    if values.ndim != 3:
        raise ValueError('Scenarios must have three dimensions')
    h = values.shape[1]
    if reference is None:
        reference = matched_return_targets(values, holding_days, upper=upper, risk_free=risk_free,
                                            native_layout=native_layout)
    if (reference.expected_returns.shape[0] < h
            or not np.allclose(reference.expected_returns[:h], values.mean(axis=0), atol=1e-15, rtol=0)
            or not np.array_equal(reference.holding_days[:h], holding_days)
            or not np.array_equal(reference.metadata['upper'], upper)):
        raise ValueError('Precomputed reference does not match the native forecast or caps')
    solver = solve_full_cvar_plan
    extra = {}
    if use_cut_backend:
        solver = solve_cvar_plan
        extra['cut_pool'] = cut_pool
    elif cut_pool is not None:
        raise ValueError('Cut pools require the exact constraint-generation backend')
    plan = solver(values, pretrade_weights, expected_returns=reference.expected_returns[:h],
        min_return=reference.targets[:h], return_reward=0., upper=upper, **extra, **settings)
    plan.metadata.update({'policy': 'Original scenario-mean matched-return CVaR plus current planning costs',
        'expected_return_source': 'Native stage scenario sample means',
        'return_floor_reference': {**reference.metadata, 'targets': reference.targets[:h].tolist(),
            'weights': reference.weights[:h].tolist(), 'holding_days': reference.holding_days[:h].tolist()}})
    return plan
