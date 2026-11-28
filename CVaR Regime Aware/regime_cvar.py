"""Forecast-input adapter for the existing finite-scenario CVaR MPO solvers.

Expected returns and loss scenarios are separate inputs: changing a mean forecast
never translates or rescales the scenario distribution used to measure CVaR.
"""
from __future__ import annotations"""Regime-aware and regime-switching policies for the CVaR MPO solver.

Expected returns and loss scenarios are separate inputs: changing a mean forecast
never translates or rescales the scenario distribution used to measure CVaR.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'CVaR MPO and Scenarios'))
import cvar_mpo as cvar

POLICIES = ('min_cvar', 'matched_return')
SOURCES = ('HMM', 'DCC GARCH', 'DCC Hybrid')
UPPER = (1., 1., .2)
MAX_SHARPE_ANNUAL_RF = .04


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoints prevent a failed/interrupted write being reused.
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value, default=plain, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)


def load_returns(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if set(frame.columns) != set(cvar.ASSETS):
        raise ValueError(f'{path}: expected Equity, Bond and BTC columns')
    frame = frame.loc[:, list(cvar.ASSETS)].astype(float)
    if (frame.empty or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing or frame.index.tz is not None
            or not frame.index.equals(frame.index.normalize())
            or not np.isfinite(frame.to_numpy()).all() or (frame <= -1).any().any()):
        raise ValueError(f'{path}: invalid dates or simple returns')
    return frame


def validate_forecast(scenarios, expected_returns, holding_days):
    values = np.asarray(scenarios, float)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1
            or values.shape[2] != 3 or not np.isfinite(values).all()
            or (values <= -1).any()):
        raise ValueError('Require finite (scenario, period, 3) simple returns greater than -1')
    means = np.asarray(expected_returns, float)
    if (means.shape != values.shape[1:] or not np.isfinite(means).all()
            or (means <= -1).any()):
        raise ValueError('Expected returns must be finite (period, 3) simple returns greater than -1')
    days = np.asarray(holding_days)
    if (days.shape != (values.shape[1],) or not np.issubdtype(days.dtype, np.integer)
            or (days < 1).any()):
        raise ValueError('Provide a positive integer holding-day count per stage')
    return values, means, days


def max_sharpe_reference(scenarios, expected_returns, holding_days, *, upper=UPPER,
                         annual_risk_free=MAX_SHARPE_ANNUAL_RF):
    """Original multistart Sharpe/floor formula, with an explicit mean forecast.

    Covariance comes from each native scenario ensemble. This is the only
    extension of the reference used by cvar_mpo.optimize_matched_cvar, whose
    public wrapper intentionally requires native scenario sample means.
    The effective annual risk-free rate is compounded over each holding period.
    """
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
    if not np.isscalar(annual_risk_free) or not np.isfinite(annual_risk_free) or annual_risk_free <= -1:
        raise ValueError('Require a finite annual risk-free rate greater than -1')
    stage_risk_free = np.expm1(np.log1p(annual_risk_free)*days/252)
    caps = np.asarray(upper, float)
    if (caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any()
            or (caps > 1).any() or caps.sum() < 1):
        raise ValueError('Caps must admit a fully invested long-only portfolio')
    starts = [caps/caps.sum()]
    for asset in range(3):
        start = np.zeros(3)
        for j in [asset, *[k for k in range(3) if k != asset]]:
            start[j] = min(caps[j], max(0., 1-start.sum()))
        starts.append(start)
    covariances, weights = [], []
    for k, mean in enumerate(means):
        covariance = np.cov(values[:, k], rowvar=False, ddof=1)

        def objective(w):
            return -float(mean @ w-stage_risk_free[k])/np.sqrt(max(float(w @ covariance @ w), 1e-20))

        candidates = []
        for start in starts:
            result = minimize(objective, start, bounds=[(0., cap) for cap in caps],
                constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
            if (result.success and np.isfinite(result.fun)
                    and abs(result.x.sum()-1) < 1e-7 and (result.x >= -1e-7).all()
                    and (result.x <= caps+1e-7).all()):
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'Max-Sharpe reference failed in period {k+1}')
        weights.append(min(candidates, key=lambda r: r.fun).x)
        covariances.append(covariance)
    weights = np.asarray(weights)
    targets = np.sum(means*weights, axis=1)-1e-6*days
    return cvar.MatchedReturnTargets(means.copy(), weights, np.asarray(covariances),
        targets, days.copy(), {'method': 'Original multistart SLSQP with explicit forecast means',
            'floor_definition': 'Forecast mean dot max-Sharpe weights minus 1e-6 times holding days',
            'covariance_source': 'Unshifted native scenario sample covariance',
            'upper': caps.tolist(), 'risk_free': stage_risk_free.tolist(),
            'annual_risk_free': float(annual_risk_free)})


def optimize_forecast_cvar(scenarios, pretrade_weights, holding_days, expected_returns,
                           *, policy='min_cvar', mean_source='explicit forecasts',
                           scenario_source='supplied scenarios', alpha=.95,
                           risk_aversion=.5, cost_rate=.001, upper=UPPER,
                           use_cut_backend=True, cut_pool=None):
    """Apply min or matched-return CVaR to HMM or conventional forecasts.

    Both minimize .5*sum(stage CVaR) + estimated absolute-trade costs by default.
    Matched return additionally enforces the per-stage max-Sharpe return floor.
    Min CVaR reports the mean forecast but has no return reward or floor.

    Uses the exact same lower-level LP as both public policies in cvar_mpo.py;
    the adapter permits independent HMM/historical/future regression means.
    Only the first allocation is executed, using cvar.execute_first_period.
    """
    if policy not in POLICIES:
        raise ValueError(f'Policy must be one of {POLICIES}')
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
    reference = (max_sharpe_reference(values, means, days, upper=upper)
                 if policy == 'matched_return' else None)
    solver = cvar.solve_cvar_plan if use_cut_backend else cvar.solve_full_cvar_plan
    extra = {}
    if use_cut_backend:
        extra['cut_pool'] = cut_pool
    elif cut_pool is not None:
        raise ValueError('Cut pools require the cut backend')
    plan = solver(values, pretrade_weights, expected_returns=means, alpha=alpha,
        risk_aversion=risk_aversion, return_reward=0., cost_rate=cost_rate, upper=upper,
        min_return=None if reference is None else reference.targets, **extra)
    plan.metadata.update({'policy': policy, 'expected_return_source': mean_source,
        'scenario_source': scenario_source, 'holding_days': days.tolist(),
        'scenario_mean_translation': False, 'regime_volatility_overlay': False,
        'return_floor_reference': None if reference is None else {
            **reference.metadata, 'targets': reference.targets.tolist(),
            'weights': reference.weights.tolist()}})
    return plan


def regimes_from_bear_probabilities(probabilities):
    """Select the more likely forecast month-end state; reject exact ties."""
    probabilities = np.asarray(probabilities, float)
    if (probabilities.ndim != 1 or not len(probabilities)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError('Supply one finite bear probability in [0,1] per stage')
    if (probabilities == .5).any():
        raise ValueError('An exact probability tie needs an explicit selection rule')
    return np.where(probabilities < .5, 'bull', 'bear')


def optimize_switching_cvar(scenarios, pretrade_weights, holding_days, expected_returns,
                            stage_regimes, *, mean_source='explicit forecasts',
                            scenario_source='supplied scenarios', alpha=.95,
                            risk_aversion=.5, cost_rate=.001, upper=UPPER,
                            use_cut_backend=True, cut_pool=None):
    """Jointly plan all stages, imposing matched-return floors only in bulls.

    Every stage minimizes the same risk_aversion*CVaR + transaction-cost
    objective. The matched-return and minimum-CVaR policies differ only in their
    return-floor constraints. No extra return reward, regime-dependent penalty,
    scenario translation, or regime-contingent decision tree is introduced.

    The unchanged core LP requires a finite return-target vector. For a mixed
    plan, bear rows are encoded as 0 @ weights >= 0, an identity that imposes no
    restriction. Actual forecast means remain intact and are used for reported
    expected returns. All-bear plans pass min_return=None. All-bull plans are
    the ordinary matched-return policy. Original solver APIs are unchanged.
    """
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
    regimes = np.asarray(stage_regimes)
    if regimes.shape != (len(days),) or not np.isin(regimes, ['bull', 'bear']).all():
        raise ValueError('Provide bull or bear for each planning stage')
    bull = regimes == 'bull'
    reference = None
    active_targets = [None]*len(days)
    constraint_means = means.copy()
    targets = None
    if bull.any():
        reference = max_sharpe_reference(values[:, bull], means[bull], days[bull], upper=upper)
        targets = np.zeros(len(days))
        targets[bull] = reference.targets
        constraint_means[~bull] = 0.
        for k, target in zip(np.flatnonzero(bull), reference.targets):
            active_targets[k] = float(target)
    solver = cvar.solve_cvar_plan if use_cut_backend else cvar.solve_full_cvar_plan
    extra = {}
    if use_cut_backend:
        extra['cut_pool'] = cut_pool
    elif cut_pool is not None:
        raise ValueError('Cut pools require the cut backend')
    plan = solver(values, pretrade_weights, expected_returns=constraint_means,
        min_return=targets, return_reward=0., alpha=alpha, risk_aversion=risk_aversion,
        cost_rate=cost_rate, upper=upper, **extra)
    # Only forecast reporting needs restoration: return_reward=0 means the
    # objective and the risk/fee accounting never depend on these bear rows.
    plan = replace(plan, expected_returns=np.sum(means*plan.weights, axis=1))
    plan.metadata.update({'policy': 'regime_switching',
        'stage_regimes': regimes.tolist(),
        'stage_policies': ['matched_return' if flag else 'min_cvar' for flag in bull],
        'bull_stage_mask': bull.tolist(), 'holding_days': days.tolist(),
        'expected_return_source': mean_source, 'scenario_source': scenario_source,
        'gross_return_targets': active_targets,
        'solver_gross_return_targets': None if targets is None else targets.tolist(),
        'solver_return_constraint_means': constraint_means.tolist(),
        'bear_constraint_encoding': 'No target, or the identity 0 @ weights >= 0 in a mixed plan',
        'scenario_mean_translation': False, 'regime_volatility_overlay': False,
        'return_floor_reference': None if reference is None else {
            **reference.metadata, 'stages': (np.flatnonzero(bull)+1).tolist(),
            'targets': reference.targets.tolist(), 'weights': reference.weights.tolist()}})
    return plan


from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'CVaR MPO and Scenarios'))
import cvar_mpo as cvar

POLICIES = ('min_cvar', 'matched_return')
SOURCES = ('HMM', 'DCC GARCH', 'DCC Hybrid')
UPPER = (1., 1., .2)
MAX_SHARPE_ANNUAL_RF = .04


def validate_forecast(scenarios, expected_returns, holding_days):
    values = np.asarray(scenarios, float)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1
            or values.shape[2] != 3 or not np.isfinite(values).all()
            or (values <= -1).any()):
        raise ValueError('Require finite (scenario, period, 3) simple returns greater than -1')
    means = np.asarray(expected_returns, float)
    if (means.shape != values.shape[1:] or not np.isfinite(means).all()
            or (means <= -1).any()):
        raise ValueError('Expected returns must be finite (period, 3) simple returns greater than -1')
    days = np.asarray(holding_days)
    if (days.shape != (values.shape[1],) or not np.issubdtype(days.dtype, np.integer)
            or (days < 1).any()):
        raise ValueError('Provide a positive integer holding-day count per stage')
    return values, means, days


def max_sharpe_reference(scenarios, expected_returns, holding_days, *, upper=UPPER,
                         annual_risk_free=MAX_SHARPE_ANNUAL_RF):
    """Original multistart Sharpe/floor formula, with an explicit mean forecast.

    Covariance comes from each native scenario ensemble. This is the only
    extension of the reference used by cvar_mpo.optimize_matched_cvar, whose
    public wrapper intentionally requires native scenario sample means.
    The effective annual risk-free rate is compounded over each holding period.
    """
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
    if not np.isscalar(annual_risk_free) or not np.isfinite(annual_risk_free) or annual_risk_free <= -1:
        raise ValueError('Require a finite annual risk-free rate greater than -1')
    stage_risk_free = np.expm1(np.log1p(annual_risk_free)*days/252)
    caps = np.asarray(upper, float)
    if (caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any()
            or (caps > 1).any() or caps.sum() < 1):
        raise ValueError('Caps must admit a fully invested long-only portfolio')
    starts = [caps/caps.sum()]
    for asset in range(3):
        start = np.zeros(3)
        for j in [asset, *[k for k in range(3) if k != asset]]:
            start[j] = min(caps[j], max(0., 1-start.sum()))
        starts.append(start)
    covariances, weights = [], []
    for k, mean in enumerate(means):
        covariance = np.cov(values[:, k], rowvar=False, ddof=1)

        def objective(w):
            return -float(mean @ w-stage_risk_free[k])/np.sqrt(max(float(w @ covariance @ w), 1e-20))

        candidates = []
        for start in starts:
            result = minimize(objective, start, bounds=[(0., cap) for cap in caps],
                constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
            if (result.success and np.isfinite(result.fun)
                    and abs(result.x.sum()-1) < 1e-7 and (result.x >= -1e-7).all()
                    and (result.x <= caps+1e-7).all()):
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'Max-Sharpe reference failed in period {k+1}')
        weights.append(min(candidates, key=lambda r: r.fun).x)
        covariances.append(covariance)
    weights = np.asarray(weights)
    targets = np.sum(means*weights, axis=1)-1e-6*days
    return cvar.MatchedReturnTargets(means.copy(), weights, np.asarray(covariances),
        targets, days.copy(), {'method': 'Original multistart SLSQP with explicit forecast means',
            'floor_definition': 'Forecast mean dot max-Sharpe weights minus 1e-6 times holding days',
            'covariance_source': 'Unshifted native scenario sample covariance',
            'upper': caps.tolist(), 'risk_free': stage_risk_free.tolist(),
            'annual_risk_free': float(annual_risk_free)})


def optimize_forecast_cvar(scenarios, pretrade_weights, holding_days, expected_returns,
                           *, policy='min_cvar', mean_source='explicit forecasts',
                           scenario_source='supplied scenarios', alpha=.95,
                           risk_aversion=.5, cost_rate=.001, upper=UPPER,
                           use_cut_backend=True, cut_pool=None):
    """Apply min or matched-return CVaR to HMM or conventional forecasts.

    Both minimize .5*sum(stage CVaR) + estimated absolute-trade costs by default.
    Matched return additionally enforces the per-stage max-Sharpe return floor.
    Min CVaR reports the mean forecast but has no return reward or floor.

    Uses the exact same lower-level LP as both public policies in cvar_mpo.py;
    the adapter permits independent HMM/historical/future regression means.
    Only the first allocation is executed, using cvar.execute_first_period.
    """
    if policy not in POLICIES:
        raise ValueError(f'Policy must be one of {POLICIES}')
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
    reference = (max_sharpe_reference(values, means, days, upper=upper)
                 if policy == 'matched_return' else None)
    solver = cvar.solve_cvar_plan if use_cut_backend else cvar.solve_full_cvar_plan
    extra = {}
    if use_cut_backend:
        extra['cut_pool'] = cut_pool
    elif cut_pool is not None:
        raise ValueError('Cut pools require the cut backend')
    plan = solver(values, pretrade_weights, expected_returns=means, alpha=alpha,
        risk_aversion=risk_aversion, return_reward=0., cost_rate=cost_rate, upper=upper,
        min_return=None if reference is None else reference.targets, **extra)
    plan.metadata.update({'policy': policy, 'expected_return_source': mean_source,
        'scenario_source': scenario_source, 'holding_days': days.tolist(),
        'scenario_mean_translation': False, 'regime_volatility_overlay': False,
        'return_floor_reference': None if reference is None else {
            **reference.metadata, 'targets': reference.targets.tolist(),
            'weights': reference.weights.tolist()}})
    return plan
