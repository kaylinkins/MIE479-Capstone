"""Forecast-input adapter for the existing finite-scenario CVaR MPO solvers.

Expected returns and loss scenarios are separate inputs: changing a mean forecast
never translates or rescales the scenario distribution used to measure CVaR.
"""
from __future__ import annotations

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


def max_sharpe_reference(scenarios, expected_returns, holding_days, *, upper=UPPER):
    """Original multistart Sharpe/floor formula, with an explicit mean forecast.

    Covariance comes from each native scenario ensemble. This is the only
    extension of the reference used by cvar_mpo.optimize_matched_cvar, whose
    public wrapper intentionally requires native scenario sample means.
    Risk-free return is zero, as in that wrapper's default policy.
    """
    values, means, days = validate_forecast(scenarios, expected_returns, holding_days)
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
            return -float(mean @ w)/np.sqrt(max(float(w @ covariance @ w), 1e-20))

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
            'upper': caps.tolist(), 'risk_free': 0.})


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
