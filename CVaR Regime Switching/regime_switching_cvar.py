"""Stage-specific bull matched-return / bear minimum-CVaR MPO policy."""
from dataclasses import replace

import numpy as np

from regime_cvar import UPPER, cvar, max_sharpe_reference, validate_forecast


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
