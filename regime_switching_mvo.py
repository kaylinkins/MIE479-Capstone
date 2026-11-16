"""Model 2: Regime-switching MVO MPO, bull -> Max Sharpe, bear -> Min Var (Bonnie).

The optimization RULE switches with the forecast regime, and it switches at
EACH PLANNING STAGE: in a plan of H months, month k uses

    Max Sharpe (MS)  if month k is forecast bull
    Min Var    (MV)  if month k is forecast bear

where month k is bull when its forecast month-end bear probability is below
0.5 and bear when it is above 0.5; an exact tie raises an error. This is the
same selection rule as Kaylin's regime-switching CVaR
(regime_switching_cvar.regimes_from_bear_probabilities).
"Bull"/"bear" are the low/high equity-variance states of the regime model,
not necessarily positive/negative returns.

Example: P(bear) = 0.73, 0.51, 0.38 for months 1-3 gives the plan MV-MV-MS:
hold the min-variance portfolio now and plan the move back to max-Sharpe
weights in month 3, paying trading costs along the way. Only month 1 is
executed; next month the plan is rebuilt from new forecasts.

Both rules share one objective (variance + trading cost; MS adds the
matched-return constraint mu_k'w_k >= mu_k'w_k^MS), so a mixed plan is a
single convex QP solved by regime_aware_mvo.solve_stages. This file adds only
the switching rule on top of Model 1's engine; keep the two files together.

    weights, details = solve_plan(current_weights, forecast, horizon, caps)   # horizon H = 1, 2 or 3

The forecast must carry p_bear (one bear probability per planned month).
"""
from __future__ import annotations

import regime_aware_mvo as engine

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

BEAR_THRESHOLD = 0.5     # bull if P(bear) < 0.5, bear if P(bear) > 0.5, tie = error (as in the CVaR model)
POLICY = "Regime switch (MS -> MV)"
RULE_BY_REGIME = {"bull": "MS", "bear": "MV"}


def regime_path(p_bear, horizon, threshold=BEAR_THRESHOLD):
    """Forecast regime ('bull' / 'bear') for each planned month.

    Matches the CVaR switching model: the more likely month-end state is
    selected, and an exact tie is rejected rather than broken arbitrarily.
    """
    probabilities = [float(p) for p in p_bear[:horizon]]
    if any(not 0.0 <= p <= 1.0 for p in probabilities):
        raise ValueError("Bear probabilities must lie in [0, 1]")
    if any(p == threshold for p in probabilities):
        raise ValueError("An exact probability tie needs an explicit selection rule")
    return ["bull" if p < threshold else "bear" for p in probabilities]


def stage_rules(p_bear, horizon, threshold=BEAR_THRESHOLD):
    """'MS' for each forecast-bull month, 'MV' for each forecast-bear month."""
    if p_bear is None:
        raise ValueError("Regime switching needs a bear probability for each planned month")
    return [RULE_BY_REGIME[regime] for regime in regime_path(p_bear, horizon, threshold)]


def solve_plan(current_weights, forecast, horizon, caps,
               cost_rate=engine.COST_RATE, gamma=engine.RISK_AVERSION,
               threshold=BEAR_THRESHOLD):
    """Regime-switching MVO MPO: the rule for each planned month follows its regime.
    horizon: H = 1, 2 or 3 planned months."""
    engine.check_horizon(horizon, forecast)
    if forecast.p_bear is not None and len(forecast.p_bear) < horizon:
        raise ValueError(f"Need a bear probability for each of the {horizon} planned months")
    stages = stage_rules(forecast.p_bear, horizon, threshold)
    return engine.solve_stages(current_weights, forecast, stages, caps, cost_rate, gamma)