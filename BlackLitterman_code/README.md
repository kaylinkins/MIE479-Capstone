# Black-Litterman added to the baseline models

Install once: `python -m pip install -r requirements.txt`

The scripts find `portfolio_train.csv` / `portfolio_validation.csv` beside
themselves or in a parent folder, so they run from this folder as is.
Results go to `bl_outputs_v2/`. BL settings (lambda, tau, benchmark weights,
view confidence, BTC cap) are at the top of `bl_core.py`.

Run in this order (mirrors report Section 2.5):

1. `python bl_insample.py` – like baseline_models.py + cvar_optimization.py
   (Table 4): BL inputs, every baseline model with historical vs BL means on the
   training set, and the CVaR frontier with historical vs BL targets.
2. `python bl_walk_forward.py` – like the new walk_forward.py (Table 5): monthly
   walk-forward with a 5-year rolling window and overlapping month-length
   historical scenarios, for every model, with BL versions of max-Sharpe MVO and
   matched-return CVaR added. BL is fitted on the same scenarios each month.

The historical-mean rows reproduce Table 4 (in-sample) and Table 5 (walk-forward).
This folder is self-contained: it does not need portfolio_optimization.py or
cvar_scenario_comparison.py.

Specification: Sigma = sample covariance (no shrinkage), Q = mean of the
estimation window (daily returns in-sample, month-length scenarios walk-forward), P = identity, pi = lambda * Sigma * w_b,
Omega_jj = (1-c_j)/c_j * p_j tau Sigma p_j', risk-free = 0 (as in the baselines).
