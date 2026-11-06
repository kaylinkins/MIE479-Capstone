"""Capstone: DCC(1,1) with GARCH Equity, EWMA Bond, GARCH Bitcoin.

Fair calendar-month horizon comparison of joint-return simulators.

Compare four scenario generators with the SAME marginal GARCH fits,
shared historical windows, observed holding-day counts, simulation count,
and random historical shock row indices:
  (1) All-GARCH FHS: GARCH(1,1) on each asset, empirical joint resampling
  (2) Mixed FHS: GARCH Equity/BTC, EWMA Bond, empirical joint resampling
  (3) DCC-GARCH: all-GARCH marginals with DCC dependence
  (4) HYBRID DCC: GARCH Equity/BTC, EWMA Bond, DCC re-fitted to hybrid
                  standardized residuals. Empirical whitened innovations,
                  NOT multivariate Student-t innovations.

At the beginning of each VALIDATION calendar month, re-estimate using ONLY
prior five years. Use that month's observed number of trading days as the
simulated horizon, but never use subsequent returns to produce a forecast.
Only the number of trading days is known here retrospectively; for genuinely
forward-operated monthly rebalancing use a known exchange trading calendar.

Files needed alongside this script:
    portfolio_train.csv
    portfolio_validation.csv
    capstone_dcc_garch.py      (the script previously provided)

Outputs in hybrid_dcc_results/. Run: py capstone_dcc_ewma_hybrid.py
Requires: numpy, pandas, scipy, arch, matplotlib (optional charts).
"""
from __future__ import annotations

# Support direct execution after organizing the source folders.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_MODELLING_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents
                      if (p / "project_paths.py").is_file())
if str(_MODELLING_ROOT) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_MODELLING_ROOT))
from project_paths import (PROJECT_ROOT, MODELLING_ROOT, source_path,
                           cache_source_bytes, cache_source_name)


from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

import importlib.util

_dcc_path = Path(__file__).resolve().with_name("DCC-GARCH Gaussian.py")
_dcc_spec = importlib.util.spec_from_file_location("capstone_dcc_gaussian", _dcc_path)
dcc = importlib.util.module_from_spec(_dcc_spec)
_dcc_spec.loader.exec_module(dcc)

ROOT = MODELLING_ROOT
DATA_DIR = ROOT.parent / "Data" / "datasets"
OUT = ROOT.parent / "outputs" / "volatility" / "hybrid_dcc_results"
ROLLING_YEARS = 5
MIN_WINDOW_OBS = 750
N_SCENARIOS = 10_000
SEED = 42
EWMA_LAMBDA = 0.94
TAIL_PROB = 0.05
PORTFOLIO_WEIGHTS = {"Equity": 1/3, "Bond": 1/3, "BTC": 1/3}
VOL_CAP_MULTIPLE = 1.5
ENERGY_PAIR_REPEATS = 8  # independent permutations reduce Monte Carlo noise
BOOTSTRAP_REPETITIONS = 5000
SAVE_MATCHED_SCENARIOS = True
MODELS = ("All-GARCH FHS", "GARCH-EWMA-GARCH FHS", "DCC-GARCH", "DCC-GARCH-EWMA-GARCH")
ASSET_PAIRS = (("Equity", "Bond"), ("Equity", "BTC"), ("Bond", "BTC"))


def validate_inputs(train, val):
    required = {"Equity", "Bond", "BTC"}
    if set(train.columns) != required or set(val.columns) != required:
        raise ValueError("Inputs must contain exactly Equity, Bond, BTC")
    if train.index[-1] >= val.index[0]:
        raise ValueError("Train and validation dates overlap")
    if train.index.has_duplicates or val.index.has_duplicates:
        raise ValueError("Duplicate dates")
    return train[list(("Equity", "Bond", "BTC"))], val[list(("Equity", "Bond", "BTC"))]


def ewma_bond(window_pct):
    """Bond EWMA conditional variance, filtered one-day-ahead (percent^2)."""
    x = window_pct["Bond"].to_numpy(float)
    mean = x.mean()
    eps = x-mean
    var = np.empty(len(x), dtype=float)
    var[0] = max(float(np.var(x[:min(len(x), 60)], ddof=1)), 1e-10)
    for t in range(1, len(x)):
        var[t] = EWMA_LAMBDA*var[t-1] + (1-EWMA_LAMBDA)*eps[t-1]**2
    raw = eps / np.sqrt(np.maximum(var,1e-10))
    z = (raw-raw.mean())/raw.std(ddof=1)
    next_v = EWMA_LAMBDA*var[-1] + (1-EWMA_LAMBDA)*eps[-1]**2
    cap = (VOL_CAP_MULTIPLE * np.sqrt(var.max()))**2 if VOL_CAP_MULTIPLE is not None else np.inf
    return dict(mu=float(mean), next_var=float(next_v), vol_cap=float(cap)),z


def standard_fhs(states, Z, sample_indices, ewma_bond_state=None):
    """GARCH joint FHS, optionally use the EWMA variance recursion for bond."""
    n_paths, horizon = sample_indices.shape
    variances = np.broadcast_to(np.array([s["next_var"] for s in states],float),
                                (n_paths,3)).copy()
    cap = np.array([s["vol_cap"] for s in states], float)
    mean = np.array([s["scenario_mu"] for s in states],float)
    omega = np.array([s["omega"] for s in states],float)
    alpha = np.array([s["alpha"] for s in states],float)
    beta = np.array([s["beta"] for s in states],float)
    if ewma_bond_state is not None:
        j=1  # required input order: Equity,Bond,BTC
        variances[:,j] = ewma_bond_state["next_var"]
        cap[j] = ewma_bond_state["vol_cap"]
        mean[j] = ewma_bond_state["mu"]
        omega[j],alpha[j],beta[j] = 0.,1-EWMA_LAMBDA,EWMA_LAMBDA
    wealth = np.ones((n_paths,3))
    cap_hits,clip_hits = 0,0
    for t in range(horizon):
        z = Z[sample_indices[:,t]]
        shocks = np.sqrt(variances)*z
        r = (mean+shocks)/100.
        clip_hits += int((r < -.99).sum())
        r = np.maximum(r,-.99)
        wealth *= (1+r)
        variances = omega + alpha*shocks**2 + beta*variances
        cap_hits += int((variances > cap).sum())
        variances = np.maximum(np.minimum(variances,cap),dcc.MIN_VAR)
    return wealth-1,cap_hits/(n_paths*horizon*3),clip_hits


def dcc_fhs(states, dcc_fit, E, sample_indices):
    """DCC paths with the SAME joint-row draws used by standard FHS."""
    n_paths, horizon = sample_indices.shape
    var = np.broadcast_to(np.array([s["next_var"] for s in states]),(n_paths,3)).copy()
    Q = np.broadcast_to(dcc_fit["Q_start"],(n_paths,3,3)).copy()
    cap = np.array([s["vol_cap"] for s in states])
    mean = np.array([s["scenario_mu"] for s in states])
    omega = np.array([s["omega"] for s in states])
    alpha = np.array([s["alpha"] for s in states])
    beta = np.array([s["beta"] for s in states])
    a,b,S = dcc_fit["a"],dcc_fit["b"],dcc_fit["S"]
    wealth=np.ones((n_paths,3))
    cap_hits,clips=0,0
    for t in range(horizon):
        R = dcc.dcc_correlation(Q)
        L = np.linalg.cholesky(R)
        z = np.einsum("nij,nj->ni",L,E[sample_indices[:,t]])
        eps=np.sqrt(var)*z
        r=(mean+eps)/100.
        clips += int((r < -.99).sum())
        r=np.maximum(r,-.99)
        wealth *= 1+r
        var=omega+alpha*eps**2+beta*var
        cap_hits += int((var>cap).sum())
        var=np.maximum(np.minimum(var,cap),dcc.MIN_VAR)
        Q=(1-a-b)*S[None,:,:] + a*z[:,:,None]*z[:,None,:] + b*Q
    return wealth-1,cap_hits/(n_paths*horizon*3),clips


def dcc_hybrid_fhs(states, hybrid_dcc, E_hybrid, sample_indices, ewma_bond_state):
    """Full DCC scenario paths with bond EWMA, equity/BTC GARCH.

    Crucial: DCC QML and whitening MUST use the hybrid standardized
    residual series, not the old all-GARCH DCC fit.
    """
    n_paths, horizon = sample_indices.shape
    var = np.broadcast_to(
        np.array([s["next_var"] for s in states], dtype=float),
        (n_paths, 3),
    ).copy()
    q = np.broadcast_to(hybrid_dcc["Q_start"], (n_paths, 3, 3)).copy()
    cap = np.array([s["vol_cap"] for s in states], dtype=float)
    mean = np.array([s["scenario_mu"] for s in states], dtype=float)
    omega = np.array([s["omega"] for s in states], dtype=float)
    alpha = np.array([s["alpha"] for s in states], dtype=float)
    beta = np.array([s["beta"] for s in states], dtype=float)

    # Bond follows exactly the same EWMA recursion as the mixed FHS benchmark.
    var[:, 1] = ewma_bond_state["next_var"]
    cap[1] = ewma_bond_state["vol_cap"]
    mean[1] = ewma_bond_state["mu"]
    omega[1] = 0.0
    alpha[1] = 1.0 - EWMA_LAMBDA
    beta[1] = EWMA_LAMBDA

    a, b, S = hybrid_dcc["a"], hybrid_dcc["b"], hybrid_dcc["S"]
    wealth = np.ones((n_paths, 3), dtype=float)
    cap_hits = clip_hits = 0
    for t in range(horizon):
        R = dcc.dcc_correlation(q)
        L = np.linalg.cholesky(R)
        whitened = E_hybrid[sample_indices[:, t]]
        z = np.einsum("nij,nj->ni", L, whitened)
        shock = np.sqrt(var) * z
        daily_return = (mean + shock) / 100.0
        clip_hits += int((daily_return < -0.99).sum())
        daily_return = np.maximum(daily_return, -0.99)
        wealth *= 1.0 + daily_return
        var = omega + alpha * shock**2 + beta * var
        cap_hits += int((var > cap).sum())
        var = np.maximum(np.minimum(var, cap), dcc.MIN_VAR)
        q = (1.0-a-b)*S[None, :, :] + a*z[:, :, None]*z[:, None, :] + b*q
    return wealth - 1.0, cap_hits/(n_paths*horizon*3), clip_hits


def energy_score(ensemble, obs, scales, pairing_seed):
    """Energy score, unbiased independent-pair approximation to E||X-X'||.

    We use all simulated forecast paths for the first term; random
    permutations with no self-pairs for the second term.
    Fixed per-asset scale is estimated from TRAINING data only.
    """
    x = ensemble/scales
    y = obs/scales
    term1 = np.linalg.norm(x-y, axis=1).mean()
    rng = np.random.default_rng(pairing_seed)
    n=len(x)
    second=0.
    for _ in range(ENERGY_PAIR_REPEATS):
        # random independent distinct pairs: shifts guarantee no self-pairs
        shift = int(rng.integers(1,n))
        perm = rng.permutation(n)
        rolled = np.roll(perm, shift)
        second += np.linalg.norm(x[perm]-x[rolled],axis=1).mean()
    return float(term1 - 0.5*second/ENERGY_PAIR_REPEATS)


def risk_metrics(sim, realized):
    q = float(np.quantile(sim,TAIL_PROB))
    es = float(np.mean(sim[sim <= q]))
    return q,es,int(realized<q)


def kupiec_p(n,h,alpha=TAIL_PROB):
    p=h/n
    ll0 = h*np.log(alpha)+(n-h)*np.log1p(-alpha)
    ll1 = (h*np.log(p) if h else 0.) + ((n-h)*np.log1p(-p) if h<n else 0.)
    return float(chi2.sf(max(0.,2*(ll1-ll0)),1))


def paired_bootstrap(scores, model_a, model_b, rng):
    """Positive A-B means B has lower (better) energy score."""
    wide = scores.pivot(index="Rebalance", columns="Model", values="Standardized energy score")
    differences=(wide[model_a]-wide[model_b]).to_numpy(float)
    samples=rng.choice(len(differences), size=(BOOTSTRAP_REPETITIONS,len(differences)),replace=True)
    boot=differences[samples].mean(axis=1)
    return dict(Comparison=f"{model_b} vs {model_a}",
                **{"Mean ES advantage (A minus B)":float(differences.mean()),
                   "95% bootstrap CI lower":float(np.quantile(boot,.025)),
                   "95% bootstrap CI upper":float(np.quantile(boot,.975))})


def run(train, val, out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    train,val=validate_inputs(train,val)
    history_all=pd.concat([train,val])
    months=list(val.groupby(val.index.to_period("M"),sort=True))
    if not months: raise ValueError("Empty validation period")
    weights=np.array([PORTFOLIO_WEIGHTS[a] for a in train.columns])
    if (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError("Portfolio weights must be nonnegative and sum to 1")
    # TRAINING-ONLY fixed standardization so BTC cannot dominate the energy score.
    scales=np.maximum(train.std(ddof=1).to_numpy(float)*np.sqrt(21),1e-5)
    es_rows=[]; risk_rows=[]; joint_rows=[]; corr_rows=[]; caps=[]
    saved={model:[] for model in MODELS}
    date_keys=[]; observed=[]; horizons=[]
    dcc_fit_rows = []

    for k,(_period, block) in enumerate(months):
        rebalance=block.index[0]
        cutoff=rebalance-pd.DateOffset(years=ROLLING_YEARS)
        window=history_all.loc[(history_all.index>=cutoff) &
                               (history_all.index<rebalance),list(train.columns)]
        if len(window)<MIN_WINDOW_OBS:
            raise ValueError(f"Insufficient history at {rebalance.date()}: {len(window)} observations")
        assert window.index[-1]<rebalance
        actual=np.prod(1+block.to_numpy(float),axis=0)-1
        horizon=len(block)
        states,Z=dcc.fit_garch(window*100)
        params=dcc.dcc_parameters(Z)
        E=dcc.whitened_empirical_shocks(Z,params["S"])
        ewstate, ewz=ewma_bond(window*100)
        Zmixed=Z.copy(); Zmixed[:,1]=ewz
        # Fit a NEW DCC model to the hybrid-marginal standardized residuals.
        params_hybrid = dcc.dcc_parameters(Zmixed)
        E_hybrid = dcc.whitened_empirical_shocks(Zmixed, params_hybrid["S"])
        dcc_fit_rows.append({
            "Rebalance":rebalance.strftime("%Y-%m-%d"),
            "DCC-GARCH a":params["a"],"DCC-GARCH b":params["b"],
            "Hybrid DCC a":params_hybrid["a"],
            "Hybrid DCC b":params_hybrid["b"],
            "Hybrid persistence":params_hybrid["a"]+params_hybrid["b"],
            "Hybrid QML objective":params_hybrid["qll"],
            "Hybrid R0 equity-bond":dcc.dcc_correlation(params_hybrid["Q_start"])[0,1],
            "Hybrid R0 equity-BTC":dcc.dcc_correlation(params_hybrid["Q_start"])[0,2],
            "Hybrid R0 bond-BTC":dcc.dcc_correlation(params_hybrid["Q_start"])[1,2],
        })
        rng=np.random.default_rng(SEED+1_000_003*k)
        indices=rng.integers(0,len(Z),size=(N_SCENARIOS,horizon),dtype=np.int32)
        model_forecasts={
            MODELS[0]: standard_fhs(states,Z,indices),
            MODELS[1]: standard_fhs(states,Zmixed,indices,ewstate),
            MODELS[2]: dcc_fhs(states,params,E,indices),
            MODELS[3]: dcc_hybrid_fhs(states,params_hybrid,E_hybrid,indices,ewstate),
        }
        date_key=rebalance.strftime('%Y-%m-%d')
        date_keys.append(date_key); observed.append(actual);horizons.append(horizon)
        for m,(scenario,cap_rate,clip_count) in model_forecasts.items():
            if SAVE_MATCHED_SCENARIOS:
                saved[m].append(scenario.astype(np.float32))
            e=energy_score(scenario,actual,scales,SEED+1007*k)
            es_rows.append({"Rebalance":date_key,"Holding days":horizon,"Model":m,
                            "Standardized energy score":e})
            caps.append({"Rebalance":date_key,"Model":m,"Cap hit share":cap_rate,
                         "Daily -99% clipped returns":clip_count})
            ensemble=np.column_stack([scenario,scenario@weights])
            realized=np.append(actual,actual@weights)
            for j,name in enumerate(list(train.columns)+["Portfolio"]):
                q,tail,hits=risk_metrics(ensemble[:,j],realized[j])
                risk_rows.append({"Rebalance":date_key,"Model":m,"Asset":name,
                                  "Holding days":horizon,"Realized return":realized[j],
                                  "VaR return 5%":q,"CVaR return 5%":tail,
                                  "VaR exceedance":hits,
                                  "Realized loss minus predicted VaR loss":q-realized[j]})
            for a,b in ASSET_PAIRS:
                i,j=[list(train.columns).index(s) for s in (a,b)]
                p=float(np.mean((scenario[:,i]<0)&(scenario[:,j]<0)))
                y=int((actual[i]<0)&(actual[j]<0))
                joint_rows.append({"Rebalance":date_key,"Model":m,
                                   "Event":f"{a} & {b} negative",
                                   "Predicted probability":p,"Occurred":y,
                                   "Brier score":(p-y)**2})
                c=float(np.corrcoef(scenario[:,[i,j]],rowvar=False)[0,1])
                corr_rows.append({"Rebalance":date_key,"Model":m,
                                  "Pair":f"{a} - {b}","Simulated correlation":c})
            p=float(np.mean(np.all(scenario<0,axis=1)))
            y=int(np.all(actual<0))
            joint_rows.append({"Rebalance":date_key,"Model":m,
                               "Event":"All three negative",
                               "Predicted probability":p,"Occurred":y,
                               "Brier score":(p-y)**2})
        print(f"{k+1:02d}/{len(months)} {date_key}: {horizon} observed trading days; "
              f"hybrid DCC a={params_hybrid['a']:.4f}, b={params_hybrid['b']:.4f}",flush=True)

    es=pd.DataFrame(es_rows); risk=pd.DataFrame(risk_rows)
    joint=pd.DataFrame(joint_rows); corr=pd.DataFrame(corr_rows)
    pd.DataFrame(dcc_fit_rows).to_csv(out/'hybrid_dcc_fit_parameters.csv',index=False)
    es.to_csv(out/'monthly_energy_scores.csv',index=False)
    risk.to_csv(out/'monthly_var_cvar.csv',index=False)
    joint.to_csv(out/'monthly_joint_loss_probabilities.csv',index=False)
    corr.to_csv(out/'monthly_scenario_correlations.csv',index=False)
    pd.DataFrame(caps).to_csv(out/'monthly_scenario_cap_hits.csv',index=False)

    comparisons=[]
    for m in MODELS:
        r=risk[(risk.Model==m)&(risk.Asset=='Portfolio')]
        violations=r['VaR exceedance'].to_numpy(bool)
        actual=r['Realized return'].to_numpy(float)
        quantile=r['VaR return 5%'].to_numpy(float)
        predicted_es=r['CVaR return 5%'].to_numpy(float)
        # Proper quantile ('pinball') loss; lower is better.
        var_quantile_score=float(np.mean((actual-quantile)*(TAIL_PROB-(actual<quantile))))
        # Expected-shortfall identification moment: population expectation is
        # 0 under joint correct VaR/ES calibration; extremely noisy with 28 months.
        es_identification=float(np.mean(predicted_es-quantile +
                    (quantile-actual)*violations.astype(float)/TAIL_PROB))
        comparisons.append({"Model":m,"Months":len(r),"Monthly portfolio VaR hits":int(violations.sum()),
                            "Monthly portfolio VaR exceedance rate":float(violations.mean()),
                            "Kupiec p (LOW POWER)":kupiec_p(len(r),violations.sum()),
                            "Average predicted VaR loss (%)":-100*quantile.mean(),
                            "Average predicted CVaR loss (%)":-100*predicted_es.mean(),
                            "Mean VaR quantile score (lower better)":var_quantile_score,
                            "Mean ES identification moment (near zero better; noisy)":es_identification,
                            "Mean realized loss on VaR violations (%)":-100*actual[violations].mean() if violations.any() else np.nan,
                            "Mean predicted ES on VaR violations (%)":-100*predicted_es[violations].mean() if violations.any() else np.nan,
                            "Mean standardized energy score":es[es.Model==m]['Standardized energy score'].mean()})
    comp=pd.DataFrame(comparisons)
    comp.to_csv(out/'model_comparison.csv',index=False)

    joint_sum=joint.groupby(['Event','Model'],sort=False).agg(
        Months=('Occurred','size'),Mean_predicted_probability=('Predicted probability','mean'),
        Realized_event_frequency=('Occurred','mean'),Mean_Brier_score=('Brier score','mean')).reset_index()
    joint_sum.to_csv(out/'joint_loss_comparison.csv',index=False)

    actual_month=np.stack(observed)
    summary=[]
    for a,b in ASSET_PAIRS:
        ia,ib=[list(train.columns).index(s) for s in (a,b)]
        realized_corr=float(np.corrcoef(actual_month[:,[ia,ib]],rowvar=False)[0,1])
        for m in MODELS:
            subset=corr[(corr.Model==m)&(corr.Pair==f"{a} - {b}")]
            summary.append({"Pair":f"{a} - {b}","Model":m,
                            "Average forecast-horizon simulated correlation":subset['Simulated correlation'].mean(),
                            "Realized calendar-month correlation (28 obs; noisy)":realized_corr})
    pd.DataFrame(summary).to_csv(out/'correlation_comparison.csv',index=False)

    rng=np.random.default_rng(SEED+99999)
    bootstrap=pd.DataFrame([
        paired_bootstrap(es, MODELS[0], MODELS[i], rng) for i in range(1,len(MODELS))
    ] + [paired_bootstrap(es, MODELS[j], MODELS[3], rng) for j in (1,2)])
    bootstrap.to_csv(out/'paired_energy_bootstrap.csv',index=False)

    if SAVE_MATCHED_SCENARIOS:
        # Variable horizons don't fit a numeric 4-D cube, but COMPOUNDED
        # return arrays all have the same shape per rebalance.
        np.savez_compressed(out/'matched_monthly_scenarios.npz',
                            rebalance_dates=np.asarray(date_keys),
                            holding_days=np.asarray(horizons),
                            assets=np.asarray(list(train.columns)),
                            realized_returns=actual_month.astype(np.float32),
                            all_garch=np.stack(saved[MODELS[0]]),
                            mixed=np.stack(saved[MODELS[1]]),
                            dcc=np.stack(saved[MODELS[2]]),
                            dcc_hybrid=np.stack(saved[MODELS[3]]))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pivot=es.pivot(index='Rebalance',columns='Model',values='Standardized energy score')
        ax=pivot.plot(figsize=(12,5),marker='.',lw=1)
        ax.set(title='Calendar-month matched joint-distribution energy scores',ylabel='Energy score (lower better)',xlabel='Rebalance date')
        plt.xticks(rotation=45,ha='right');plt.tight_layout()
        plt.savefig(out/'monthly_energy_scores.png',dpi=150);plt.close()
    except ImportError:pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fits = pd.DataFrame(dcc_fit_rows)
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, col, label in zip(axs,
                ["Hybrid R0 equity-bond", "Hybrid R0 equity-BTC", "Hybrid R0 bond-BTC"],
                ["Equity - Bond", "Equity - BTC", "Bond - BTC"]):
            ax.plot(pd.to_datetime(fits["Rebalance"]), fits[col], marker=".")
            ax.axhline(0,color="gray",lw=.8)
            ax.set(ylabel=label,ylim=(-1,1))
        fig.suptitle("Hybrid DCC: one-day-ahead correlations at each monthly rebalance")
        plt.tight_layout(); plt.savefig(out/"hybrid_dcc_correlations.png",dpi=150);plt.close()
    except ImportError:pass
    print('\nMONTH-MATCHED MODEL COMPARISON\n'+comp.round(4).to_string(index=False))
    print('\nJOINT LOSS SUMMARY\n'+joint_sum.round(4).to_string(index=False))
    print('\nPAIRED ENERGY DIFFERENCES\n'+bootstrap.round(5).to_string(index=False))
    print(f'\nSaved matched-horizon comparison to {out}')
    return comp,joint_sum,es,risk


if __name__=='__main__':
    training=dcc.load_returns(DATA_DIR/'portfolio_train.csv')
    validation=dcc.load_returns(DATA_DIR/'portfolio_validation.csv')
    run(training,validation)
