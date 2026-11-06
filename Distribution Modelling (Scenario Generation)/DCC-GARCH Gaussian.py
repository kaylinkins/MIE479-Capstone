"""DCC(1,1)-GARCH(1,1), rolling 5-year estimation, for capstone validation.

Inputs (project Data/datasets directory):
    portfolio_train.csv, portfolio_validation.csv
Each CSV: ascending dates in column 1, Equity/Bond/BTC columns (any order),
          aligned SIMPLE DAILY RETURNS AS DECIMALS (0.01 = +1%).

Method:
  - GARCH(1,1) marginals fitted with skewed-t likelihood, as in the prior FHS.
  - Gaussian DCC *quasi-likelihood* estimates correlation dynamics; this is
    NOT a multivariate Student-t DCC implementation.
  - Joint 21-day scenarios resample *whitened empirical shock rows*, then
    recolor them with simulated DCC correlations. No multivariate Student-t
    innovations are introduced in this version.
  - Validation is walk-forward: fit using prior 5 calendar years, monthly;
    update GARCH volatilities and DCC correlations each day after its observed
    return becomes known. No future observations enter forecast inputs.
  - The historical chart joins a DESCRIPTIVE in-sample training history
    (2015-Jan 2022) and out-of-sample validation forecasts (Feb 2022-May 2024).
    It DOES NOT use the full sample to fit any validation forecast.

Outputs saved into project outputs/volatility/dcc_results/
Run: py capstone_dcc_garch.py
Requires: numpy, pandas, scipy, matplotlib, arch
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
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from arch import arch_model

# ----------------------------- SETTINGS -----------------------------
ROOT = MODELLING_ROOT
DATA_DIR = ROOT.parent / "Data" / "datasets"
OUT = PROJECT_ROOT / "outputs/volatility/dcc_results"
ROLLING_YEARS = 5
MIN_WINDOW_OBS = 750
HORIZON = 21
N_SCENARIOS = 10_000
N_MONTH_BACKTEST = 4_000
SEED = 42
VAR_ALPHA = 0.05
CHECK_WEIGHTS = None                    # None = 1/3 each (in asset-column order)
MARGINAL_DIST = "skewt"                # keep same marginal fit as old GARCH-FHS
VOL_CAP_MULTIPLE = 1.5                 # None disables cap; scenario paths only
SAVE_DAILY_PATHS = False               # True: large .npy file (~70 MB)
MIN_VAR = 1e-10                        # returns in percentage points
DCC_MAX_PERSISTENCE = 0.998            # impose stationarity a+b <= 0.998
DCC_MAX_ITER = 250
# --------------------------------------------------------------------


def load_returns(path):
    x = pd.read_csv(path, index_col=0, parse_dates=True)
    if x.empty or x.index.has_duplicates or not x.index.is_monotonic_increasing:
        raise ValueError(f"Dates in {path.name} must be unique and increasing")
    if x.isna().any().any():
        raise ValueError(f"Missing observations in {path.name}; align first")
    x = x.astype(float)
    if not np.isfinite(x.to_numpy()).all() or (x <= -1).any().any():
        raise ValueError(f"Invalid daily simple returns in {path.name}")
    return x


def fit_garch(window_pct):
    """Fit one GARCH model per asset; return states and n x 3 innovations."""
    states = []
    standardized = []
    for asset in window_pct.columns:
        series = window_pct[asset]
        model = arch_model(series, mean="Constant", vol="GARCH", p=1, o=0,
                           q=1, dist=MARGINAL_DIST)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp="off", show_warning=False)
        if fit.convergence_flag != 0:
            warnings.warn(f"GARCH fit for {asset} at {series.index[-1].date()} "
                          f"did not converge (code={fit.convergence_flag})")
        par = fit.params
        z_raw = fit.std_resid.to_numpy(dtype=float)
        zm = float(np.mean(z_raw[np.isfinite(z_raw)]))
        zs = float(np.std(z_raw[np.isfinite(z_raw)], ddof=1))
        if zs < 1e-9 or not np.isfinite(zs):
            raise ValueError(f"Bad GARCH standardized residuals: {asset}")
        z = (z_raw-zm)/zs
        standardized.append(z)
        last_variance = float(fit.conditional_volatility.iloc[-1]**2)
        last_resid = float(fit.resid.iloc[-1])
        s = {
            "asset":asset, "mu":float(par["mu"]),
            "scenario_mu":float(series.mean()),  # same drift convention as old FHS
            "omega":float(par["omega"]),
            "alpha":float(par["alpha[1]"]), "beta":float(par["beta[1]"]),
            "zmean":zm, "zstd":zs,
            "next_var":max(MIN_VAR,float(par["omega"] + par["alpha[1]"]*last_resid**2
                                         + par["beta[1]"]*last_variance)),
            "vol_cap":(np.inf if VOL_CAP_MULTIPLE is None else
                       float((VOL_CAP_MULTIPLE*fit.conditional_volatility.max())**2)),
            "persistence":float(par["alpha[1]"]+par["beta[1]"]),
            "convergence":int(fit.convergence_flag)
        }
        states.append(s)
    Z = np.column_stack(standardized)
    if not np.isfinite(Z).all():
        raise ValueError("Nonfinite standardized residuals, inspect GARCH fits")
    return states, Z


def dcc_correlation(Q):
    d = np.sqrt(np.maximum(np.diagonal(Q, axis1=-2, axis2=-1), 1e-15))
    return Q/(d[..., :, None]*d[..., None, :])


def dcc_parameters(Z):
    """Gaussian DCC(1,1) QML fit to centered/scaled GARCH residuals.

    DCC equations: Q_t=(1-a-b)S + a z_{t-1}z_{t-1}' + b Q_{t-1};
                   R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2).
    For 3 assets use analytic inverse of 3x3 correlation matrix to speed QML.
    """
    S = np.cov(Z, rowvar=False, ddof=1)
    if S.shape != (3, 3):
        raise ValueError("This version explicitly implements three assets")
    if np.linalg.eigvalsh(S).min() < 1e-9:
        raise ValueError("Sample standardized-residual covariance not SPD")
    Z = np.asarray(Z, dtype=float)
    zz = Z[:-1]
    observations = Z[1:]
    S0 = np.array(S, dtype=float)
    def loss(ab):
        a, b = map(float, ab)
        if min(a,b) < 0 or a+b > DCC_MAX_PERSISTENCE:
            return 1e12
        Q = S0.copy()
        total = 0.0
        q_constant = (1-a-b)*S0
        for i, z in enumerate(observations):
            Q = q_constant + a*np.outer(zz[i], zz[i]) + b*Q
            R = dcc_correlation(Q)
            x,y,h = R[0,1],R[0,2],R[1,2]
            det = 1 + 2*x*y*h - x*x-y*y-h*h
            if det <= 1e-12 or not np.isfinite(det):
                return 1e12
            u,v,w = z
            numerator = ((1-h*h)*u*u + (1-y*y)*v*v + (1-x*x)*w*w
                         +2*(y*h-x)*u*v + 2*(x*h-y)*u*w + 2*(x*y-h)*v*w)
            total += np.log(det) + numerator/det
        return float(total / len(observations))

    cons = [{"type":"ineq", "fun":lambda ab: DCC_MAX_PERSISTENCE-ab[0]-ab[1]}]
    bounds = ((0,0.35), (0,DCC_MAX_PERSISTENCE))
    solutions = []
    for x0 in ((0.025,0.94), (0.08,0.85), (0.01,0.75)):
        result = minimize(loss, x0=np.array(x0), method="SLSQP", bounds=bounds,
                          constraints=cons,
                          options={"maxiter":DCC_MAX_ITER, "ftol":1e-8})
        if (result.success and np.isfinite(result.fun) and
                result.x.min() >= -1e-9 and result.x.sum() <= DCC_MAX_PERSISTENCE+1e-8):
            solutions.append(result)
            break
    if not solutions:
        raise RuntimeError("DCC optimization failed; inspect residual data and numerical settings")
    best = min(solutions,key=lambda r:r.fun)
    a,b = [float(x) for x in best.x]
    if a+b > 0.995:
        warnings.warn(f"DCC persistence near constrained boundary: a+b={a+b:.6f}")
    # R[k] is the forecast correlation for observation k, based on prior returns.
    Q = S0.copy()
    R_filtered = np.empty((len(Z),3,3))
    for k in range(len(Z)):
        R_filtered[k] = dcc_correlation(Q)
        Q = ((1-a-b)*S0 + a*np.outer(Z[k],Z[k]) + b*Q)
    # Q after the last observed shock is the FIRST out-of-sample Q forecast.
    return {"a":a,"b":b,"S":S0,"Q_start":Q,"R_training":R_filtered,
            "qll":float(best.fun), "message":str(best.message)}


def whitened_empirical_shocks(Z, S):
    """Preserve observed multivariate tail draws while removing linear covariance.

    Sample rows of e = Z @ inv(chol(S)).T, Cov(e)=I; during simulation
    set z_t=chol(R_t)@e_t, so Cov(z_t | history)=R_t.
    """
    L = np.linalg.cholesky(S)
    E = np.linalg.solve(L, Z.T).T
    return E


def simulate(states, dcc, E, npaths, days, rng, save_paths=False):
    """Vectorized DCC-FHS scenario generation with path-specific Q and GARCH."""
    n = len(states)
    variance = np.broadcast_to([s["next_var"] for s in states], (npaths,n)).copy()
    q = np.broadcast_to(dcc["Q_start"], (npaths,n,n)).copy()
    means = np.array([s["scenario_mu"] for s in states])
    w = np.array([s["omega"] for s in states])
    a_g = np.array([s["alpha"] for s in states])
    b_g = np.array([s["beta"] for s in states])
    cap = np.array([s["vol_cap"] for s in states])
    a,b,S = dcc["a"],dcc["b"],dcc["S"]
    paths = np.empty((npaths,days,n),dtype=np.float32) if save_paths else None
    wealth = np.ones((npaths,n),dtype=np.float64)
    cap_hits=0
    clipped=0
    for t in range(days):
        R = dcc_correlation(q)
        L = np.linalg.cholesky(R)
        e = E[rng.integers(len(E),size=npaths)]  # empirical JOINT innovations
        z = np.einsum("nij,nj->ni", L, e)
        eps = np.sqrt(variance)*z
        r = (means + eps)/100
        clipped += int(np.sum(r < -0.99))
        r = np.maximum(r, -0.99)    # simple asset losses cannot exceed 100%
        wealth *= 1 + r
        if save_paths:
            paths[:,t,:] = r
        variance = w+a_g*eps*eps+b_g*variance
        cap_hits += int(np.sum(variance > cap))
        variance = np.maximum(np.minimum(variance,cap),MIN_VAR)
        q = (1-a-b)*S[None,:,:]+a*z[:,:,None]*z[:,None,:]+b*q
    return (wealth-1),paths,cap_hits/(npaths*days*n),clipped


def marginal_next(states, realized_pct, variances):
    """After observing today's return, update tomorrow's GARCH variance."""
    eps = realized_pct - np.array([s["mu"] for s in states])
    omega = np.array([s["omega"] for s in states])
    alpha = np.array([s["alpha"] for s in states])
    beta = np.array([s["beta"] for s in states])
    return omega+alpha*eps**2+beta*variances, eps


def risk_rows(scenarios, actual, weights, label, rebalance, holding_days, observation_date=None):
    simulated = np.column_stack([scenarios,scenarios@weights])
    realized = np.append(actual,actual@weights)
    rows=[]
    for j,asset in enumerate(label):
        tail = np.quantile(simulated[:,j],VAR_ALPHA)
        ES = simulated[simulated[:,j]<=tail,j].mean()
        rows.append({"Rebalance":rebalance,"Observation date":(observation_date if observation_date is not None else rebalance),
                     "Asset":asset,"Holding days":holding_days,
                     "Realized return":float(realized[j]),
                     "VaR 5% return threshold":float(tail),
                     "Predicted ES return":float(ES),
                     "VaR exceedance":int(realized[j]<tail)})
    return rows


def kupiec_p(N, hits):
    if N==0:return np.nan
    h=int(hits); q=VAR_ALPHA; p=h/N
    null=h*np.log(q)+(N-h)*np.log1p(-q)
    best=(h*np.log(p) if h else 0) + ((N-h)*np.log1p(-p) if h<N else 0)
    return float(chi2.sf(max(0,2*(best-null)),1))


def indep_p(hits):
    h=np.asarray(hits,dtype=int)
    if len(h)<3:return np.nan
    n00=np.sum((h[:-1]==0)&(h[1:]==0));n01=np.sum((h[:-1]==0)&(h[1:]==1))
    n10=np.sum((h[:-1]==1)&(h[1:]==0));n11=np.sum((h[:-1]==1)&(h[1:]==1))
    def bin_ll(n0,n1,p):
        return (n1*np.log(p) if n1 and p>0 else 0) + (n0*np.log1p(-p) if n0 and p<1 else 0)
    pooled=(n01+n11)/(n00+n01+n10+n11)
    p01=n01/(n00+n01) if n00+n01 else 0
    p11=n11/(n10+n11) if n10+n11 else 0
    lr = 2*(bin_ll(n00,n01,p01)+bin_ll(n10,n11,p11)-bin_ll(n00+n10,n01+n11,pooled))
    return float(chi2.sf(max(0,lr),1))


def correlation_rows(dates, corr_array, regime):
    pairs=[(0,1),(0,2),(1,2)]
    return [{"Date":day,"Series":regime,"Asset pair":f"{ASSET_NAMES[i]} - {ASSET_NAMES[j]}",
             "Conditional correlation":float(R[i,j])}
            for day,R in zip(dates,corr_array) for i,j in pairs]


def main():
    global ASSET_NAMES
    OUT.mkdir(parents=True,exist_ok=True)
    train=load_returns(DATA_DIR/"portfolio_train.csv")
    val=load_returns(DATA_DIR/"portfolio_validation.csv")
    ASSET_NAMES=list(train.columns)
    if len(ASSET_NAMES)!=3:
        raise ValueError("This implementation expects exactly Equity, Bond, BTC columns")
    if set(ASSET_NAMES)!={"Equity","Bond","BTC"}:
        raise ValueError("Column names must be Equity, Bond, BTC (any order)")
    val=val[ASSET_NAMES]
    if train.index[-1]>=val.index[0]:
        raise ValueError("Train/validation periods overlap")
    all_data=pd.concat([train,val])
    weights=(np.full(3,1/3) if CHECK_WEIGHTS is None else np.asarray(CHECK_WEIGHTS,float))
    if weights.shape!=(3,) or not np.isclose(weights.sum(),1) or not np.isfinite(weights).all():
        raise ValueError("CHECK_WEIGHTS must contain three finite weights totaling 1")
    labels=ASSET_NAMES+["Portfolio"]

    # Training section of the historical CORRELATION GRAPH. Training-fit,
    # in-sample descriptive filter only, NEVER used to forecast validation.
    print("Building 2015–training-end descriptive DCC correlation history...")
    train_states,Z_train=fit_garch(train*100)
    dcc_train=dcc_parameters(Z_train)
    corr_rows=correlation_rows(train.index,dcc_train["R_training"],"Training in-sample")

    periods=[(p,g.index) for p,g in val.groupby(val.index.to_period("M"),sort=True)]
    if not periods:raise ValueError("No validation observations")
    months=[]; daily=[]; fit_info=[]; scenario_dep=[]; capinfo=[]
    monthly_scenarios=[]; rebalance_dates=[]; paths_out=[]
    for i,(period,days) in enumerate(periods):
        rebalance=days[0]
        cutoff=rebalance-pd.DateOffset(years=ROLLING_YEARS)
        history=all_data.loc[(all_data.index>=cutoff)&(all_data.index<rebalance),ASSET_NAMES]
        if len(history)<MIN_WINDOW_OBS:
            raise ValueError(f"Only {len(history)} observations before {rebalance.date()}")
        assert history.index.max()<rebalance
        states,Z=fit_garch(history*100)
        dcc=dcc_parameters(Z)
        E=whitened_empirical_shocks(Z,dcc["S"])
        for s in states:
            fit_info.append({"Rebalance":rebalance,"Asset":s["asset"],
                             "Estimation start":history.index[0],"Estimation end":history.index[-1],
                             "Training days":len(history),"GARCH alpha":s["alpha"],
                             "GARCH beta":s["beta"],"GARCH persistence":s["persistence"],
                             "GARCH convergence":s["convergence"],
                             "DCC a":dcc["a"],"DCC b":dcc["b"],
                             "DCC persistence":dcc["a"]+dcc["b"],
                             "DCC QML objective":dcc["qll"]})
        print(f"{i+1:02d}/{len(periods)} {period}: rolling {len(history)} days, "
              f"DCC a={dcc['a']:.4f} b={dcc['b']:.4f}",flush=True)

        # Save 10k FIXED 21-trading-day forecast scenarios for CVaR/MPC.
        rng=np.random.default_rng(SEED+100_003*i)
        scenarios, paths, caprate, clipped=simulate(states,dcc,E,N_SCENARIOS,HORIZON,rng,
                                                    save_paths=SAVE_DAILY_PATHS)
        rebalance_dates.append(str(rebalance.date()))
        monthly_scenarios.append(scenarios.astype(np.float32))
        if SAVE_DAILY_PATHS:
            paths_out.append(paths)
        capinfo.append({"Rebalance":rebalance,"Scenario cap hit rate":caprate,
                        "Daily return clips at -99%":clipped})
        simulated_cor=np.corrcoef(scenarios,rowvar=False)
        for a in range(3):
            for b in range(a+1,3):
                scenario_dep.append({"Rebalance":rebalance,"Pair":f"{ASSET_NAMES[a]} - {ASSET_NAMES[b]}",
                                     "Simulated 21d correlation":simulated_cor[a,b],
                                     "P(both negative)":float(np.mean((scenarios[:,a]<0)&(scenarios[:,b]<0)))})
        scenario_dep.append({"Rebalance":rebalance,"Pair":"All three",
                             "Simulated 21d correlation":np.nan,
                             "P(both negative)":float(np.mean(np.all(scenarios<0,axis=1)))})

        # Monthly VaR/ES uses ACTUAL calendar-month trading days, not always 21.
        if len(days)==HORIZON:
            month_draws=scenarios
        else:
            month_draws,_,_,_=simulate(states,dcc,E,N_MONTH_BACKTEST,len(days),
                         np.random.default_rng(SEED+400_003*i),save_paths=False)
        actual=np.prod(1+val.loc[days,ASSET_NAMES].to_numpy(float),axis=0)-1
        months.extend(risk_rows(month_draws,actual,weights,labels,rebalance,len(days)))

        # Daily 1-step conditional forecasts, before seeing today's returns.
        # Refit params monthly; filter Q and GARCH volatility DAILY after return.
        var=np.array([s["next_var"] for s in states])
        Q=dcc["Q_start"].copy()
        mu=np.array([s["scenario_mu"] for s in states])
        zmean=np.array([s["zmean"] for s in states])
        zstd=np.array([s["zstd"] for s in states])
        for today in days:
            R=dcc_correlation(Q)
            corr_rows.extend(correlation_rows([today],R[None,:,:],"Validation OOS forecast"))
            z_draws=E@np.linalg.cholesky(R).T
            simdaily=np.maximum((mu + np.sqrt(var)*z_draws)/100,-0.99)
            observed=val.loc[today,ASSET_NAMES].to_numpy(float)
            daily.extend(risk_rows(simdaily,observed,weights,labels,rebalance,1,
                                   observation_date=today))
            newvar,eps=marginal_next(states,observed*100,var)
            realized_z=(eps/np.sqrt(var)-zmean)/zstd
            Q=(1-dcc["a"]-dcc["b"])*dcc["S"]+dcc["a"]*np.outer(realized_z,realized_z)+dcc["b"]*Q
            var=np.maximum(newvar,MIN_VAR)  # NEVER cap filtered realized volatility

    rolling=np.stack(monthly_scenarios)
    np.savez_compressed(OUT/"dcc_rolling_scenarios_21day.npz",
                        rebalance_dates=np.asarray(rebalance_dates),
                        assets=np.asarray(ASSET_NAMES),returns=rolling)
    if SAVE_DAILY_PATHS:
        np.save(OUT/"dcc_rolling_paths_21day.npy",np.stack(paths_out))
    pd.DataFrame(months).to_csv(OUT/"dcc_rolling_monthly_var.csv",index=False)
    pd.DataFrame(daily).to_csv(OUT/"dcc_rolling_daily_var.csv",index=False)
    pd.DataFrame(fit_info).to_csv(OUT/"dcc_fit_parameters.csv",index=False)
    pd.DataFrame(corr_rows).to_csv(OUT/"dcc_dynamic_correlations.csv",index=False)
    pd.DataFrame(scenario_dep).to_csv(OUT/"dcc_scenario_dependence.csv",index=False)
    pd.DataFrame(capinfo).to_csv(OUT/"dcc_cap_hits.csv",index=False)

    # Charts: training in-sample history + true walk-forward OOS validation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    correlations=pd.DataFrame(corr_rows)
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for axis,pair in zip(axes,[f"{ASSET_NAMES[0]} - {ASSET_NAMES[1]}",
                                f"{ASSET_NAMES[0]} - {ASSET_NAMES[2]}",
                                f"{ASSET_NAMES[1]} - {ASSET_NAMES[2]}"]):
        sub=correlations[correlations["Asset pair"]==pair]
        for regime,color in (("Training in-sample","tab:blue"),
                             ("Validation OOS forecast","tab:orange")):
            q=sub[sub.Series==regime]
            axis.plot(pd.to_datetime(q.Date),q["Conditional correlation"],
                      color=color,lw=1.0,label=regime)
        axis.axvline(val.index.min(),c="red",ls="--",lw=1,label="Validation begins")
        axis.axhline(0,color="gray",lw=0.6)
        axis.set_ylim(-1,1)
        axis.set_ylabel("Correlation")
        axis.set_title(pair)
        axis.set_xlim(pd.Timestamp("2015-01-01"),pd.Timestamp("2024-12-31"))
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.tick_params(axis="x",labelbottom=True)
        axis.set_xlabel("Year")
        axis.legend(loc="upper left",fontsize=8)
    fig.suptitle("DCC(1,1) dynamic daily conditional correlations | 2015–2024\n"
                 "Blue = training in-sample; orange = monthly-refitted validation forecasts",fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"dcc_correlations_2015_2024.png",dpi=170)
    plt.close(fig)

    # Descriptive monthly-realization vs simulated conditional correlation summary.
    dep=pd.DataFrame(scenario_dep)
    realized_months=pd.DataFrame({"Date":[ix[0] for _,ix in periods],
          **{a:[np.prod(1+val.loc[ix,a])-1 for _,ix in periods] for a in ASSET_NAMES}})
    dependence_summary=[]
    for aa,bb in [(0,1),(0,2),(1,2)]:
        asset1,asset2=ASSET_NAMES[aa],ASSET_NAMES[bb]
        pair=f"{asset1} - {asset2}"
        dd=dep[dep.Pair==pair]
        dependence_summary.append({"Pair":pair,
             "Mean simulated 21d correlation":dd["Simulated 21d correlation"].mean(),
             "Realized validation calendar-month correlation":realized_months[[asset1,asset2]].corr().iloc[0,1],
             "Mean simulated both-negative probability":dd["P(both negative)"].mean(),
             "Realized fraction both negative":float(np.mean((realized_months[asset1]<0)&(realized_months[asset2]<0)))})
    dd=dep[dep.Pair=="All three"]
    dependence_summary.append({"Pair":"All three",
              "Mean simulated 21d correlation":np.nan,
              "Realized validation calendar-month correlation":np.nan,
              "Mean simulated both-negative probability":dd["P(both negative)"].mean(),
              "Realized fraction both negative":float(np.mean(np.all(realized_months[ASSET_NAMES].to_numpy()<0,axis=1)))})
    pd.DataFrame(dependence_summary).to_csv(OUT/"dcc_dependence_summary.csv",index=False)

    daily_df=pd.DataFrame(daily)
    mon_df=pd.DataFrame(months)
    comparison=[]
    for asset in labels:
        one=daily_df[daily_df.Asset==asset].sort_values("Observation date")
        month=mon_df[mon_df.Asset==asset]
        hit=int(one["VaR exceedance"].sum());mhit=int(month["VaR exceedance"].sum())
        comparison.append({"Asset":asset,"Daily observations":len(one),
             "Daily VaR exceedances":hit,"Daily exceedance rate":hit/len(one),
             "Daily Kupiec p":kupiec_p(len(one),hit),
             "Daily violation independence p":indep_p(one["VaR exceedance"].to_numpy()),
             "Monthly observations":len(month),"Monthly VaR exceedances":mhit,
             "Monthly exceedance rate":mhit/len(month),
             "Monthly Kupiec p (low power)":kupiec_p(len(month),mhit),
             "Average predicted monthly CVaR loss (%)":-100*month["Predicted ES return"].mean()})
    pd.DataFrame(comparison).to_csv(OUT/"dcc_model_comparison.csv",index=False)
    print("\nDAILY AND MONTHLY RISK FORECAST SUMMARY:")
    print(pd.DataFrame(comparison).round(4).to_string(index=False))
    print("\nJOINT DEPENDENCE (monthly forecasts vs only 28 realized months):")
    print(pd.DataFrame(dependence_summary).round(3).to_string(index=False))
    print("\nIMPORTANT: Historical blue correlations are descriptive in-sample estimates;")
    print("orange correlations are genuinely out-of-sample conditional forecasts.")
    print("Historical simulated scenarios use 21 trading days; realized validation")
    print("returns use calendar months with varying trading-day counts.")
    print("Saved results to:",OUT)


if __name__=="__main__":
    main()
