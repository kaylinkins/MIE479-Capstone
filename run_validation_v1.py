"""Walk-forward validation of HMM and tuned Jump with paired DCC-GARCH risk."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
HMM_DIR = ROOT / "2-State HMM" / "HMM Forecasts"
JUMP_DIR = HMM_DIR.parent / "Jump Model Forecasts"
DCC_DIR = ROOT / "CVaR Regime Models" / "forecast_comparison_results_rf4"
sys.path[:0] = [
    str(ROOT), str(HMM_DIR), str(JUMP_DIR), str(ROOT / "MVO Regime Models"),
]

import hmm_regime_model as hmm
import jump_regime_model_tuned as jump
import regime_aware_mvo as aware_mvo
import regime_switching_mvo as switching_mvo
from portfolio_metrics import annualized_excess_sharpe
from regime_cvar import (UPPER, cvar, load_returns, max_sharpe_reference, optimize_forecast_cvar,
                         optimize_switching_cvar, regimes_from_bear_probabilities, sha)
from simple_regime_strategy import comparison_rotation_paths
from regime_comparison_figures import HEURISTIC, STRATEGY_LINES, format_dates, save_allocations, save_paths

ASSETS = cvar.ASSETS
FORECASTS = ("HMM", "Jump Model")
POLICIES = ("Regime Aware", "Regime Switching")
HORIZON = 3
N_SCENARIOS = 10_000
ALPHA = .95
CVAR_RISK_AVERSION = .5
COST_RATE = .001


def results_folder(policy, optimizer):
    return ROOT / f"{policy} {optimizer} Results v1"


def read_dcc(origin, information_date, sessions, manifest):
    name = f"forecasts/{origin.date()}.npz"
    path = DCC_DIR / name
    if sha(path) != manifest["output_sha256"][name]:
        raise ValueError(f"DCC forecast archive changed at {origin.date()}")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if pd.Timestamp(metadata["information_date"]) != information_date:
            raise ValueError(f"DCC information date disagrees at {origin.date()}")
        for dates, archived in zip(sessions, metadata["dcc_generator"]["period_dates"], strict=True):
            if not dates.equals(pd.DatetimeIndex(archived)):
                raise ValueError(f"DCC calendar disagrees at {origin.date()}")
        days = np.array([len(dates) for dates in sessions])
        np.testing.assert_array_equal(saved["holding_days"], days)
        cube = saved["DCC GARCH"].astype(float)
        if cube.shape != (N_SCENARIOS, HORIZON, len(ASSETS)):
            raise ValueError(f"Unexpected DCC scenario shape at {origin.date()}")
        return cube, days, saved["means_hmm"]


def solve_and_verify(optimizer, policy, name, means, probabilities, regimes, cube, days, prior, pool, reference):
    """Supply the same stage inputs and verify the resulting policy and floors."""
    switching = policy == POLICIES[1]
    active = (regimes == "bull") if switching else np.ones(HORIZON, dtype=bool)

    if optimizer == "CVaR":
        options = dict(alpha=ALPHA, risk_aversion=CVAR_RISK_AVERSION, upper=UPPER,
                       cost_rate=COST_RATE, mean_source=name, scenario_source="DCC GARCH",
                       cut_pool=pool)
        if switching:
            plan = optimize_switching_cvar(cube, prior, days, means, regimes, **options)
            rules = plan.metadata["stage_policies"]
            constraint_means = np.where(active[:, None], means, 0.)
            np.testing.assert_array_equal(plan.metadata["solver_return_constraint_means"],
                                          constraint_means if active.any() else means)
        else:
            plan = optimize_forecast_cvar(cube, prior, days, means, policy="matched_return", **options)
            rules = ["matched_return"] * HORIZON
        weights, first, floors = plan.weights, plan.first_weights, plan.metadata["gross_return_targets"]
        np.testing.assert_allclose(plan.expected_returns, np.sum(means * weights, axis=1),
                                   atol=1e-12, rtol=0)
        np.testing.assert_array_equal(rules, np.where(active, "matched_return", "min_cvar"))
    else:
        forecast = aware_mvo.Forecast(means, reference.covariances, probabilities, days, reference.targets)
        np.testing.assert_array_equal(forecast.mu, means)
        np.testing.assert_array_equal(forecast.p_bear, probabilities)
        if switching:
            first, plan = switching_mvo.solve_plan(prior, forecast, HORIZON, np.asarray(UPPER))
        else:
            first, plan = aware_mvo.solve_plan(prior, forecast, HORIZON, "Max Sharpe MVO", np.asarray(UPPER))
        weights, floors, rules = plan["plan"], plan["floors"], plan["stages"]
        np.testing.assert_array_equal(rules, np.where(active, "MS", "MV"))

    np.testing.assert_array_equal([floor is not None for floor in floors], active)
    np.testing.assert_allclose(np.asarray(floors, dtype=float)[active], reference.targets[active],
                               atol=1e-10, rtol=0)
    if np.any(np.sum(means[active] * weights[active], axis=1) < reference.targets[active] - 2e-7):
        raise AssertionError(f"An active return floor was missed by {policy} {optimizer} for {name}")
    return first, weights, floors, rules


def drawdown(returns):
    wealth = np.r_[1., (1 + returns).cumprod()]
    return (wealth / np.maximum.accumulate(wealth) - 1).min()


def write_tables(output, daily, monthly, train_end, penalty, policy, optimizer):
    rows = []
    for name in FORECASTS:
        daily_account = daily[daily.forecast.eq(name)]
        month_account = monthly[monthly.forecast.eq(name)]
        episodes = daily_account.common_regime.ne(daily_account.common_regime.shift()).cumsum()
        np.testing.assert_allclose(
            (1 + daily_account.net_return).groupby(daily_account.origin).prod() - 1,
            month_account.net_return, atol=1e-12, rtol=0,
        )
        np.testing.assert_allclose((1 + month_account.net_return).cumprod(), month_account.ending_wealth,
                                   atol=1e-12, rtol=0)
        for regime in ("Overall", "Bull", "Bear"):
            returns = daily_account.net_return
            months = month_account
            if regime != "Overall":
                select = daily_account.common_regime.eq(regime.lower())
                returns = returns[select]
                months = months[months.common_regime.eq(regime.lower())]
                max_dd = returns.groupby(episodes[select]).apply(drawdown).min()
            else:
                max_dd = drawdown(returns)
            rows.append({
                "Forecasts": name, "Scenarios": "DCC-GARCH", "Regime": regime,
                "Net CAGR (%)": 100 * np.expm1(np.log1p(returns).sum() * 252 / len(returns)),
                "Annualized volatility (%)": 100 * returns.std(ddof=1) * np.sqrt(252),
                "Annualized Sharpe": annualized_excess_sharpe(returns),
                "Daily 95% VaR (%)": 100 * np.quantile(-returns, .95),
                "Monthly 95% VaR (%)": 100 * np.quantile(-months.net_return, .95),
                "Max drawdown (%)": 100 * max_dd,
                "Annualized transaction costs (%)": 100 * months.fee_fraction.sum() * 252 / len(returns),
                "Avg Equity weight (%)": 100 * months.Equity.mean(),
                "Avg Bond weight (%)": 100 * months.Bond.mean(),
                "Avg BTC weight (%)": 100 * months.BTC.mean(),
                "Avg monthly turnover (%)": 100 * months.one_way_turnover.mean(),
                "BTC cap binding frequency (%)": 100 * np.isclose(months.BTC, UPPER[2], atol=1e-6, rtol=0).mean(),
            })
    metrics = pd.DataFrame(rows)
    bull = monthly[(monthly.forecast == "HMM") & (monthly.common_regime == "bull")].shape[0]
    count = monthly.forecast.eq("HMM").sum()
    rule = ("The max-Sharpe return floor applies in all stages." if policy == POLICIES[0]
            else "The max-Sharpe return floor applies in forecast bull stages. Forecast bear stages minimize risk plus trading costs.")
    risk = ("CVaR uses 95% confidence and a risk weight of 0.5." if optimizer == "CVaR"
            else "MVO uses a variance weight of 1.25.")
    context = (
        f"{policy} {optimizer}, {daily.date.min().date()} through {daily.date.max().date()}. "
        f"Three-month plans use 10,000 DCC-GARCH paths per rebalance, a 20% Bitcoin cap, and "
        f"10 bps per dollar traded. {risk} {rule} Both the max-Sharpe reference and reported "
        f"Sharpe use a 4% annual risk-free rate. Both forecasters average over future regime paths. "
        f"The Jump penalty of {penalty:g} was selected using data through {train_end.date()}.\n\n"
        f"Overall, bull, and bear performance uses common first-stage HMM month-end labels "
        f"({bull} bull and {count - bull} bear months). Regime-specific drawdown is the worst "
        f"drawdown within a continuous episode of that regime. Monthly VaR uses complete months.\n\n"
    )
    keys = list(metrics.columns[:3])
    for filename, title, data in (
        ("table_1_performance", "Table 1. Portfolio risk and return performance", keys + list(metrics.columns[3:10])),
        ("table_2_allocations", "Table 2. Portfolio allocation behaviour", keys + list(metrics.columns[10:])),
    ):
        frame = metrics[data]
        frame.to_csv(output / f"{filename}.csv", index=False)
        headers = "| " + " | ".join(data) + " |"
        divider = "| " + " | ".join(["---"] * len(keys) + ["---:"] * (len(data) - len(keys))) + " |"
        lines = [headers, divider]
        for row in frame.itertuples(index=False, name=None):
            cells = [str(value) if col in keys else
                     f"{value:.3f}" if "Sharpe" in col or "costs" in col else f"{value:.2f}"
                     for col, value in zip(data, row, strict=True)]
            lines.append("| " + " | ".join(cells) + " |")
        (output / f"{filename}.md").write_text(f"# {title}\n\n{context}" + "\n".join(lines) + "\n",
                                                encoding="utf-8")


def save_figures(policy, runs, heuristic, initial_date):
    output = results_folder(policy, "MVO")
    labels = (("Matched-return CVaR", "Max-Sharpe MVO") if policy == POLICIES[0]
              else ("Switching CVaR", "Switching MVO"))
    styles = dict(zip((*labels, HEURISTIC), STRATEGY_LINES.values(), strict=True))
    paths, allocations, probabilities = {}, {}, {}
    for name in FORECASTS:
        daily, alloc, signals = {}, {}, {}
        for optimizer, label in zip(("CVaR", "MVO"), labels, strict=True):
            returns, monthly = runs[policy, optimizer]
            daily[label] = returns[returns.forecast.eq(name)].set_index("date").net_return
            monthly = monthly[monthly.forecast.eq(name)].set_index("origin")
            alloc[label] = monthly[list(ASSETS)]
            signals[label] = monthly.p_bear_month_end, .5
        daily[HEURISTIC] = heuristic[name][0]
        frame = pd.DataFrame(daily)
        if frame.isna().any().any():
            raise ValueError(f"Portfolio and heuristic daily dates disagree for {name}")
        initial = pd.DataFrame(1., index=[initial_date], columns=frame.columns)
        paths[name] = pd.concat([initial, (1 + frame).cumprod()])
        allocations[name], probabilities[name] = alloc, signals
    drawdowns = {name: frame.div(frame.cummax()).sub(1) for name, frame in paths.items()}
    save_paths(paths, output / "figure_1_cumulative_wealth.png", ylabel="Portfolio wealth",
               formatter=StrMethodFormatter("${x:.2f}"), reference=1, styles=styles)
    save_allocations(allocations, probabilities, frame.index[-1], output / "figure_2_portfolio_allocations.png")
    save_paths(drawdowns, output / "figure_3_drawdowns.png", ylabel="Drawdown",
               formatter=PercentFormatter(1, decimals=0), reference=0, styles=styles)


def save_dcc_scenario_figures(first_month, output):
    """Plot stage-one dependence and the spread in tail risk for a fixed portfolio."""
    output.mkdir(exist_ok=True)
    dates = sorted(first_month)
    correlations = np.array([np.corrcoef(first_month[date], rowvar=False) for date in dates])
    fig = Figure(figsize=(9, 4.4), facecolor="white", layout="constrained")
    ax = fig.subplots()
    for i, j, label, color in (
        (0, 1, "Equity–Bonds", "#176d8a"),
        (0, 2, "Equity–Bitcoin", "#b95623"),
        (1, 2, "Bonds–Bitcoin", "#675094"),
    ):
        ax.plot(dates, correlations[:, i, j], color=color, linewidth=1.8, label=label)
    ax.axhline(0., color="#9ca8ae", linewidth=.8)
    ax.set(title="DCC-GARCH First-Month Scenario Correlations", ylabel="Pearson correlation",
           xlabel="Rebalance date")
    ax.grid(axis="y", color="#e1e7ea", linewidth=.8)
    ax.set_axisbelow(True)
    format_dates(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False)
    fig.savefig(output / "dcc_first_month_correlations.png", dpi=220)

    fixed_weights = np.array([.4, .4, .2])
    returns = {date: first_month[date] @ fixed_weights for date in dates}
    risk = {date: cvar.loss_var_cvar(values, alpha=ALPHA) for date, values in returns.items()}
    higher = max(dates, key=lambda date: risk[date][1])
    lower = min(dates, key=lambda date: risk[date][1])
    bins = np.histogram_bin_edges(np.r_[-returns[higher], -returns[lower]], bins=55)
    fig = Figure(figsize=(11, 4.8), facecolor="white", layout="constrained")
    fig.suptitle("DCC-GARCH First-Month Portfolio Loss Distributions\n"
                 "Fixed weights of 40% Equity, 40% Bonds and 20% Bitcoin", fontsize=14)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, date in zip(axes, (higher, lower), strict=True):
        var, expected_shortfall = risk[date]
        ax.axvspan(var, bins[-1], color="#962f38", alpha=.09, zorder=0)
        ax.hist(-returns[date], bins=bins, color="#447b91", edgecolor="white", linewidth=.3)
        ax.axvline(0., color="#87949a", linewidth=.8)
        ax.axvline(var, color="#a7601e", linewidth=1.7, label=f"{ALPHA:.0%} VaR ({var:.2%})")
        ax.axvline(expected_shortfall, color="#962f38", linewidth=1.7,
                   label=f"{ALPHA:.0%} CVaR ({expected_shortfall:.2%})")
        ax.set(title=date.strftime("%B %Y rebalance"), xlabel="Gross portfolio loss")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e7ea", linewidth=.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Scenario count")
    fig.savefig(output / "dcc_first_month_loss_distributions.png", dpi=220)


def main():
    data_dir = ROOT / "Data"
    train = load_returns(data_dir / "portfolio_train.csv")
    validation = load_returns(data_dir / "portfolio_validation.csv")
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Training must precede validation")
    history = pd.concat([train, validation])
    calendar = hmm.make_calendar(history, validation)
    hmm_means = pd.read_csv(HMM_DIR / "expected_returns.csv", parse_dates=["rebalance_date"])
    hmm_signals = pd.read_csv(HMM_DIR / "regime_probabilities.csv",
                              parse_dates=["rebalance_date", "information_date"]).set_index("rebalance_date")
    manifest = json.loads((DCC_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest["completed"] or manifest["protocol"]["n_scenarios"] != N_SCENARIOS:
        raise ValueError("Expected a completed 10,000-scenario validation risk run")

    penalty = jump.select_settings(train, calendar)
    groups = list(validation.groupby(validation.index.to_period("M")))
    keys = [(policy, optimizer) for policy in POLICIES for optimizer in ("CVaR", "MVO")]
    accounts = {(key, name): (np.zeros(len(ASSETS)), 1.) for key in keys for name in FORECASTS}
    records = {key: ([], [], []) for key in keys}
    signal_rows = []
    source_rows = []
    dcc_first_month = {}
    for seed, (month, block) in enumerate(groups):
        origin = block.index[0]
        prior_data = history.loc[history.index < origin]
        information_date = prior_data.index[-1]
        sessions = hmm.forecast_sessions(calendar, origin)
        cube, days, archived_hmm = read_dcc(origin, information_date, sessions, manifest)
        if not block.index.equals(sessions[0]):
            raise ValueError(f"Validation month is incomplete or misaligned at {origin.date()}")
        dcc_first_month[origin] = cube[:, 0, :].copy()
        hrows = hmm_means[hmm_means.rebalance_date.eq(origin)].sort_values("month_ahead")
        np.testing.assert_array_equal(hrows.month_ahead, np.arange(1, HORIZON + 1))
        np.testing.assert_array_equal(hrows.forecast_month, [str(month + k) for k in range(HORIZON)])
        hm = hrows[list(ASSETS)].to_numpy(float)
        np.testing.assert_array_equal(hm, archived_hmm)
        hs = hmm_signals.loc[origin]
        if hs.information_date != information_date:
            raise ValueError(f"HMM information cutoff disagrees at {origin.date()}")
        hp = hs[[f"p_bear_month{k}_end" for k in range(1, HORIZON + 1)]].to_numpy(float)

        model, state = jump.fit_jump(prior_data, seed=seed, penalty=penalty)
        _, jump_means, _ = hmm.forecast_moments_exact(model, state, days)
        jm = np.asarray(jump_means, dtype=float)
        jp = np.array([(state @ np.linalg.matrix_power(model.transmat_, int(d)))[1] for d in days.cumsum()])
        common_regime = regimes_from_bear_probabilities(hp)[0]
        pool = cvar.CVaRCutPool()
        for name, means, probabilities, current in (
            ("HMM", hm, hp, hs.p_bear_today), ("Jump Model", jm, jp, state[1]),
        ):
            regimes = regimes_from_bear_probabilities(probabilities)
            reference = max_sharpe_reference(cube, means, days, upper=UPPER)
            signal_rows.append(dict(source=name, rebalance_date=origin,
                                    information_date=information_date, p_bear_today=current))
            origin_rows = [
                dict(forecast=name, origin=origin, information_date=information_date,
                     stage=k + 1, forecast_month=str(month + k), holding_days=days[k],
                     scenario_count=N_SCENARIOS, p_bear_today=current, p_bear_month_end=probabilities[k],
                     forecast_regime=regimes[k], **dict(zip(ASSETS, means[k], strict=True)))
                for k in range(HORIZON)
            ]
            source_rows.extend(origin_rows)
            for key in keys:
                policy, optimizer = key
                daily, monthly, planned = records[key]
                prior, wealth = accounts[key, name]
                first, weights, floors, rules = solve_and_verify(
                    optimizer, policy, name, means, probabilities, regimes, cube, days, prior, pool, reference,
                )
                execution = cvar.execute_first_period(
                    SimpleNamespace(first_weights=first, pretrade_weights=prior, cost_rates=COST_RATE),
                    block.to_numpy(), initial_wealth=wealth,
                )
                accounts[key, name] = execution.ending_weights, execution.ending_wealth
                metadata = dict(forecast=name, origin=origin, common_regime=common_regime)
                daily.extend(dict(**metadata, date=date, net_return=ret)
                             for date, ret in zip(block.index, execution.net_daily_returns, strict=True))
                monthly.append(dict(**metadata, **dict(zip(ASSETS, first, strict=True)),
                                    p_bear_month_end=probabilities[0], forecast_regime=regimes[0],
                                    days=len(block), net_return=execution.net_period_return,
                                    ending_wealth=execution.ending_wealth, fee_fraction=execution.fee_fraction,
                                    one_way_turnover=.5 * np.abs(first - prior).sum() if prior.any() else 0.))
                for k, row in enumerate(origin_rows):
                    planned.append(dict(**row, stage_policy=rules[k], return_floor=floors[k],
                                        expected_portfolio_return=means[k] @ weights[k],
                                        **{f"weight_{asset}": value for asset, value in zip(ASSETS, weights[k], strict=True)}))
        print(f"{seed + 1:02d}/{len(groups)} {origin.date()} verified across eight portfolios", flush=True)

    heuristic = comparison_rotation_paths(train, validation, pd.DataFrame(signal_rows))
    inputs = pd.DataFrame(source_rows)
    runs = {}
    for key, frames in records.items():
        policy, optimizer = key
        daily, monthly, planned = map(pd.DataFrame, frames)
        pd.testing.assert_frame_equal(planned[inputs.columns], inputs)
        output = results_folder(*key)
        output.mkdir(exist_ok=True)
        write_tables(output, daily, monthly, train.index[-1], penalty, *key)
        daily.to_csv(output / "portfolio_daily.csv", index=False)
        monthly.to_csv(output / "portfolio_monthly.csv", index=False)
        planned.to_csv(output / "forecast_and_plan_audit.csv", index=False)
        runs[key] = daily, monthly
    for policy in POLICIES:
        save_figures(policy, runs, heuristic, train.index[-1])
    dcc_output = DCC_DIR / "figures"
    save_dcc_scenario_figures(dcc_first_month, dcc_output)
    print(f"Saved DCC-GARCH scenario figures in {dcc_output.relative_to(ROOT)}.", flush=True)
    print(f"Verified {len(inputs)} forecast-stage inputs and {len(inputs) * len(keys)} optimizer-stage decisions.", flush=True)


if __name__ == "__main__":
    main()

