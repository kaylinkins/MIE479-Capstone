# Rotation table: HMM vs Jump tuned, 3 assets (Equity/Bond/BTC), 4% rf, validation only.
# Reads saved CSVs only. Does not refit, does not modify any existing file.
# Save in Simple_Regime_Heuristic/ and run.
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" 
OUT.mkdir(parents=True, exist_ok=True)

RF_ANNUAL, TD = 0.04, 252
EXTRA_BASELINES = False      # True adds always-80/20 buy-and-hold and 100% Bond
SOURCES = {"HMM rotation (80% Equity / 20% BTC, else Bond)": "hmm",
           "Jump tuned rotation (80% Equity / 20% BTC, else Bond)": "jump_tuned"}


def metrics(r, cost_pct_total):
    a = r.to_numpy(float)
    n, years = len(a), len(a) / TD
    wealth = np.cumprod(1 + a)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    sd = a.std(ddof=1)
    return {
        "Net CAGR (%)": (wealth[-1] ** (1 / years) - 1) * 100,
        "Annual volatility (%)": sd * np.sqrt(TD) * 100,
        "Annual Sharpe": (a.mean() - RF_ANNUAL / TD) / sd * np.sqrt(TD),
        "Daily 95% VaR (%)": np.quantile(-a, .95) * 100,
        "Monthly 95% VaR (%)": np.quantile(-monthly.to_numpy(), .95) * 100,
        "Max drawdown (%)": (1 - wealth / peak).max() * 100,
        "Annual transaction costs (%)": cost_pct_total / years,
        "Total return (%)": (wealth[-1] - 1) * 100,
        "Daily observations": n,
    }


def find_col(df, text):
    hits = [c for c in df.columns if text.lower() in str(c).lower()]
    if not hits:
        raise KeyError(f"No column containing '{text}' in {list(df.columns)}")
    return hits[0]


def load(folder):
    d = HERE / "outputs" / folder
    daily = pd.read_csv(d / "step2_validation_daily_net_returns.csv", index_col=0, parse_dates=True)
    res = pd.read_csv(d / "step2_validation_results.csv")
    return daily.sort_index(), res


rows, checks = [], []
for label, folder in SOURCES.items():
    daily, res = load(folder)
    names = list(res["Strategy"]) if "Strategy" in res.columns else list(res.iloc[:, 0])
    cost_col = find_col(res, "Total costs")
    ret_col = find_col(res, "Total return")
    sharpe_col = find_col(res, "Sharpe")

    series = [(label, daily.iloc[:, 0])]                       # rotation = first data column
    series += [(c, daily[c]) for c in daily.columns[1:]
               if "Equal weight" in c or (EXTRA_BASELINES and ("Always" in c or "Bond" in c))]
    for lab, r in series:
        if folder == "jump_tuned" and not lab.startswith("Jump tuned"):
            continue                                            # baselines are the same in both files; show once
        pos = list(daily.columns).index(r.name)
        res_row = res.iloc[pos]                                 # same order as daily columns
        m = metrics(r, float(res_row[cost_col]))
        m["Strategy"] = lab
        rows.append(m)
        checks.append((lab, m["Total return (%)"], float(res_row[ret_col]),
                       round(m["Annual Sharpe"], 2), float(res_row[sharpe_col])))

table = pd.DataFrame(rows).set_index("Strategy")
cols = ["Net CAGR (%)", "Annual volatility (%)", "Annual Sharpe", "Daily 95% VaR (%)",
        "Monthly 95% VaR (%)", "Max drawdown (%)", "Annual transaction costs (%)"]
table.to_csv(OUT / "rotation_hmm_vs_jump_tuned_table.csv")

print("| Strategy | " + " | ".join(cols) + " |")
print("|---|" + "---:|" * len(cols))
for name, row in table.iterrows():
    print(f"| {name} | " + " | ".join(f"{row[c]:.3f}" if c == "Annual Sharpe" else f"{row[c]:.2f}"
                                     for c in cols) + " |")

print("\nDaily observations:", sorted(table["Daily observations"].unique()))
print("\nCHECK vs saved results file (computed vs saved total return %, computed vs saved Sharpe):")
for lab, tr, tr_saved, sh, sh_saved in checks:
    print(f"  {lab[:45]:45s} {tr:8.2f} vs {tr_saved:8.2f} | {sh:6.2f} vs {sh_saved:6.2f}")
print(f"\nSaved to {OUT}")