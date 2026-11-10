# ============================================================
# HMM REGIME DETECTION (WP 2.1): FIGURES FOR THE REPORT
# ============================================================
#
# Run hmm_regime_model.py first. This script only reads its saved outputs
# (no refitting) and draws four graphs into hmm_outputs/:
#   fig1_regimes_training.png       regimes on the training data, all three assets
#   fig2_regime_probability.png     bear probability over validation
#   fig3_forecast_horizons.png      forecast bear probability for months 1, 2, 3
#   fig4_regime_characteristics.png how the two regimes differ
# ============================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
output = folder / "hmm_outputs"

train = pd.read_csv(folder / "portfolio_train.csv", index_col=0, parse_dates=True)
validation = pd.read_csv(folder / "portfolio_validation.csv", index_col=0, parse_dates=True)
p_bear = pd.read_csv(output / "daily_bear_probability.csv", index_col=0, parse_dates=True)["p_bear"]
probs = pd.read_csv(output / "regime_probabilities.csv", parse_dates=["rebalance_date"])
summary = pd.read_csv(output / "regime_summary.csv", index_col=0)

assets = list(train.columns)
p_train = p_bear.loc[train.index]
p_val = p_bear.loc[p_bear.index >= validation.index[0]]
colors = {"Equity": "#2a78d6", "Bond": "#1baf7a", "BTC": "#eb6834"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# Fig 1: regimes on the training data, all three assets
fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
for ax, a in zip(axes, assets):
    growth = (1 + train[a]).cumprod()
    ax.plot(growth, color=colors[a])
    ax.fill_between(train.index, growth.min(), growth.max(), where=p_train > 0.5,
                    color="grey", alpha=0.2, label="Bear regime")
    ax.set_yscale("log" if a == "BTC" else "linear")
    ax.set_title(f"{a}: growth of $1", loc="left")
axes[0].legend(loc="upper left")
axes[3].fill_between(train.index, p_train, color="grey")
axes[3].set_title("Filtered bear probability", loc="left")
fig.suptitle("Fig 1. Regimes on the training data", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(output / "fig1_regimes_training.png", dpi=150)

# Fig 2: bear probability over the validation period
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for a in assets:
    ax1.plot((1 + validation[a]).cumprod(), color=colors[a], label=a)
ax1.set_yscale("log")
ax1.legend(ncol=3)
ax1.set_title("Growth of $1 (log scale)", loc="left")
ax2.fill_between(p_val.index, p_val, color="grey")
ax2.set_title("Filtered bear probability (refit monthly)", loc="left")
fig.suptitle("Fig 2. Validation period", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(output / "fig2_regime_probability.png", dpi=150)

# Fig 3: forecast bear probability for months 1, 2, 3
fig, ax = plt.subplots(figsize=(9, 3.2))
for k, c in zip([1, 2, 3], ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.plot(probs.rebalance_date, probs[f"p_bear_month{k}"], marker="o", ms=3,
            color=c, label=f"Month {k} ahead")
ax.set_ylim(0, 1)
ax.set_ylabel("Forecast P(bear)")
ax.legend(ncol=3)
ax.set_title("Fig 3. Forecast bear probability at each rebalance", loc="left")
fig.tight_layout()
fig.savefig(output / "fig3_forecast_horizons.png", dpi=150)

# Fig 4: how the two regimes differ
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
x = np.arange(3)
panels = [("Mean return (ann., %)", [f"{a} return (ann.)" for a in assets], assets, 100),
          ("Volatility (ann., %)", [f"{a} volatility (ann.)" for a in assets], assets, 100),
          ("Correlation", ["Corr Equity-Bond", "Corr Equity-BTC", "Corr Bond-BTC"],
           ["Equity-Bond", "Equity-BTC", "Bond-BTC"], 1)]
for ax, (title, rows, labels, scale) in zip(axes, panels):
    ax.bar(x - 0.2, summary.loc[rows, "bull"] * scale, 0.4, color="#2a78d6", label="Bull")
    ax.bar(x + 0.2, summary.loc[rows, "bear"] * scale, 0.4, color="#eb6834", label="Bear")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_title(title, loc="left")
axes[1].legend()
fig.suptitle("Fig 4. How the regimes differ (training data)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(output / "fig4_regime_characteristics.png", dpi=150)

print(f"Saved 4 figures to {output}")
