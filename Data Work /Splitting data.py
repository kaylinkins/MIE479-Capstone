from pathlib import Path

import pandas as pd

# Approximate chronological 60% / 20% / 20% split; never shuffle.
# Round the target dates to the nearest calendar-month boundary.
# Training: estimate initial portfolio and model parameters.
# Validation: select models, constraints, and hyperparameters.
# Testing: evaluate finalized models; do not tune using test performance.
# Walk-forward refitting uses only history available at each decision date.

# Resolve files relative to this script, regardless of working directory.
output = Path(__file__).resolve().parent
returns = pd.read_csv(
    output / "portfolio_daily_returns.csv",
    index_col=0,
    parse_dates=True,
).sort_index()

if returns.index.hasnans or not returns.index.is_unique:
    raise ValueError("Return dates must be valid and unique.")
if returns.empty or returns.isna().any().any():
    raise ValueError("Returns must be nonempty without missing values.")

# Locate the original cumulative 60% and 80% boundaries by row count,
# then round each first held-out date to the nearest month start.
# A tie selects the earlier month start. No month crosses a split.
# Dates adjust automatically when the source dataset is refreshed.
total = len(returns)
if total < 5:
    raise ValueError("Too few observations for three nonempty datasets.")


def nearest_month_start(date):
    earlier = date.to_period("M").start_time
    later = earlier + pd.offsets.MonthBegin(1)
    return earlier if date - earlier <= later - date else later


validation_start = nearest_month_start(returns.index[total * 60 // 100])
test_start = nearest_month_start(returns.index[total * 80 // 100])
train = returns.loc[returns.index < validation_start].copy()
validation = returns.loc[
    (returns.index >= validation_start) & (returns.index < test_start)
].copy()
test = returns.loc[returns.index >= test_start].copy()

datasets = {"train": train, "validation": validation, "test": test}
if any(frame.empty for frame in datasets.values()):
    raise ValueError("Too few observations for three nonempty datasets.")

print(f"Total return observations: {total}")
print("Approximate 60% / 20% / 20% split (nearest calendar-month boundaries)")
print(f"Validation starts: {validation_start.date()}; testing starts: {test_start.date()}")
for name, frame in datasets.items():
    print(
        f"{name.capitalize()}: {len(frame):,} observations "
        f"({len(frame) / total:.2%}), "
        f"{frame.index.min().date()} to {frame.index.max().date()}"
    )
    frame.to_csv(output / f"portfolio_{name}.csv")

print(f"\nTraining, validation, and testing files saved to: {output}")
