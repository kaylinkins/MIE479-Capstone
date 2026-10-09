import pandas as pd

#This code splits our dataset from 2015-2026 into two periods for training and testing. The training period is from 2015-2019, and the testing period is from 2020-2026.
#When training models, use the train variables. When evaluating models, use the test variables.

# Load the previously prepared return dataset.
returns = pd.read_csv(
    "portfolio_daily_returns.csv",
    index_col=0,
    parse_dates=True
)

# Historical data available before the testing period.
train = returns.loc[:"2019-12-31"]

# Later observations reserved for evaluation.
test = returns.loc["2020-01-01":]

print("Training observations:", len(train))
print("Testing observations:", len(test))