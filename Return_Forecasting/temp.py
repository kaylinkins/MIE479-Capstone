from pathlib import Path
import pandas as pd

ROOT = Path("/Users/elizzy/Downloads/MIE479-Capstone")
RF = ROOT / "Return_Forecasting"

cov = pd.read_csv(RF / "outputs/misc/history_coverage.csv")
print(cov.to_string(index=False))

print("\nLines mentioning regressions in Return_Forecasting scripts:")
for p in sorted(RF.rglob("*.py")):
    if "test" in p.name.lower():
        continue
    for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
        low = line.lower()
        if "reg_ar1" in low or "reg_mom" in low or "min_history" in low \
           or "min_obs" in low or ("btc" in low and "reg" in low):
            print(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:150]}")