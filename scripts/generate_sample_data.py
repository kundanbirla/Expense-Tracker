"""Write data/sample_synthetic_transactions.csv (fake data, safe to commit and screenshot)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.synthetic_data import generate_transactions  # noqa: E402

rows = generate_transactions()  # ends 2026-10-11, seed 42
df = pd.DataFrame(rows).drop(columns=["is_injected_anomaly"])
out = ROOT / "data/sample_synthetic_transactions.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df)} rows to {out}")
