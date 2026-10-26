"""Memory-conscious PaySim schema and feasibility inspection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def profile_csv(path: Path, chunksize: int = 500_000) -> dict:
    usecols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    rows = fraud = flagged = 0
    steps: list[int] = []
    fraud_by_type: Counter = Counter()
    rows_by_type: Counter = Counter()
    missing: Counter = Counter()
    amount_sum = 0.0
    amount_min = float("inf")
    amount_max = float("-inf")
    seen_hashes: set[int] = set()
    duplicate_rows = 0
    schema = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        rows += len(chunk)
        schema = schema or {c: str(t) for c, t in chunk.dtypes.items()}
        missing.update(chunk.isna().sum().to_dict())
        fraud += int(chunk["isFraud"].sum())
        flagged += int(chunk["isFlaggedFraud"].sum())
        amount_sum += float(chunk["amount"].sum())
        amount_min = min(amount_min, float(chunk["amount"].min()))
        amount_max = max(amount_max, float(chunk["amount"].max()))
        steps.extend([int(chunk.step.min()), int(chunk.step.max())])
        rows_by_type.update(chunk["type"].value_counts().to_dict())
        fraud_by_type.update(chunk.loc[chunk.isFraud.eq(1), "type"].value_counts().to_dict())
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
        for h in hashes:
            ih = int(h)
            if ih in seen_hashes:
                duplicate_rows += 1
            else:
                seen_hashes.add(ih)
    return {
        "input": str(path), "rows": rows, "columns": schema, "missing_values": dict(missing),
        "exact_duplicate_rows_by_64bit_hash": duplicate_rows, "fraud_rows": fraud,
        "fraud_rate": fraud / rows if rows else None, "flagged_fraud_rows": flagged,
        "steps_min": min(steps) if steps else None, "steps_max": max(steps) if steps else None,
        "amount_min": amount_min if rows else None, "amount_max": amount_max if rows else None,
        "amount_mean": amount_sum / rows if rows else None, "rows_by_type": dict(rows_by_type),
        "fraud_rows_by_type": dict(fraud_by_type),
        "notes": ["Duplicate counts use 64-bit row hashes, so a theoretical hash collision is possible.", "PaySim labels transactions; they do not establish independently verified fraud rings."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_profile.json"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    result = profile_csv(args.input, args.chunksize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

