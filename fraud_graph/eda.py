"""Chunked exploratory summaries and figures for the PaySim transaction file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(input_path: Path, outdir: Path, chunksize: int = 500_000, sample_cap: int = 250_000) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    by_step = Counter()
    fraud_by_step = Counter()
    by_type = Counter()
    fraud_by_type = Counter()
    amount_samples = []
    activity_sample = Counter()
    n_seen = 0
    for chunk in pd.read_csv(input_path, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"], chunksize=chunksize):
        by_step.update(chunk.groupby("step", observed=True).size().to_dict())
        fraud_by_step.update(chunk.loc[chunk.isFraud.eq(1)].groupby("step", observed=True).size().to_dict())
        by_type.update(chunk["type"].value_counts().to_dict())
        fraud_by_type.update(chunk.loc[chunk.isFraud.eq(1), "type"].value_counts().to_dict())
        vals = chunk.amount.to_numpy()
        if len(vals) > 20_000:
            vals = vals[:: max(1, len(vals) // 20_000)]
        amount_samples.append(vals)
        # Activity is explicitly estimated on a deterministic row sample, avoiding a huge account map.
        if n_seen < sample_cap:
            take = chunk.iloc[: min(len(chunk), sample_cap - n_seen)]
            activity_sample.update(take.nameOrig.value_counts().to_dict())
            activity_sample.update(take.nameDest.value_counts().to_dict())
            n_seen += len(take)
    amounts = np.concatenate(amount_samples)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    types = list(by_type)
    axes[0, 0].bar(types, [by_type[t] for t in types], color="#2878B5")
    axes[0, 0].set_title("Transactions by type")
    axes[0, 0].tick_params(axis="x", rotation=25)
    fraud_types = list(fraud_by_type)
    axes[0, 1].bar(fraud_types, [fraud_by_type[t] for t in fraud_types], color="#D1495B")
    axes[0, 1].set_title("Fraud-labeled transactions by type")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[1, 0].hist(np.log10(amounts + 1), bins=60, color="#59A14F")
    axes[1, 0].set_title("Transaction amount distribution (log10(amount + 1), sampled)")
    axes[1, 0].set_xlabel("log10(amount + 1)")
    steps = sorted(by_step)
    axes[1, 1].plot(steps, [by_step[s] for s in steps], label="all")
    axes[1, 1].plot(steps, [fraud_by_step[s] for s in steps], label="fraud label")
    axes[1, 1].set_title("Transaction volume by simulation step")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "eda_overview.png", dpi=160)
    plt.close(fig)
    result = {
        "rows": int(sum(by_step.values())), "rows_by_type": dict(by_type), "fraud_rows_by_type": dict(fraud_by_type),
        "fraud_rows_by_step": {str(k): int(v) for k, v in sorted(fraud_by_step.items())},
        "transactions_by_step": {str(k): int(v) for k, v in sorted(by_step.items())},
        "amount_sample_size": int(len(amounts)), "amount_sample_quantiles": {str(q): float(np.quantile(amounts, q)) for q in [0.5, 0.9, 0.99, 0.999]},
        "activity_sample": {"transaction_rows_sampled": n_seen, "top_account_ids_in_endpoint_sample": activity_sample.most_common(20),
                            "note": "Source/destination endpoint counts over the first deterministic sample_cap rows; not full-data top accounts."},
    }
    (outdir / "eda_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

