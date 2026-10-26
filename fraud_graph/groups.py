"""Compatibility entry point for the label-free ring candidate workflow.

Use :mod:`fraud_graph.rings` for the implementation and options. This module is
kept so existing ``python -m fraud_graph.groups`` instructions do not silently
continue to produce label-ranked connected components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rings import discover_paysim


def suspicious_groups(path: Path, *, window_steps: int = 24, chunksize: int = 250_000,
                      limit: int = 100, min_accounts: int = 3,
                      max_accounts: int = 20, min_transactions: int = 4) -> dict:
    return discover_paysim(path, window_steps=window_steps, chunksize=chunksize,
                           limit=limit, min_accounts=min_accounts,
                           max_accounts=max_accounts, min_transactions=min_transactions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ring_candidates.json"))
    parser.add_argument("--window-steps", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--min-accounts", type=int, default=3)
    parser.add_argument("--max-accounts", type=int, default=20)
    parser.add_argument("--min-transactions", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    result = suspicious_groups(args.input, window_steps=args.window_steps, chunksize=args.chunksize,
                               min_accounts=args.min_accounts, max_accounts=args.max_accounts,
                               min_transactions=args.min_transactions, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
