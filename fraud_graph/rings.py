"""Offline, time-windowed candidate discovery for coordinated account activity.

The output is an investigative ranking, not a declaration that an account is
fraudulent. PaySim has transaction labels but no independently verified ring IDs.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


EDGE_TYPES = {"TRANSFER", "CASH_OUT"}
INPUT_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest"]


def _group_evidence(rows: pd.DataFrame, members: set[str], window_width: int) -> dict:
    """Score a group using only its transaction rows and explainable evidence."""
    local = rows.loc[rows.nameOrig.isin(members) & rows.nameDest.isin(members)]
    n_accounts = len(members)
    n_rows = len(local)
    if not n_rows or n_accounts < 2:
        return {"score": 0.0, "evidence": {}}

    pair_counts = local.groupby(["nameOrig", "nameDest"], sort=False).size()
    pairs = set(pair_counts.index.tolist())
    possible_pairs = n_accounts * (n_accounts - 1)
    density = min(1.0, len(pairs) / possible_pairs)
    reciprocal_pairs = sum((dst, src) in pairs for src, dst in pairs) / max(1, len(pairs))
    local_graph = nx.DiGraph()
    local_graph.add_edges_from(pairs)
    cycle_edges = set()
    cycle_accounts = set()
    for strongly_connected in nx.strongly_connected_components(local_graph):
        if len(strongly_connected) > 1:
            cycle_accounts.update(strongly_connected)
            cycle_edges.update((src, dst) for src, dst in pairs
                               if src in strongly_connected and dst in strongly_connected)
    cycle_edge_fraction = len(cycle_edges) / max(1, len(pairs))

    out_count = local.groupby("nameOrig").size()
    in_count = local.groupby("nameDest").size()
    relay_accounts = set(out_count.index).intersection(in_count.index)
    relay_fraction = len(relay_accounts) / n_accounts
    first_in_step = local.groupby("nameDest").step.min()
    last_out_step = local.groupby("nameOrig").step.max()
    ordered_relays = {node for node in relay_accounts if first_in_step[node] < last_out_step[node]}
    ordered_relay_fraction = len(ordered_relays) / n_accounts

    span = max(1, int(local.step.max() - local.step.min() + 1))
    active_steps = int(local.step.nunique())
    temporal_concentration = 1.0 - (span - 1) / max(1, window_width - 1)
    temporal_concentration = max(0.0, min(1.0, temporal_concentration))
    repeated_pair_fraction = float((pair_counts[pair_counts > 1].sum() / n_rows))

    incoming = local.groupby("nameDest").amount.sum()
    outgoing = local.groupby("nameOrig").amount.sum()
    account_flow_ratios = [min(float(incoming[node]), float(outgoing[node]))
                           / max(float(incoming[node]), float(outgoing[node]), 1e-12)
                           for node in relay_accounts]
    flow_continuity = float(np.mean(account_flow_ratios)) if account_flow_ratios else 0.0
    amount_cv = float(local.amount.std(ddof=0) / max(float(local.amount.mean()), 1e-12))

    # Fixed, declared weights; this is a transparent heuristic, not a trained model.
    amount_similarity = 1.0 / (1.0 + amount_cv)
    score = (0.08 * density + 0.05 * reciprocal_pairs + 0.12 * ordered_relay_fraction
             + 0.40 * temporal_concentration + 0.03 * repeated_pair_fraction
             + 0.05 * flow_continuity + 0.20 * cycle_edge_fraction
             + 0.07 * amount_similarity)
    return {
        "score": float(score),
        "evidence": {
            "directed_density": float(density),
            "reciprocal_pair_fraction": float(reciprocal_pairs),
            "directed_cycle_edge_fraction": float(cycle_edge_fraction),
            "directed_cycle_account_count": len(cycle_accounts),
            "accounts_with_both_inflow_and_outflow_fraction": float(relay_fraction),
            "relay_account_count": len(relay_accounts),
            "temporally_ordered_relay_account_count": len(ordered_relays),
            "temporally_ordered_relay_fraction": float(ordered_relay_fraction),
            "window_temporal_compactness": float(temporal_concentration),
            "repeated_pair_transaction_fraction": float(repeated_pair_fraction),
            "mean_account_in_out_amount_continuity": flow_continuity,
            "amount_coefficient_of_variation": amount_cv,
            "amount_similarity": float(amount_similarity),
            "transaction_count": n_rows,
            "active_steps": active_steps,
            "time_span_steps": span,
        },
    }


def _charikar_style_dense_subset(graph: nx.DiGraph, min_size: int = 3) -> set[str]:
    """Greedy minimum-degree peeling on the undirected simple view; small groups only."""
    working = nx.Graph(graph)
    best_members = set(working.nodes)
    best_density = -1.0
    while len(working) >= min_size:
        members = set(working.nodes)
        density = working.number_of_edges() / max(1, len(members))
        if density > best_density:
            best_density = density
            best_members = members
        node = min(working, key=lambda account: (working.degree(account), str(account)))
        working.remove_node(node)
    return best_members


def discover_in_window(
    rows: pd.DataFrame,
    *,
    window_start: int,
    window_end: int,
    min_accounts: int = 3,
    max_accounts: int = 20,
    min_transactions: int = 4,
    min_active_steps: int = 2,
    min_relay_accounts: int = 2,
    min_cycle_accounts: int = 3,
) -> list[dict]:
    """Find weak components within one bounded window and rank by coordination evidence.

    Components are candidate boundaries only. They must contain a directed cycle,
    or a multi-step relay motif; connectivity alone cannot create a candidate.
    """
    edges = rows.loc[
        rows.type.isin(EDGE_TYPES) & (rows.nameOrig != rows.nameDest)
        & rows.amount.gt(0), INPUT_COLUMNS
    ].copy()
    if edges.empty:
        return []
    account_codes, accounts = pd.factorize(
        pd.concat([edges.nameOrig, edges.nameDest], ignore_index=True), sort=False
    )
    source_codes = account_codes[:len(edges)]
    destination_codes = account_codes[len(edges):]
    adjacency = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (source_codes, destination_codes)),
        shape=(len(accounts), len(accounts)),
    ).tocsr()
    _, node_component = connected_components(adjacency, directed=False, return_labels=True)
    component_nodes: dict[int, list[str]] = {}
    for node_code, component_id in enumerate(node_component):
        component_nodes.setdefault(int(component_id), []).append(str(accounts[node_code]))

    candidates = []
    retained: dict[int, set[str]] = {}
    for component_id, component in component_nodes.items():
        if not min_accounts <= len(component) <= max_accounts:
            continue
        retained[component_id] = set(component)
    if not retained:
        return []
    edges["_component"] = node_component[source_codes]
    component_rows = {int(key): value for key, value in edges.dropna(subset=["_component"]).groupby("_component", sort=False)}
    for component_id, component in retained.items():
        subrows = component_rows.get(component_id)
        if subrows is None or len(subrows) < min_transactions:
            continue
        proposals: dict[frozenset[str], set[str]] = {frozenset(component): {"weak_component"}}
        local_graph = nx.DiGraph()
        local_graph.add_edges_from(set(zip(subrows.nameOrig, subrows.nameDest)))
        dense_members = _charikar_style_dense_subset(local_graph, min_size=min_accounts)
        if len(dense_members) >= min_accounts:
            proposals.setdefault(frozenset(dense_members), set()).add("charikar_style_dense_subgraph")
        for scc in nx.strongly_connected_components(local_graph):
            if 3 <= len(scc) <= max_accounts:
                proposals.setdefault(frozenset(scc), set()).add("strong_component")

        incoming_neighbors: dict[str, set[str]] = {}
        outgoing_neighbors: dict[str, set[str]] = {}
        for src, dst in set(zip(subrows.nameOrig, subrows.nameDest)):
            outgoing_neighbors.setdefault(str(src), set()).add(str(dst))
            incoming_neighbors.setdefault(str(dst), set()).add(str(src))
        for center in set(incoming_neighbors).intersection(outgoing_neighbors):
            inbound = incoming_neighbors[center]
            outbound = outgoing_neighbors[center]
            if len(inbound) < 2 or len(outbound) < 2:
                continue
            members = {center, *inbound, *outbound}
            if len(members) > max_accounts:
                continue
            if len(members) >= min_accounts:
                proposals.setdefault(frozenset(members), set()).add("fan_in_out")

        for member_key, proposal_types in proposals.items():
            members = set(member_key)
            scored = _group_evidence(subrows, members, max(1, window_end - window_start + 1))
            evidence = scored["evidence"]
            if evidence.get("transaction_count", 0) < min_transactions:
                continue
            has_cycle = evidence["directed_cycle_account_count"] >= min_cycle_accounts
            has_temporal_relay = (evidence["active_steps"] >= min_active_steps
                                  and evidence["temporally_ordered_relay_account_count"] >= min_relay_accounts)
            if not (has_cycle or has_temporal_relay):
                continue
            candidates.append({
                "window_start_step": int(window_start),
                "window_end_step": int(window_end),
                "accounts": sorted(members),
                "account_count": len(members),
                "proposal_types": sorted(proposal_types),
                "score": scored["score"],
                "evidence": evidence,
            })
    candidates.sort(key=lambda item: (-item["score"], -item["evidence"]["transaction_count"], item["accounts"]))
    return candidates


def _stream_windows(path: Path, window_steps: int, chunksize: int) -> Iterable[tuple[int, int, pd.DataFrame]]:
    """Stream PaySim rows by non-overlapping step windows (requires sorted steps)."""
    current_window: int | None = None
    pending: list[pd.DataFrame] = []
    last_step: int | None = None
    for chunk in pd.read_csv(path, usecols=INPUT_COLUMNS, chunksize=chunksize):
        chunk["step"] = chunk["step"].astype(int)
        if not chunk.step.is_monotonic_increasing:
            raise ValueError("Input rows must be ordered by non-decreasing step for streaming windows.")
        if last_step is not None and int(chunk.step.min()) < last_step:
            raise ValueError("Input rows must be ordered by non-decreasing step for streaming windows.")
        last_step = int(chunk.step.max())
        chunk = chunk.loc[chunk.type.isin(EDGE_TYPES)].copy()
        if chunk.empty:
            continue
        chunk["_window"] = ((chunk.step - 1) // window_steps).astype(int)
        for window_id, part in chunk.groupby("_window", sort=True):
            window_id = int(window_id)
            if current_window is None:
                current_window = window_id
            if window_id != current_window:
                yield (current_window * window_steps + 1,
                       (current_window + 1) * window_steps,
                       pd.concat(pending, ignore_index=True))
                pending = []
                current_window = window_id
            pending.append(part[INPUT_COLUMNS])
    if current_window is not None and pending:
        yield (current_window * window_steps + 1,
               (current_window + 1) * window_steps,
               pd.concat(pending, ignore_index=True))


def discover_paysim(
    path: Path,
    *,
    window_steps: int = 24,
    chunksize: int = 250_000,
    min_accounts: int = 3,
    max_accounts: int = 20,
    min_transactions: int = 4,
    min_active_steps: int = 2,
    min_relay_accounts: int = 2,
    min_cycle_accounts: int = 3,
    limit: int = 100,
    window_limit: int | None = None,
    progress: bool = False,
) -> dict:
    """Discover unlabeled candidate groups without reading fraud labels."""
    all_candidates: list[dict] = []
    processed_windows = 0
    total_candidate_count = 0
    for start, end, rows in _stream_windows(path, window_steps, chunksize):
        if window_limit is not None and processed_windows >= window_limit:
            break
        processed_windows += 1
        window_candidates = discover_in_window(
            rows, window_start=start, window_end=end,
            min_accounts=min_accounts, max_accounts=max_accounts,
            min_transactions=min_transactions, min_active_steps=min_active_steps,
            min_relay_accounts=min_relay_accounts, min_cycle_accounts=min_cycle_accounts,
        )
        total_candidate_count += len(window_candidates)
        # Any global top-limit candidate must also be in its window's top limit.
        all_candidates.extend(window_candidates[:limit])
        if progress:
            print(f"Processed window {processed_windows}: steps {start}-{end}, "
                  f"{len(window_candidates):,} eligible candidates", flush=True)
    all_candidates.sort(key=lambda item: (-item["score"], -item["evidence"]["transaction_count"],
                                          item["window_start_step"], item["accounts"]))
    return {
        "method": "Non-overlapping time-window weak components plus overlapping SCC, fan-in/fan-out, and Charikar-style dense-subgraph proposals; eligible groups ranked by fixed structural, temporal, and flow evidence.",
        "window_steps": window_steps,
        "processed_windows": processed_windows,
        "window_limit": window_limit,
        "complete_timeline_scan": window_limit is None,
        "candidate_count": total_candidate_count,
        "label_use": "No transaction fraud labels or model predictions are read or used.",
        "interpretation": "Candidates are unverified groups for investigation; connectivity and score do not establish fraud or common control.",
        "candidate_eligibility": "At least the configured number of accounts in directed cycles, or at least the configured number of temporally ordered relays across the minimum number of active steps.",
        "score_definition": {
            "weighted_terms": {
                "directed_density": 0.08,
                "reciprocal_pair_fraction": 0.05,
                "directed_cycle_edge_fraction": 0.20,
                "temporally_ordered_relay_fraction": 0.12,
                "window_temporal_compactness": 0.40,
                "repeated_pair_transaction_fraction": 0.03,
                "mean_account_in_out_amount_continuity": 0.05,
                "amount_similarity": 0.07,
            },
            "note": "Weights are a declared heuristic and have not been optimized against PaySim labels.",
        },
        "candidates": all_candidates[:limit],
    }


def _synthetic_benchmark(seed: int = 23, n_rings: int = 30, n_decoys: int = 90) -> dict:
    """Controlled recovery benchmark with temporal ring episodes and noisy decoys."""
    rng = np.random.default_rng(seed)
    rows = []
    truth: list[set[str]] = []
    episode = 0
    # Half are cyclic relays, half are feed-forward layering patterns.
    for ring_id in range(n_rings):
        nodes = [f"R{ring_id}_{i}" for i in range(5)]
        truth.append(set(nodes))
        base = episode * 10 + 1
        amounts = rng.uniform(800, 1300, size=5)
        if ring_id % 2 == 0:
            motif = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]),
                     (nodes[3], nodes[0]), (nodes[0], nodes[4]), (nodes[1], nodes[4])]
            offsets = [0, 1, 2, 3, 0, 1]
        else:
            motif = [(nodes[0], nodes[1]), (nodes[0], nodes[2]), (nodes[1], nodes[3]),
                     (nodes[2], nodes[3]), (nodes[3], nodes[4])]
            offsets = list(range(len(motif)))
        for j, (src, dst) in enumerate(motif):
            rows.append((base + offsets[j], "TRANSFER", float(amounts[j % 5]), src, dst))
        # Modest background links make the group less perfectly isolated.
        rows.append((base + 5, "CASH_OUT", float(rng.uniform(700, 1400)), nodes[4], f"N{ring_id}_sink"))
        episode += 1
    # Decoys include repeated chains and dense benign-looking interaction clusters.
    for decoy_id in range(n_decoys):
        nodes = [f"D{decoy_id}_{i}" for i in range(5)]
        base = episode * 10 + 1
        if decoy_id % 3:
            for j in range(7):
                src = nodes[min(j // 2, 3)]
                dst = nodes[min(j // 2 + 1, 4)]
                rows.append((base + int(rng.integers(0, 8)), "TRANSFER", float(rng.uniform(100, 1800)), src, dst))
        else:
            dense_pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[0]), (nodes[1], nodes[2]),
                           (nodes[2], nodes[1]), (nodes[2], nodes[3]), (nodes[3], nodes[2]),
                           (nodes[3], nodes[4]), (nodes[4], nodes[3])]
            for j, (src, dst) in enumerate(dense_pairs):
                rows.append((base + int(rng.integers(0, 10)), "TRANSFER", float(rng.uniform(100, 1800)), src, dst))
        episode += 1
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    # Add low-rate noisy traffic among background accounts. This is transaction-level
    # clutter, but does not connect into the planted ring episodes.
    noisy = []
    for i in range(600):
        src, dst = rng.choice(120, size=2, replace=False)
        noisy.append((int(rng.integers(1, max(2, episode * 10))), "PAYMENT",
                      float(rng.uniform(1, 400)), f"B{src}", f"B{dst}"))
    frame = pd.concat([frame, pd.DataFrame(noisy, columns=INPUT_COLUMNS)], ignore_index=True)
    predictions = []
    for window_id, part in frame.groupby(((frame.step - 1) // 10), sort=False):
        start, end = int(window_id * 10 + 1), int((window_id + 1) * 10)
        predictions.extend(discover_in_window(part, window_start=start, window_end=end,
                                               min_accounts=3, max_accounts=10, min_transactions=4))

    candidate_positive = []
    for candidate in predictions:
        accounts = set(candidate["accounts"])
        candidate_positive.append(any(
            len(accounts & ring) / len(accounts | ring) >= 0.5 for ring in truth
        ))
    candidate_positive = np.asarray(candidate_positive, dtype=bool)
    scores = np.asarray([candidate["score"] for candidate in predictions], dtype=float)
    k = min(30, len(scores))
    def rank_summary(rank_scores: np.ndarray) -> dict:
        ranked = np.argsort(-rank_scores, kind="stable")
        claimed: set[int] = set()
        true_positives = 0
        precision_sum = 0.0
        top_true_positives = 0
        top_member_recovery = 0.0
        for rank, idx in enumerate(ranked, start=1):
            accounts = set(predictions[int(idx)]["accounts"])
            overlaps = [(len(accounts & ring) / len(accounts | ring), ring_idx)
                        for ring_idx, ring in enumerate(truth) if ring_idx not in claimed]
            best_overlap, best_ring = max(overlaps, default=(0.0, -1))
            matched = best_overlap >= 0.5
            if matched:
                claimed.add(best_ring)
                true_positives += 1
                precision_sum += true_positives / rank
            if rank <= k:
                if matched:
                    top_true_positives += 1
                    top_member_recovery += len(accounts & truth[best_ring]) / len(truth[best_ring])
        return {
            "average_precision": precision_sum / n_rings,
            "precision_at_30": top_true_positives / max(1, k),
            "member_recovery_at_30": top_member_recovery / max(1, k),
            "ring_recall_at_30": top_true_positives / n_rings,
            "candidate_coverage_recall": true_positives / n_rings,
        }

    model_summary = rank_summary(scores)
    volume_scores = np.asarray([candidate["evidence"]["transaction_count"] for candidate in predictions], dtype=float)
    density_scores = np.asarray([candidate["evidence"]["directed_density"] for candidate in predictions], dtype=float)
    return {
        "synthetic_only": True,
        "seed": seed,
        "planted_ring_count": n_rings,
        "decoy_episode_count": n_decoys,
        "transaction_rows_including_noise": int(len(frame)),
        "candidate_count": len(predictions),
        "group_match_rule": "Candidates are ranked; each planted ring may match only once, at the first candidate with account-set intersection-over-union at least 0.5.",
        "score_separation": {
            "positive_candidate_mean": float(scores[candidate_positive].mean()) if candidate_positive.any() else None,
            "negative_candidate_mean": float(scores[~candidate_positive].mean()) if (~candidate_positive).any() else None,
        },
        "rankings": {
            "interpretable_group_score": model_summary,
            "candidate_transaction_count_baseline": rank_summary(volume_scores),
            "candidate_density_baseline": rank_summary(density_scores),
        },
        "limitations": "Controlled generator only; ring patterns, background, and thresholds are artificial and do not validate PaySim ring accuracy.",
    }


def _synthetic_benchmark_repeated(repeats: int = 5, first_seed: int = 23) -> dict:
    """Repeat the planted benchmark across seeds and summarize ranking variability."""
    runs = [_synthetic_benchmark(seed=first_seed + offset) for offset in range(repeats)]
    metrics = ("average_precision", "precision_at_30", "member_recovery_at_30",
               "ring_recall_at_30", "candidate_coverage_recall")
    summaries = {}
    for ranking in runs[0]["rankings"]:
        summaries[ranking] = {}
        for metric in metrics:
            values = np.asarray([run["rankings"][ranking][metric] for run in runs], dtype=float)
            summaries[ranking][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
    return {
        "synthetic_only": True,
        "first_seed": first_seed,
        "repeat_count": repeats,
        "runs": runs,
        "summary_mean_and_std": summaries,
        "limitations": "Controlled generator only; these seeded repetitions do not validate PaySim ring accuracy or real-world performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ring_candidates.json"))
    parser.add_argument("--window-steps", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--min-accounts", type=int, default=3)
    parser.add_argument("--max-accounts", type=int, default=20)
    parser.add_argument("--min-transactions", type=int, default=4)
    parser.add_argument("--min-active-steps", type=int, default=2)
    parser.add_argument("--min-relay-accounts", type=int, default=2)
    parser.add_argument("--min-cycle-accounts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--window-limit", type=int, help="Stop after this many chronological windows; output will be partial.")
    parser.add_argument("--progress", action="store_true", help="Print progress after each time window.")
    parser.add_argument("--synthetic-only", action="store_true", help="Run controlled ring recovery without reading PaySim.")
    parser.add_argument("--synthetic-output", type=Path, default=Path("outputs/ring_recovery_synthetic.json"))
    parser.add_argument("--synthetic-repeats", type=int, default=5)
    parser.add_argument("--synthetic-first-seed", type=int, default=23)
    args = parser.parse_args()
    if args.synthetic_only:
        if args.synthetic_repeats < 1:
            parser.error("--synthetic-repeats must be at least 1")
        result = _synthetic_benchmark_repeated(args.synthetic_repeats, args.synthetic_first_seed)
        args.synthetic_output.parent.mkdir(parents=True, exist_ok=True)
        args.synthetic_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    started = time.perf_counter()
    result = discover_paysim(args.input, window_steps=args.window_steps, chunksize=args.chunksize,
                             min_accounts=args.min_accounts, max_accounts=args.max_accounts,
                             min_transactions=args.min_transactions,
                             min_active_steps=args.min_active_steps,
                             min_relay_accounts=args.min_relay_accounts,
                             min_cycle_accounts=args.min_cycle_accounts, limit=args.limit,
                             window_limit=args.window_limit, progress=args.progress)
    result["runtime_seconds"] = round(time.perf_counter() - started, 2)
    result["peak_process_max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
