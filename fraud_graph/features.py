"""Build transaction features and strictly prior-step account features."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "isFraud"


def _prior_role_stats(frame: pd.DataFrame, account_col: str, partner_col: str, amount_col: str, prefix: str) -> pd.DataFrame:
    """Attach account statistics through step-1; all events from current step are excluded."""
    amount2 = f"{prefix}_amount_squared"
    frame[amount2] = np.square(frame[amount_col].to_numpy(dtype="float64"))
    daily = (
        frame.groupby([account_col, "step"], sort=False, observed=True)
        .agg(n=(amount_col, "size"), total=(amount_col, "sum"), squares=(amount2, "sum"), low=(amount_col, "min"), high=(amount_col, "max"))
        .reset_index()
        .sort_values([account_col, "step"], kind="mergesort")
    )
    frame.drop(columns=[amount2], inplace=True)
    g = daily.groupby(account_col, sort=False, observed=True)
    for col in ["n", "total", "squares"]:
        daily[f"cum_{col}"] = g[col].cumsum()
    daily["available_step"] = daily["step"] + 1
    daily[f"{prefix}_prior_count"] = daily["cum_n"]
    daily[f"{prefix}_prior_sum"] = daily["cum_total"]
    daily[f"{prefix}_prior_sumsq"] = daily["cum_squares"]
    daily[f"{prefix}_prior_mean"] = daily[f"cum_total"] / daily["cum_n"].clip(lower=1)
    daily[f"{prefix}_prior_std"] = np.sqrt((daily["cum_squares"] / daily["cum_n"].clip(lower=1) - daily[f"{prefix}_prior_mean"] ** 2).clip(lower=0))
    daily[f"{prefix}_prior_min"] = g["low"].cummin()
    daily[f"{prefix}_prior_max"] = g["high"].cummax()
    daily[f"{prefix}_last_seen_step"] = daily["step"]
    stats = [c for c in daily.columns if c.startswith(prefix + "_")]
    snapshots = daily[[account_col, "available_step", *stats]].sort_values(["available_step", account_col], kind="mergesort")
    # First occurrence of each directed account pair yields an exact historical
    # distinct-counterparty count, which is separate from multigraph edge degree.
    first_pairs = frame[[account_col, partner_col, "step"]].sort_values("step", kind="mergesort")
    first_pairs.drop_duplicates([account_col, partner_col], keep="first", inplace=True)
    introductions = first_pairs.groupby([account_col, "step"], sort=False, observed=True).size().rename("new_counterparties").reset_index()
    introductions.sort_values([account_col, "step"], kind="mergesort", inplace=True)
    introductions["unique_counterparties"] = introductions.groupby(account_col, sort=False, observed=True)["new_counterparties"].cumsum()
    introductions["available_step"] = introductions["step"] + 1
    partner_snapshots = introductions[[account_col, "available_step", "unique_counterparties"]].sort_values(["available_step", account_col], kind="mergesort")
    left = frame[[account_col, "step"]].copy()
    left["_row_id"] = np.arange(len(frame), dtype=np.int64)
    left = left.sort_values(["step", account_col], kind="mergesort")
    matched = pd.merge_asof(left, snapshots, left_on="step", right_on="available_step", by=account_col, direction="backward", allow_exact_matches=True).sort_values("_row_id", kind="mergesort")
    matched = pd.merge_asof(matched.sort_values(["step", account_col], kind="mergesort"), partner_snapshots,
                            left_on="step", right_on="available_step", by=account_col, direction="backward",
                            allow_exact_matches=True, suffixes=("", "_partner")).sort_values("_row_id", kind="mergesort")
    matched[f"{prefix}_steps_since_last"] = matched["step"] - matched[f"{prefix}_last_seen_step"]
    matched.rename(columns={"unique_counterparties": f"{prefix}_prior_unique_counterparties"}, inplace=True)
    matched.drop(columns=["_row_id", "available_step", "available_step_partner", f"{prefix}_last_seen_step"], errors="ignore", inplace=True)
    return matched.reset_index(drop=True)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build prediction-time features plus labels, excluding post-transaction fields.

    PaySim's new-balance fields are outcomes of the current transaction, so they are
    deliberately not returned as predictors. Account IDs and existing fraud flags
    are also excluded.
    """
    frame = frame.sort_values("step", kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame(index=frame.index)
    out["step"] = frame["step"].astype("int16")
    out["type"] = frame["type"].astype("category")
    for col in ["amount", "oldbalanceOrg", "oldbalanceDest"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    out[TARGET] = frame[TARGET].astype("uint8")
    origin = _prior_role_stats(frame, "nameOrig", "nameDest", "amount", "orig_behavior")
    destination = _prior_role_stats(frame, "nameDest", "nameOrig", "amount", "dest_behavior")
    for stats in [origin, destination]:
        for col in stats.columns:
            if col not in {"step", "nameOrig", "nameDest"}:
                out[col] = pd.to_numeric(stats[col], errors="coerce").astype("float32")
    # These are role-correct edge features: source's historical out-degree and
    # destination's historical in-degree, with distinct counterparties and repeat
    # intensity. They are all strictly prior-step snapshots.
    out["graph_source_out_degree_prior"] = origin["orig_behavior_prior_count"].astype("float32")
    out["graph_destination_in_degree_prior"] = destination["dest_behavior_prior_count"].astype("float32")
    out["graph_source_out_neighbors_prior"] = origin["orig_behavior_prior_unique_counterparties"].astype("float32")
    out["graph_destination_in_neighbors_prior"] = destination["dest_behavior_prior_unique_counterparties"].astype("float32")
    out["graph_source_repeat_intensity_prior"] = (
        out["graph_source_out_degree_prior"] / (out["graph_source_out_neighbors_prior"] + 1)
    )
    out["graph_destination_repeat_intensity_prior"] = (
        out["graph_destination_in_degree_prior"] / (out["graph_destination_in_neighbors_prior"] + 1)
    )
    return out


def feature_sets() -> dict[str, list[str]]:
    transaction = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
    transaction_no_balances = ["step", "type", "amount"]
    behavior = ["orig_behavior_prior_count", "orig_behavior_prior_sum", "orig_behavior_prior_mean", "orig_behavior_prior_std", "orig_behavior_prior_min", "orig_behavior_prior_max", "orig_behavior_steps_since_last", "dest_behavior_prior_count", "dest_behavior_prior_sum", "dest_behavior_prior_mean", "dest_behavior_prior_std", "dest_behavior_prior_min", "dest_behavior_prior_max", "dest_behavior_steps_since_last"]
    graph = ["graph_source_out_neighbors_prior", "graph_destination_in_neighbors_prior",
             "graph_source_repeat_intensity_prior", "graph_destination_repeat_intensity_prior"]
    return {
        "A_pretransaction": transaction,
        "A_no_balance_ablation": transaction_no_balances,
        "B_behavior": transaction + behavior,
        "C_graph": transaction + behavior + graph,
    }

