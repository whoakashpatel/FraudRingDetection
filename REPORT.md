# PaySim Transaction Fraud and Coordinated-Group Discovery

## Objective and dataset

This offline ASBD prototype studies two distinct tasks: transaction-level fraud classification and ranking candidate groups with coordinated transaction patterns. PaySim provides transaction labels, not verified fraudulent-account or ring identifiers. Thus, transaction results are evaluated on PaySim; ring-recovery results are evaluated only on controlled planted groups.

The local file contains 6,362,620 transactions over steps 1–743, including 8,213 fraud-labelled rows (0.1291%). The chronological split is train through step 520, validation 521–631, and test from 632 onward. Train has 6,082,007 rows / 5,781 frauds; validation has 191,147 / 1,180; test has 89,466 / 1,252. Test prevalence (1.399%) is about 14.7 times training prevalence (0.095%), so the final period is a substantial distribution shift.

The streamed schema audit found no missing values and no exact duplicate rows in its 64-bit row-hash scan. Transaction types are CASH_OUT (2,237,500), PAYMENT (2,151,495), CASH_IN (1,399,284), TRANSFER (532,909), and DEBIT (41,432). All 8,213 fraud labels occur on CASH_OUT (4,116) or TRANSFER (4,097). `isFlaggedFraud` is present on only 16 rows and is excluded from predictors. Amount quantiles and endpoint activity estimates are documented in the EDA JSON; the endpoint frequency sample is explicitly not a full-data estimate.

## Transaction classification

All primary predictors are available at transaction decision time: transaction type, amount, simulation step, and pre-transaction origin/destination balances. Post-transaction balances, `isFlaggedFraud`, account IDs, and target-derived aggregates are excluded. Behavioral and graph features use strictly earlier steps; transactions in the same step do not update one another because PaySim supplies no within-step order.

The experiment compares a transfer/cash-out amount rule, class-weighted Logistic Regression, and XGBoost models A/B/C. A uses pre-transaction fields; a separate ablation removes both balance features. B adds prior endpoint behavior. C adds prior distinct-neighbor and repeat-intensity features. XGBoost uses histogram trees, training-derived `scale_pos_weight` (1,051.07), validation AP for early stopping, and the same chronological split across models. Test ranking uses stable row order to break score ties and selects exactly the requested top-k alerts.

| Model | Test AP | Validation AP | P/R @90 alerts | P/R @448 alerts | P/R @895 alerts |
|---|---:|---:|---:|---:|---:|
| Transfer/cash-out amount rule | 0.2183 | — | 0.7889 / 0.0567 | 0.4554 / 0.1629 | 0.3486 / 0.2492 |
| Logistic Regression A | 0.5166 | 0.3863 | 1.0000 / 0.0719 | 0.7589 / 0.2716 | 0.5184 / 0.3706 |
| XGBoost A, pre-transaction | 0.9710 | 0.9510 | 1.0000 / 0.0719 | 1.0000 / 0.3578 | 0.9899 / 0.7077 |
| XGBoost A, no balances | 0.3261 | 0.2490 | 1.0000 / 0.0719 | 0.6161 / 0.2204 | 0.4335 / 0.3099 |
| XGBoost B, behavior | 0.9375 | 0.9106 | 1.0000 / 0.0719 | 1.0000 / 0.3578 | 0.9844 / 0.7037 |
| XGBoost C, graph | 0.9449 | 0.9204 | 1.0000 / 0.0719 | 1.0000 / 0.3578 | 0.9888 / 0.7069 |

The 90-alert setting selects exactly 90 rows, so even perfect precision gives only 7.19% recall. Score ties are common at that cutoff: a simple `score >= cutoff` would exceed the budget, so the metrics include tie-overflow diagnostics. At 448 and 895 alerts, model rankings recover more frauds while precision falls.

The balance ablation exposes a major simulator shortcut: **8,034 of 8,213 fraud rows (97.82%) satisfy `amount == oldbalanceOrg`, while none of 6,354,407 non-fraud rows do**. These fields are available before the transaction, so this is not post-event leakage; it is a highly distinctive PaySim generator signature. XGBoost A's AP of 0.9710 should not be interpreted as expected real-institution performance. Removing balances drops AP to 0.3261. Behavioral features lower AP relative to A. Graph features lift B by about 0.0074 AP on test (0.0098 on validation), with small changes at fixed alert budgets; most model importance remains on transaction type and balance/amount fields. This does not establish a large graph benefit. A reached the configured 600-tree cap rather than early-stopping; B and C stopped at 126 and 312 trees, while the no-balance ablation stopped at 2. Treat A's score as the result at the tested cap, not as a confirmed converged optimum.

## Candidate-group discovery

Group discovery never reads transaction fraud labels or classifier scores. For each non-overlapping 24-step window, it proposes weak components, smaller strongly connected components, fan-in/fan-out neighborhoods, and Charikar-style greedy dense subsets. Dense-subgraph peeling is bounded to weak components of at most 20 accounts. A candidate must contain at least three accounts in a directed cycle or at least two accounts with an inflow at an earlier step than a later outflow across multiple steps. The score reports structural density, cycles/reciprocity, ordered relays, temporal compactness, repeated pairs, per-account flow continuity, and amount similarity. The score is a manually fixed heuristic, not a trained probability.

The final full PaySim scan processed all 31 windows in 276.05 seconds at 734 MiB peak RSS. **No group passed the strict eligibility rule.** A prior, looser screen found 31 compact weak components, but each had only one temporally ordered relay and no directed cycle; those chain-like groups were excluded from the final ring-candidate output. This means no group met the chosen coordination criteria in this scan. It does not mean PaySim contains no fraud, and the data cannot validate PaySim ring precision/recall because it has no ring ground truth. The window boundaries and 20-account proposal cap can also miss longer or cross-window structures.

The controlled benchmark plants 30 cyclic or acyclic relay groups among 90 chain/dense reciprocal decoy episodes, over five random seeds. Overlapping proposals are ranked and each planted group can be matched once at account-set IoU ≥0.5. Mean group AP is **0.241 ± 0.040**, precision@30 **0.447 ± 0.069**, member recovery@30 **0.357 ± 0.055**, and ring recall@30 **0.447 ± 0.069**. Candidate transaction-count and density rankings have mean AP **0.201** and **0.097**. All planted groups appear somewhere among candidates, but top-rank recovery remains modest. This generator is still small and easier than real networks: background noise is not strongly connected to planted groups, and there are no verified real fraud labels. Treat these metrics as a controlled recovery check, not real-world performance.

## Reproduction and artifacts

Run the commands in [README.md](README.md). `outputs/metrics.json` records transaction splits, parameters, AP, exact fixed-budget precision/recall, runtime, and peak RSS. `outputs/test_predictions.npz` stores test labels/steps/scores; `outputs/feature_importance.json` records XGBoost importances. The completed transaction experiment used a 600-tree cap and finished in **397.54 seconds** at **7,486.2 MiB** peak RSS. `outputs/ring_candidates.json` records the complete label-free scan (zero eligible groups), and `outputs/ring_recovery_synthetic.json` contains five-seed controlled metrics. `outputs/suspicious_groups.json` is a legacy artifact ranked using fraud labels; do not use it as current discovery output.

Key artifacts: [data profile](outputs/data_profile.json), [EDA summary](outputs/eda_summary.json), [EDA figure](outputs/eda_overview.png), [model comparison figure](outputs/model_comparison.png), [transaction metrics](outputs/metrics.json), [full timeline ring scan](outputs/ring_candidates.json), and [controlled ring recovery](outputs/ring_recovery_synthetic.json).

## Limitations and next steps

The classifier has one chronological split and substantial prevalence shift; the XGBoost performance is strongly influenced by a PaySim balance signature. Next work should test additional chronological cutoffs and realistic data with independent labels, and report balance-free results prominently. For group discovery, expand controlled injections to overlapping rings, noisy bridges, hubs, partial activity, and larger groups; evaluate Charikar/ANTICO/SADE-style mining with one-to-one group metrics and threshold sensitivity. The current PaySim result does not support claiming that a fraud ring was discovered. Streaming remains future work after offline group recovery is validated.

Method references: [Charikar, “Greedy Approximation Algorithms for Finding Dense Components in a Graph”](https://dblp.org/rec/conf/approx/Charikar00); [SADE](https://doi.org/10.1145/3383455.3422548); [ANTICO](https://doi.org/10.1145/3534678.3539252).
