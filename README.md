# PaySim Transaction Fraud and Coordinated-Group Discovery

Offline Analytics and Systems of Big Data project with two separate tasks: transaction-level fraud classification and time-windowed candidate-group discovery. PaySim supplies transaction labels, not verified fraud-ring labels. The project never converts a fraud-labelled transaction into a fraud-labelled account or group.

## Data

Place the Kaggle PaySim `data.csv` at `dataset/data.csv`. The project does not download data or require credentials. Expected fields are the standard PaySim transaction type, step, source/destination IDs, pre/post balances, `isFraud`, and `isFlaggedFraud`. The latter is not a predictor. Post-transaction balances are excluded from the primary classifier.

## Reproduce

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m fraud_graph.profile --input dataset/data.csv --output outputs/data_profile.json
    python -m fraud_graph.eda --input dataset/data.csv --output-dir outputs
    python -m fraud_graph.rings --synthetic-only --synthetic-output outputs/ring_recovery_synthetic.json
    python -m fraud_graph.experiment --input dataset/data.csv --output-dir outputs
    python -m fraud_graph.rings --input dataset/data.csv --output outputs/ring_candidates.json --progress

Profiling, EDA, and group scans stream the CSV by chunks/windows. Transaction feature construction and XGBoost fitting use the full dataset in memory. The experiment writes transaction metrics, saved test predictions, feature importance, and a plot. It uses chronological step splits (train through 520, validation 521–631, test from 632). Validation controls XGBoost early stopping; test is reserved for final metrics. Exact top-k alert budgets use a stable tie-break and report how many rows a threshold-inclusive implementation would select.

## Method and limitations

- Model A uses transaction type, amount, step, and pre-transaction endpoint balances. A balance ablation removes both balance fields. Model B adds strictly prior source/destination behavior. Model C adds strictly prior graph degree, distinct-neighbor, and repeat-intensity features. All models exclude post-transaction balances, `isFlaggedFraud`, and raw account identifiers.
- Logistic Regression is the conventional baseline. XGBoost uses `hist`, training-derived positive-class weighting, and validation-based early stopping for A, the balance ablation, B, and C. Reported transaction metrics are average precision and precision/recall at exact fixed alert budgets.
- Candidate groups are proposed independently within non-overlapping windows from weak components, strongly connected components, bounded fan-in/fan-out motifs, and Charikar-style greedy dense-subgraph peeling. Candidates require at least three accounts in a directed cycle or at least two temporally ordered relays. A fixed evidence score ranks them; it is not a calibrated probability.
- Ring discovery never reads transaction labels or transaction-model predictions. PaySim groups remain unverified candidates because the data has no ring IDs. Ring metrics come only from a controlled planted-group benchmark with one-to-one member-overlap matching.
- `python -m fraud_graph.groups` remains a compatibility entry point to the label-free workflow. The former all-history, fraud-label-ranked output is legacy and must not be treated as prospective group discovery.

For a bounded feasibility run, add `--window-limit 1`; the output marks the scan as incomplete. A full-timeline group scan takes several minutes on the project environment. The full transaction experiment takes longer and records its runtime and peak RSS.

