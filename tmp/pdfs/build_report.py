from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

metrics = json.loads((ROOT / "outputs/metrics.json").read_text())
profile = json.loads((ROOT / "outputs/data_profile.json").read_text())
rings = json.loads((ROOT / "outputs/ring_candidates.json").read_text())
synthetic = json.loads((ROOT / "outputs/ring_recovery_synthetic.json").read_text())

models = metrics["models"]
names = [
    ("Logistic Regression", "logistic_A_pretransaction"),
    ("XGBoost A: transaction", "xgb_A_pretransaction"),
    ("XGBoost A: no balances", "xgb_A_no_balance_ablation"),
    ("XGBoost B: + behavior", "xgb_B_behavior"),
    ("XGBoost C: + graph", "xgb_C_graph"),
]

# Figure 1: test and validation AP make the model and ablation comparisons visible.
fig, ax = plt.subplots(figsize=(9.2, 4.25))
y = np.arange(len(names))
height = 0.34
test_ap = [models[key]["average_precision"] for _, key in names]
val_ap = [models[key].get("validation_average_precision", np.nan) for _, key in names]
ax.barh(y + height / 2, test_ap, height, color="#176B87", label="Test AP")
ax.barh(y - height / 2, np.nan_to_num(val_ap), height, color="#64A6BD", label="Validation AP")
for yi, (tv, vv) in enumerate(zip(test_ap, val_ap)):
    ax.text(tv + .012, yi + height / 2, f"{tv:.3f}", va="center", fontsize=9)
    if np.isfinite(vv):
        ax.text(vv + .012, yi - height / 2, f"{vv:.3f}", va="center", fontsize=9)
ax.set_yticks(y, [label for label, _ in names], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.1)
ax.set_xlabel("Average precision")
ax.set_title("Transaction model comparison and feature ablations", loc="left", weight="bold")
ax.grid(axis="x", alpha=.2)
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "classification_ablation.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Figure 2: operational precision/recall under exact, stable top-k alert budgets.
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
budgets = ["0.1%", "0.5%", "1.0%"]
labels = ["90 alerts", "448 alerts", "895 alerts"]
plot_models = [
    ("XGB A", "xgb_A_pretransaction", "#176B87"),
    ("A, no balances", "xgb_A_no_balance_ablation", "#D1495B"),
    ("XGB B", "xgb_B_behavior", "#59A14F"),
    ("XGB C", "xgb_C_graph", "#E39C37"),
]
for label, key, color in plot_models:
    for ax, metric, title in zip(axes, ["precision", "recall"], ["Precision at alert budget", "Recall at alert budget"]):
        values = [models[key]["fixed_alert_budgets"][budget][metric] for budget in budgets]
        ax.plot(range(3), values, marker="o", lw=2, ms=5, label=label, color=color)
        ax.set_title(title, weight="bold")
        ax.set_ylim(-.03, 1.08)
        ax.set_xticks(range(3), labels)
        ax.grid(alpha=.22)
axes[0].set_ylabel("Fraction")
handles, legend_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, legend_labels, frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, .90))
fig.suptitle("Exact top-k test alerts (stable tie-break)", x=.06, y=.995, ha="left", weight="bold")
fig.tight_layout(rect=[0, 0, 1, .84])
fig.savefig(FIG / "alert_budget_results.png", dpi=220, bbox_inches="tight")
plt.close(fig)

shutil.copy2(ROOT / "outputs/eda_overview.png", FIG / "eda_overview.png")

# Derive benchmark means and standard deviations from the saved seed-level outputs.
rank_names = ["interpretable_group_score", "candidate_transaction_count_baseline", "candidate_density_baseline"]
benchmark = {}
for rank_name in rank_names:
    benchmark[rank_name] = {}
    for metric_name in ["average_precision", "precision_at_30", "member_recovery_at_30", "ring_recall_at_30"]:
        values = [run["rankings"][rank_name][metric_name] for run in synthetic["runs"]]
        benchmark[rank_name][metric_name] = (float(np.mean(values)), float(np.std(values, ddof=1)))

def esc(value: str) -> str:
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        value = value.replace(old, new)
    return value

# Editable LaTeX source is supplied alongside the PDF. It uses the same sections,
# tables, and figures as the rendered report.
tex = r'''\documentclass[11pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{graphicx,booktabs,tabularx,array,xcolor,hyperref,caption}
\usepackage[T1]{fontenc}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Fraud Transaction Detection and Coordinated-Group Discovery\\\large PaySim Offline ASBD Project}
\author{Analytics and Systems of Big Data}
\date{October 2026}
\begin{document}\maketitle
\section*{Executive summary}
The project evaluates transaction-level fraud classification and a separate, label-free workflow for discovering candidate coordinated groups. XGBoost attained test average precision (AP) of 0.971 with pre-transaction balances, but a balance ablation fell to 0.326. The result is strongly influenced by a PaySim-specific signature. The full graph scan produced no groups meeting the declared coordination criteria. PaySim has transaction labels, not verified ring labels; ring recovery is therefore reported only on a controlled planted-group benchmark.
\section*{Data and experimental setup}
PaySim contains 6,362,620 transactions, 8,213 fraud-labelled rows (0.1291\%), and 743 simulation steps. The audit found no missing values and no exact duplicates in a 64-bit row-hash scan. Fraud labels occur only on CASH\_OUT (4,116) and TRANSFER (4,097). The chronological split is train through step 520, validation 521--631, and test from 632; test prevalence is 1.399\% versus 0.095\% in training. Predictors exclude post-transaction balances, account IDs, and isFlaggedFraud. Behavioral and graph aggregates use only steps earlier than the scored transaction's step.
\begin{figure}[h]\centering\includegraphics[width=.96\linewidth]{figures/eda_overview.png}\caption{Dataset overview. Amounts and account activity are sampled as described in the accompanying EDA summary.}\end{figure}
\section*{Feature construction and evaluation protocol}
The transaction task predicts the PaySim label $y_t\in\{0,1\}$. Account histories are derived only from strictly earlier simulation steps. Same-step transactions do not update one another because no within-step ordering is provided. Account IDs are used to join histories but are not classifier inputs; labels are excluded from aggregates.
\begin{table}[h]\centering\small\begin{tabularx}{\linewidth}{p{1.2in}p{2.6in}X}\toprule Feature set & Inputs & Purpose\\\midrule
A & Type, step, amount, pre-event source/destination balances & Transaction-only reference.\\
A without balances & Type, step, amount & Ablation for the simulator balance signature.\\
B & A plus prior endpoint counts, amount summaries, and time since last activity & Adds historical account behavior.\\
C & B plus prior distinct counterparties and repeat intensity & Tests simple graph neighborhood features.\\\bottomrule\end{tabularx}\caption{Feature sets and modeling purpose. Post-event balances, raw account IDs, and target-derived aggregates are excluded.}\end{table}
For a transaction at step $s$, behavioral snapshots contain events only through $s-1$. This avoids using the current transaction or assuming an order among events at the same step.
\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule Split & Rows & Fraud rows & Fraud rate\\\midrule
Train ($s\leq520$) & 6,082,007 & 5,781 & 0.095\%\\
Validation ($521\leq s\leq631$) & 191,147 & 1,180 & 0.617\%\\
Test ($s\geq632$) & 89,466 & 1,252 & 1.399\%\\\bottomrule\end{tabular}\caption{Chronological evaluation split.}\end{table}
XGBoost uses histogram trees, learning rate 0.08, depth 6, minimum child weight 10, row and column subsampling 0.8, and L2 regularization 5. Training-only positive-class weight is 1,051.07. Validation AP controls early stopping (30 rounds; maximum 600 trees); the test set is reserved for final metrics. Logistic Regression uses class weighting. The rule baseline ranks transfer and cash-out transactions by log amount.
Average precision summarizes precision-recall ranking and is more informative than accuracy under severe imbalance. Operational results use exact alert budgets of 90, 448, and 895 test transactions; stable row order breaks tied scores.
\clearpage
\section*{Transaction classification}
Model A uses transaction type, amount, step, and pre-transaction balances; a separate ablation removes balances. B adds prior account behavior. C adds prior graph degree, distinct-counterparty, and repeat-intensity features. Models use a common chronological split. XGBoost uses histogram trees, training-derived class weighting, validation AP for early stopping, and a 600-tree cap. Alert metrics select exactly the top $k$ test rows with a stable tie-break.
\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule Model & Validation AP & Test AP & P/R at 90 alerts\\\midrule
Logistic Regression & 0.386 & 0.517 & 1.000 / 0.072\\
XGBoost A & 0.951 & 0.971 & 1.000 / 0.072\\
XGBoost A, no balances & 0.249 & 0.326 & 1.000 / 0.072\\
XGBoost B, behavior & 0.911 & 0.938 & 1.000 / 0.072\\
XGBoost C, graph & 0.920 & 0.945 & 1.000 / 0.072\\\bottomrule\end{tabular}\caption{Transaction results. AP is average precision. At 90 alerts, recall is limited to 90 of 1,252 test frauds.}\end{table}
\begin{figure}[h]\centering\includegraphics[width=.94\linewidth]{figures/classification_ablation.png}\caption{Validation and test AP by model and feature set.}\end{figure}
\textbf{Ablation finding.} 8,034/8,213 fraud rows (97.82\%) satisfy amount = oldbalanceOrg, compared with 0/6,354,407 non-fraud rows. These are pre-event fields, so this is not post-event leakage, but it is a simulator-specific shortcut. AP decreases from 0.971 (A) to 0.326 without balances. B reduces AP; C raises B by only 0.007 on test. A reaches the 600-tree cap and is not confirmed converged.
\clearpage
\section*{Operating points and interpretation}
\begin{table}[h]\centering\small\begin{tabular}{lrrr}\toprule Model & P/R at 90 & P/R at 448 & P/R at 895\\\midrule
XGBoost A & 1.000 / 0.072 & 1.000 / 0.358 & 0.990 / 0.708\\
A, no balances & 1.000 / 0.072 & 0.616 / 0.220 & 0.434 / 0.310\\
XGBoost B & 1.000 / 0.072 & 1.000 / 0.358 & 0.984 / 0.704\\
XGBoost C & 1.000 / 0.072 & 1.000 / 0.358 & 0.989 / 0.707\\\bottomrule\end{tabular}\caption{Precision and recall at exact test alert counts.}\end{table}
\begin{figure}[h]\centering\includegraphics[width=.94\linewidth]{figures/alert_budget_results.png}\caption{Precision and recall under exact test alert budgets for XGBoost variants.}\end{figure}
At 90 alerts, each XGBoost variant retrieves the same 90 frauds and no false positives, so recall is only 7.19\%. At 448 alerts, A, B, and C each recover 448 frauds; the balance-free model recovers 276. At 895 alerts, A, B, and C remain close. Graph features improve B by only 0.0074 test AP and do not surpass transaction-only A.

At the 90-alert score cutoff, a threshold-inclusive rule would select 93 A rows, 214 no-balance rows, 199 B rows, and 165 C rows. Metrics instead use an explicit stable top-$k$ tie-break to honor the review capacity. A real deployment would need an explicit secondary ranking or queue rule for tied scores.
\clearpage
\section*{Candidate-group discovery}
The separate group miner does not read fraud labels or classifier scores. It uses non-overlapping 24-step windows and proposes weak components, strongly connected components, fan-in/fan-out motifs, and Charikar-style dense subsets. A candidate must contain at least three accounts in a directed cycle or two temporally ordered relay accounts across multiple active steps. A fixed evidence score combines structure, transaction timing, repeated pairs, amount similarity, and flow continuity; it is a ranking heuristic, not a probability.

The complete scan covered all 31 windows and returned zero eligible PaySim groups. This means none passed the chosen screen, not that fraud rings are absent. Without verified ring IDs, PaySim ring precision/recall cannot be calculated. On a controlled benchmark of 30 planted groups and 90 decoy episodes per seed (five seeds), the group score obtained mean AP 0.241 (SD 0.040), precision@30 0.447 (0.069), member recovery@30 0.357 (0.055), and ring recall@30 0.447 (0.069). Matching uses one-to-one account-set IoU >= 0.5. This synthetic benchmark does not validate real-world performance.
\begin{table}[h]\centering\small\begin{tabularx}{\linewidth}{p{1.35in}X}\toprule Stage & Evidence and decision\\\midrule
Proposal & Weak components bound membership; SCCs, fan-in/fan-out, and dense peeling add overlapping subgraph proposals.\\
Coordination evidence & Ordered inflow then outflow, activity span, repeated pairs, flow continuity, and amount similarity.\\
Eligibility & At least three accounts in a directed cycle, or two temporally ordered relays across multiple active steps.\\
Ranking & Fixed weighted heuristic for review priority; not a calibrated probability and not trained on PaySim labels.\\\bottomrule\end{tabularx}\caption{Candidate group proposal, eligibility, and ranking are separate stages.}\end{table}
Strict criteria may miss longer or cross-window groups, weakly connected rings, and partially observed behavior. Windows are non-overlapping; dense-subgraph peeling is limited to components of at most 20 accounts. These bounds limit computational work but create false-negative risks.
\section*{Conclusions and limits}
The strongest transaction result is inseparable from a synthetic balance signature; balance-free AP is materially lower. Behavioral and graph features do not outperform transaction-only XGBoost on AP. The group workflow is a candidate miner, not a validated fraud-ring classifier, and this run yielded no eligible PaySim groups. The controlled experiment gives a reproducible but modest ring-recovery check. Additional time cutoffs, noisy/overlapping planted groups, larger-scale sensitivity checks, and independently labeled real data are needed before making generalization claims. Streaming remains future work.

\textbf{Reproducibility.} Commands are in README.md. Metrics, saved test predictions, data profile, EDA, full group scan, and five-seed benchmark are in outputs/.
\end{document}
'''
(OUT / "ASBD_Fraud_Ring_Report.tex").write_text(tex)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.HexColor("#153B50"), alignment=TA_LEFT, spaceAfter=6))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=11, leading=14, textColor=colors.HexColor("#4A6472"), spaceAfter=14))
styles.add(ParagraphStyle(name="SectionX", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#153B50"), spaceBefore=8, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.7, leading=13.2, textColor=colors.HexColor("#24343B"), spaceAfter=5))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#394D56"), spaceAfter=3))
styles.add(ParagraphStyle(name="CaptionX", parent=styles["BodyText"], fontSize=8.3, leading=10.5, textColor=colors.HexColor("#52656D"), alignment=TA_LEFT, spaceBefore=2, spaceAfter=7))
styles.add(ParagraphStyle(name="CalloutX", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#153B50"), backColor=colors.HexColor("#EAF3F5"), borderColor=colors.HexColor("#A9CCD5"), borderWidth=.7, borderPadding=8, spaceBefore=5, spaceAfter=8))

class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self.decorate)])
    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D5E0E4"))
            canvas.line(doc.leftMargin, letter[1] - .48 * inch, letter[0] - doc.rightMargin, letter[1] - .48 * inch)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#58717C"))
            canvas.drawString(doc.leftMargin, letter[1] - .38 * inch, "ASBD | Fraud Transaction and Coordinated-Group Analysis")
        canvas.setStrokeColor(colors.HexColor("#D5E0E4"))
        canvas.line(doc.leftMargin, .48 * inch, letter[0] - doc.rightMargin, .48 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#58717C"))
        canvas.drawString(doc.leftMargin, .32 * inch, "PaySim offline prototype | October 2026")
        canvas.drawRightString(letter[0] - doc.rightMargin, .32 * inch, str(doc.page))
        canvas.restoreState()

doc = ReportDoc(str(OUT / "ASBD_Fraud_Ring_Report.pdf"), pagesize=letter,
                leftMargin=.68*inch, rightMargin=.68*inch, topMargin=.66*inch, bottomMargin=.62*inch,
                title="Fraud Transaction Detection and Coordinated-Group Discovery",
                author="Analytics and Systems of Big Data")
story = []
P = lambda text, style="BodyX": Paragraph(text, styles[style])
story += [Spacer(1, .1*inch), P("Fraud Transaction Detection<br/>and Coordinated-Group Discovery", "ReportTitle"),
          P("PaySim offline prototype | Analytics and Systems of Big Data | October 2026", "Subtitle"),
          HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1A7892"), spaceAfter=9),
          P("Executive summary", "SectionX"),
          P("This project evaluates two distinct tasks: transaction-level fraud classification and label-free discovery of candidate coordinated groups. XGBoost reached <b>0.971 test average precision</b> with pre-transaction balances, but the balance-free ablation fell to <b>0.326</b>. The high result reflects a strong PaySim-specific signature. The full graph scan returned <b>zero groups</b> meeting the declared coordination criteria. PaySim has transaction labels, not verified ring labels, so no PaySim ring-level performance is claimed."),
          P("Dataset and split", "SectionX")]

data_rows = [
    ["Transactions", "6,362,620", "Fraud labels", "8,213 (0.1291%)"],
    ["Steps", "1-743", "Nulls / duplicates", "0 / 0*"],
    ["Train", "step <= 520", "Validation", "521-631"],
    ["Test", "step >= 632", "Test fraud prevalence", "1.399%"],
]
tbl = Table(data_rows, colWidths=[1.35*inch, 1.55*inch, 1.55*inch, 2.0*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF3F5")),
    ("BACKGROUND", (0,1), (0,-1), colors.HexColor("#F3F7F8")),
    ("BACKGROUND", (2,1), (2,-1), colors.HexColor("#F3F7F8")),
    ("FONTNAME", (0,0), (-1,-1), "Helvetica"), ("FONTSIZE", (0,0), (-1,-1), 8.8),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
    ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#24343B")),
    ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5E0E4")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
]))
story += [tbl, Spacer(1, 5), P("No missing fields were found; exact duplicates were checked by 64-bit row hash. Fraud labels occur only on CASH_OUT (4,116) and TRANSFER (4,097). Test prevalence is 14.7x training prevalence (0.095%), indicating substantial temporal shift. *A theoretical hash collision is possible.", "SmallX"),
          Image(str(FIG / "eda_overview.png"), width=6.95*inch, height=3.75*inch),
          P("Figure 1. Transaction mix, fraud counts, log-scale amount sample, and activity over simulation steps. The account activity panel is based on a deterministic sample.", "CaptionX"),
          PageBreak()]

story += [P("1. Feature construction and evaluation protocol", "SectionX"),
          P("The transaction task predicts the PaySim label for each event. Account histories use only transactions from strictly earlier simulation steps. Records at the same step do not update one another because PaySim provides no within-step event order. Account IDs are used to construct histories but excluded from classifier inputs; fraud labels are never aggregated into account features.", "BodyX")]

feature_rows = [
    ["Set", "Predictors", "Purpose"],
    ["A", "Type, step, amount, pre-event source/destination balances", "Transaction-only reference"],
    ["A without balances", "Type, step, amount", "Tests balance-signature dependence"],
    ["B", "A + prior endpoint counts, amount summaries, last-activity gap", "Adds account behavior"],
    ["C", "B + prior distinct counterparties and repeat intensity", "Adds simple graph neighborhoods"],
]
ft = Table([[str(cell) if ri == 0 else P(str(cell), "SmallX") for cell in row] for ri, row in enumerate(feature_rows)], colWidths=[1.2*inch, 3.7*inch, 2.1*inch], repeatRows=1)
ft.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#153B50")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,-1),8.5), ("VALIGN",(0,0),(-1,-1),"TOP"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")),
    ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
]))
story += [ft, Spacer(1, 7),
          P("For a transaction at step s, account snapshots end at step s-1. The current event and all other step-s events are unavailable to its history features. This gives a prospective offline feature definition without inventing within-step order.", "BodyX"),
          P("Chronological evaluation", "SectionX")]

split_rows = [["Split", "Rows", "Fraud", "Rate", "Purpose"],
              ["Train through 520", "6,082,007", "5,781", "0.095%", "Fit models and class weights"],
              ["Validation 521-631", "191,147", "1,180", "0.617%", "Early stopping / selection"],
              ["Test from 632", "89,466", "1,252", "1.399%", "Final evaluation"]]
st = Table(split_rows, colWidths=[1.45*inch, 1.05*inch, .7*inch, .75*inch, 3.05*inch], repeatRows=1)
st.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#153B50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.1),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
]))
story += [st, Spacer(1, 7),
          P("Test fraud prevalence is 14.7 times the training prevalence. The chronological split preserves detection direction, but the shift means test metrics describe this final PaySim period rather than a stationary deployment population.", "BodyX"),
          P("Model fitting and metrics", "SectionX"),
          P("Logistic Regression uses class weighting. XGBoost uses histogram trees, learning rate 0.08, depth 6, minimum child weight 10, row and column subsampling 0.8, and L2 regularization 5. The training-only positive-class weight is 1,051.07. Validation average precision controls early stopping (30 rounds; 600-tree maximum); test metrics are computed after model selection. The transparent rule baseline ranks transfer and cash-out events by log amount.", "BodyX"),
          P("Average precision summarizes precision-recall ranking and is more useful than accuracy under severe imbalance; an all-negative classifier would have high accuracy. Operational results use exact top-k budgets of 90, 448, and 895 test alerts. Stable row order resolves score ties, and inclusive-cutoff overflow is recorded as a diagnostic.", "BodyX"),
          P("The transaction-only model A reached the 600-tree ceiling rather than early-stopping; treat its score as the tested-cap result, not a confirmed optimum. The no-balance ablation stopped at two rounds; B and C stopped at 126 and 312.", "CalloutX"),
          PageBreak()]

story += [P("2. Transaction fraud classification", "SectionX"),
          P("Model A uses transaction type, amount, step, and pre-transaction origin/destination balances. The no-balance ablation removes both balance fields. Model B adds strictly prior-step endpoint behavior; Model C adds prior graph degree, distinct-counterparty counts, and repeat intensity. Post-transaction balances, `isFlaggedFraud`, raw account IDs, and target-derived features are excluded. Logistic Regression is the conventional baseline; XGBoost uses histogram trees, training-derived class weighting, validation AP for early stopping, and a 600-tree cap."),
          Image(str(FIG / "classification_ablation.png"), width=6.95*inch, height=3.22*inch),
          P("Figure 2. Validation and test average precision by model and feature set. The holdout is chronological; test ranking is evaluated once after validation-based model selection.", "CaptionX")]

table_data = [["Model", "Val AP", "Test AP", "P / R @ 90"],
              ["Rule: transfer/cash-out amount", "-", ".218", ".789 / .057"],
              ["Logistic Regression A", ".386", ".517", "1.000 / .072"],
              ["XGBoost A", ".951", ".971", "1.000 / .072"],
              ["A, no balances", ".249", ".326", "1.000 / .072"],
              ["XGBoost B, behavior", ".911", ".938", "1.000 / .072"],
              ["XGBoost C, graph", ".920", ".945", "1.000 / .072"]]
tbl = Table(table_data, colWidths=[3.25*inch, .85*inch, .85*inch, 1.5*inch], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#153B50")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8.7), ("ALIGN",(1,1),(-1,-1),"CENTER"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
]))
story += [tbl, Spacer(1, 5),
          P("Key observation: 8,034/8,213 fraud rows (97.82%) have amount equal to oldbalanceOrg, versus 0/6,354,407 non-fraud rows. The fields are available before the transaction, so this is not post-event leakage; it is a simulator-specific signal. AP falls sharply when balances are removed. Behavior reduces AP from A; graph features recover only about +0.007 AP over B. XGBoost A reached the 600-tree cap and is not confirmed converged.", "CalloutX"),
          P("At 90 alerts, perfect precision covers only 90 of 1,252 test frauds (7.19% recall). Fixed-budget metrics use stable top-k selection; a score threshold alone may exceed the budget when scores tie.", "BodyX"),
          PageBreak()]

story += [P("3. Operating points and interpretation", "SectionX"),
          P("Average precision summarizes ranking across thresholds. The alert-budget view answers the operational question: how do precision and recall change when investigators can review only a fixed number of alerts? Labels are used here only for retrospective evaluation.", "BodyX")]

budget_table = [["XGBoost variant", "90 alerts P / R", "448 alerts P / R", "895 alerts P / R"]]
for display, key in [("A: transaction", "xgb_A_pretransaction"), ("A: no balances", "xgb_A_no_balance_ablation"), ("B: behavior", "xgb_B_behavior"), ("C: graph", "xgb_C_graph")]:
    cells = [display]
    for budget in ["0.1%", "0.5%", "1.0%"]:
        val = models[key]["fixed_alert_budgets"][budget]
        cells.append(f"{val['precision']:.3f} / {val['recall']:.3f}")
    budget_table.append(cells)
bt = Table(budget_table, colWidths=[1.7*inch, 1.65*inch, 1.65*inch, 1.65*inch], repeatRows=1)
bt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#153B50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,-1),8.2),("ALIGN",(1,1),(-1,-1),"CENTER"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")),
    ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
]))
story += [bt, Spacer(1, 8), Image(str(FIG / "alert_budget_results.png"), width=6.95*inch, height=2.85*inch),
          P("Figure 3. Precision and recall at exact 0.1%, 0.5%, and 1.0% test alert budgets.", "CaptionX"),
          P("Reading the operating points", "SectionX"),
          P("At 90 alerts, all XGBoost variants retrieve the same 90 frauds and no false positives, giving 7.19% recall. At 448 alerts, A, B, and C each recover 448 frauds; the no-balance model recovers 276. At 895 alerts, A, B, and C remain close: precision ranges from 0.984 to 0.990 and recall from 0.704 to 0.708. The graph addition improves B by only 0.0074 test AP and does not surpass A.", "BodyX"),
          P("Scores are tied near several cutoffs. At 90 alerts, an inclusive threshold would select 93 A rows, 214 no-balance rows, 199 B rows, and 165 C rows. The reported results select exactly k rows using a stable tie-break. Operational use would need a secondary ranking or queue policy for ties.", "BodyX"),
          PageBreak()]

story += [P("4. Coordinated-group discovery", "SectionX"),
          P("The group miner is independent of transaction labels and model predictions. It scans non-overlapping 24-step windows and proposes weak components, strongly connected components, fan-in/fan-out motifs, and Charikar-style dense subsets (bounded to small components). Candidates require at least three accounts in a directed cycle or at least two accounts with temporally ordered relay activity across multiple steps. A fixed heuristic ranks structural density, cycles, temporal coordination, repeated pairs, flow continuity, and amount similarity; it is not a calibrated probability."),
          P(f"Full PaySim result: <b>{rings['processed_windows']} windows scanned; {rings['candidate_count']} eligible groups.</b> The earlier permissive screen returned chain-like components that failed the stricter coordination rule. This negative scan result does not show that PaySim contains no rings. The dataset has no independently verified ring IDs, so ring precision and recall on PaySim are unavailable.", "CalloutX"),
          P("Controlled ring-recovery check", "SectionX")]

bench_rows = [["Ranking", "AP (mean +/- SD)", "Precision@30", "Member recovery@30", "Ring recall@30"],
              ["Evidence score", ".241 +/- .040", ".447 +/- .069", ".357 +/- .055", ".447 +/- .069"],
              ["Transaction-count baseline", ".201 +/- .004", "0.000", "0.000", "0.000"],
              ["Density baseline", ".097 +/- .000", "0.000", "0.000", "0.000"]]
tbl = Table(bench_rows, colWidths=[1.75*inch, 1.2*inch, 1.1*inch, 1.25*inch, 1.15*inch], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#153B50")), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 7.7), ("ALIGN",(1,1),(-1,-1),"CENTER"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
]))
story += [tbl, Spacer(1, 5),
          P("Benchmark: five seeds, each with 30 planted cyclic or relay groups and 90 decoy episodes. Matching is one-to-one at account-set IoU >= 0.5. Mean candidate coverage was 1.00, but top-30 recovery was modest. The synthetic background is easier than real transaction networks and does not validate PaySim ring accuracy.", "SmallX"),
          P("How candidate groups are formed", "SectionX")]

group_rows = [["Stage", "Evidence / rule"],
              ["Proposal", "Weak components bound membership; SCCs, fan-in/fan-out, and dense peeling add subgraph proposals."],
              ["Coordination", "Directed cycles, ordered inflow then outflow, temporal compactness, repeated pairs, amount similarity, and flow continuity."],
              ["Eligibility", "At least 3 accounts in a directed cycle, or at least 2 temporally ordered relays over multiple active steps."],
              ["Ranking", "Fixed weighted score for review priority; not a probability and not trained on PaySim fraud labels."]]
gt = Table([[str(c) if ri == 0 else P(str(c), "SmallX") for c in row] for ri, row in enumerate(group_rows)], colWidths=[1.35*inch, 5.65*inch], repeatRows=1)
gt.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#153B50")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
    ("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F7F8")]),
    ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D5E0E4")),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
]))
story += [gt, Spacer(1, 7),
          P("The strict screen favors coordination over mere connectivity, but can miss longer or cross-window rings, weakly connected groups, and partial activity. Windows do not overlap, and dense-subgraph peeling is limited to components of at most 20 accounts. These bounds control runtime but create false-negative risk.", "BodyX"),
          P("Conclusions and next work", "SectionX"),
          P("The strongest fraud-classification result is heavily tied to a PaySim balance signature; the balance-free result is substantially weaker. Behavioral and graph features do not improve AP over transaction-only XGBoost. No verified fraud ring was identified in PaySim. The controlled experiment supports continued development of interpretable group mining, but stronger noisy, overlapping, and cross-window injections plus sensitivity analysis are needed. Additional chronological cutoffs and independently labeled real data are needed for generalization claims. Streaming is deferred until offline ring recovery is validated.", "BodyX"),
          P("Reproduction: commands are in README.md; metrics and ring outputs are in outputs/. Runtime / peak RSS: classifier 398 s / 7.3 GiB; group scan 276 s / 734 MiB.", "SmallX")]

doc.build(story)
print(OUT / "ASBD_Fraud_Ring_Report.pdf")
