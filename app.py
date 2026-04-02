from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

st.set_page_config(
    page_title="Fraud Detection Agent",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_agent_runtime():
    """Load LangGraph agent once per process (avoids re-import churn on reruns)."""
    from agents import graph as ag

    return ag.agent_app, ag.FEATURES


@st.cache_data(show_spinner=False)
def load_flagged_transactions():
    """
    Full test set, scored with same preprocessing/models as agents/graph.
    Returns all rows with combined_score > 0.3 (no sampling).
    """
    from agents import graph as g

    DATA_DIR = g.ROOT / "data"
    train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv", low_memory=False)
    train_identity = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)
    merged = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    holdout_mask = (merged["ProductCD"] == "S") & (merged["isFraud"] == 1)
    remaining = merged.loc[~holdout_mask].copy()

    features = g.FEATURES
    X = remaining[features].copy()
    y = remaining["isFraud"].astype(int).copy()
    _, X_test_df, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    test_idx = X_test_df.index
    test_rows = remaining.loc[test_idx].copy()

    X_enc = g._encode_dataframe(test_rows[features])

    sup = g.lgbm_model.predict_proba(X_enc)[:, 1]
    raw = g.iso_model.decision_function(X_enc)
    min_raw = float(g.scaler["min"])
    max_raw = float(g.scaler["max"])
    ano = g._min_max_anomaly_score(np.asarray(raw), min_raw, max_raw)
    combined = np.maximum(sup, ano)
    pattern = np.where(sup > ano, "known_pattern", "novel_anomaly")

    out = test_rows.copy()
    out["supervised_score"] = sup
    out["anomaly_score"] = ano
    out["combined_score"] = combined
    out["pattern_type"] = pattern

    flagged = out[out["combined_score"] > 0.3].reset_index(drop=True)
    return flagged


def row_to_transaction_dict(row: pd.Series, features: list[str]) -> dict:
    tx: dict = {}
    for f in features:
        if f not in row.index:
            tx[f] = np.nan
        else:
            v = row[f]
            tx[f] = np.nan if pd.isna(v) else v
    return tx


def _dataframe_selected_rows(event) -> list[int]:
    """Extract row indices from st.dataframe(..., on_select='rerun') return value."""
    try:
        if event is not None:
            sel = event["selection"] if isinstance(event, dict) else event.selection
            if isinstance(sel, dict):
                r = list(sel.get("rows", []))
                if r:
                    return r
            else:
                r = list(getattr(sel, "rows", []) or [])
                if r:
                    return r
    except (KeyError, TypeError, AttributeError):
        pass
    ss = st.session_state.get("flagged_tx_table")
    if isinstance(ss, dict) and "selection" in ss:
        return list(ss["selection"].get("rows", []))
    return []


def make_risk_gauge(combined_score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(combined_score),
            number={"font": {"size": 28}, "valueformat": ".3f"},
            title={"text": "Combined risk"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "#1f2937"},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 0.3], "color": "#22c55e", "thickness": 0.85},
                    {"range": [0.3, 0.6], "color": "#eab308", "thickness": 0.85},
                    {"range": [0.6, 1.0], "color": "#ef4444", "thickness": 0.85},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": float(combined_score),
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def badge_html(text: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span style="background-color:{bg};color:{fg};padding:6px 12px;'
        f'border-radius:6px;font-weight:600;display:inline-block;">{text}</span>'
    )


# --- Main UI ---
tab1, tab2 = st.tabs(["Transaction Evaluator", "Model Performance"])

with tab1:
    with st.spinner("Loading and scoring test transactions... this may take a minute"):
        flagged_df = load_flagged_transactions()

    agent_app, FEATURES = get_agent_runtime()

    known_n = int((flagged_df["pattern_type"] == "known_pattern").sum())
    novel_n = int((flagged_df["pattern_type"] == "novel_anomaly").sum())

    display_cols = [
        "TransactionID",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "supervised_score",
        "anomaly_score",
        "combined_score",
        "pattern_type",
        "isFraud",
    ]
    disp = flagged_df[display_cols].copy()
    for c in ("supervised_score", "anomaly_score", "combined_score"):
        disp[c] = disp[c].round(4)

    col_left, col_right = st.columns([0.4, 0.6])

    with col_left:
        st.subheader("Flagged transactions")
        st.caption(
            f"Total flagged: **{len(flagged_df)}** transactions | "
            f"Known patterns: **{known_n}** | Novel anomalies: **{novel_n}**"
        )
        event = st.dataframe(
            disp,
            on_select="rerun",
            selection_mode="single-row",
            key="flagged_tx_table",
            hide_index=True,
            width="stretch",
        )

    with col_right:
        st.subheader("Agent output")
        selected_rows = _dataframe_selected_rows(event)

        if not selected_rows:
            st.info("← Select a transaction from the table to run the agent")
        else:
            row_idx = int(selected_rows[0])
            full_row = flagged_df.iloc[row_idx]
            tx_dict = row_to_transaction_dict(full_row, FEATURES)

            initial_state = {
                "transaction": tx_dict,
                "supervised_score": 0.0,
                "anomaly_score": 0.0,
                "combined_score": 0.0,
                "investigation": {},
                "explanation": "",
                "decision": "",
            }

            with st.spinner("Running agent pipeline..."):
                result = agent_app.invoke(initial_state)

            c1, c2, c3 = st.columns(3)
            c1.metric("Supervised score", f"{result['supervised_score']:.4f}")
            c2.metric("Anomaly score", f"{result['anomaly_score']:.4f}")
            c3.metric("Combined score", f"{result['combined_score']:.4f}")

            st.plotly_chart(
                make_risk_gauge(result["combined_score"]),
                use_container_width=True,
            )

            inv = result.get("investigation") or {}
            ptype = inv.get("pattern_type", "unknown")
            if ptype == "known_pattern":
                st.markdown(
                    badge_html("Known Fraud Pattern", "#b91c1c"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    badge_html("Novel Anomaly", "#ea580c"),
                    unsafe_allow_html=True,
                )

            st.markdown("")
            dec = result.get("decision", "ESCALATE")
            if dec == "APPROVE":
                st.markdown(badge_html("APPROVE", "#16a34a"), unsafe_allow_html=True)
            elif dec == "DECLINE":
                st.markdown(badge_html("DECLINE", "#dc2626"), unsafe_allow_html=True)
            else:
                st.markdown(badge_html("ESCALATE", "#ca8a04", "#111"), unsafe_allow_html=True)

            st.subheader("Agent Explanation")
            st.info(result.get("explanation") or "")

            top_shap = inv.get("top_shap_features") or []
            if top_shap:
                shap_df = pd.DataFrame(
                    [
                        {
                            "Feature": x.get("feature"),
                            "SHAP Value": round(float(x.get("shap_value", 0.0)), 6),
                            "Direction": x.get("direction", ""),
                        }
                        for x in top_shap
                    ]
                )
                st.subheader("Top SHAP drivers")
                st.dataframe(shap_df, hide_index=True, use_container_width=True)

            vel = inv.get("velocity") or {}
            vel_count = next(iter(vel.values()), 0) if isinstance(vel, dict) else 0
            amt_dev = float(inv.get("amount_deviation", 0.0))

            st.subheader("Behavioral context")
            bc1, bc2 = st.columns(2)
            bc1.metric("Card velocity (prior txns in training)", f"{vel_count}")
            bc2.metric("Amount deviation (σ from card mean)", f"{amt_dev:.2f}")

            sim = inv.get("similar_cases") or []
            if sim:
                st.subheader("Similar cases (training)")
                st.dataframe(pd.DataFrame(sim), hide_index=True, use_container_width=True)

with tab2:
    st.title("Model Performance")

    det_path = OUT_DIR / "detection_comparison.csv"
    novel_path = OUT_DIR / "novel_fraud_comparison.csv"
    dollar_path = OUT_DIR / "dollar_impact.csv"

    if not det_path.exists():
        st.warning("Run `models/evaluator.py` to generate comparison CSVs in outputs/.")
    else:
        det_df = pd.read_csv(det_path)
        st.subheader("Full Test Set Performance")
        num_cols = [c for c in ("precision", "recall", "f1", "auc") if c in det_df.columns]
        fmt = {c: "{:.4f}" for c in num_cols}
        st.dataframe(
            det_df.style.format(fmt, na_rep="—"),
            hide_index=True,
            width="stretch",
        )

    if novel_path.exists():
        novel_df = pd.read_csv(novel_path)
        st.subheader("Novel Fraud Holdout Performance")
        nfmt = {
            c: "{:.4f}"
            for c in ("recall", "auc")
            if c in novel_df.columns
        }
        st.dataframe(
            novel_df.style.format(nfmt, na_rep="—"),
            hide_index=True,
            width="stretch",
        )
        sup_auc = novel_df.loc[
            novel_df["mode"] == "Supervised only", "auc"
        ].values
        comb_auc = novel_df.loc[
            novel_df["mode"] == "Combined (max score)", "auc"
        ].values
        s_auc = float(sup_auc[0]) if len(sup_auc) else float("nan")
        c_auc = float(comb_auc[0]) if len(comb_auc) else float("nan")
        st.info(
            f"The supervised model scores AUC **{s_auc:.2f}** on novel fraud — worse than random. "
            f"The anomaly layer (combined) recovers this to AUC **{c_auc:.2f}**."
        )

    if dollar_path.exists():
        dollar_df = pd.read_csv(dollar_path)
        st.subheader("Dollar Impact by Detection Mode")
        dmelt = dollar_df.melt(
            id_vars=["mode"],
            value_vars=["fraud_caught_dollars", "fraud_missed_dollars"],
            var_name="metric",
            value_name="dollars",
        )
        dmelt["metric"] = dmelt["metric"].map(
            {
                "fraud_caught_dollars": "Fraud caught ($)",
                "fraud_missed_dollars": "Fraud missed ($)",
            }
        )
        fig_d = px.bar(
            dmelt,
            x="mode",
            y="dollars",
            color="metric",
            barmode="group",
            color_discrete_map={
                "Fraud caught ($)": "#22c55e",
                "Fraud missed ($)": "#ef4444",
            },
            title="Dollar Impact by Detection Mode",
        )
        fig_d.update_layout(xaxis_title="Detection Mode", yaxis_title="Dollar amount")
        st.plotly_chart(fig_d, use_container_width=True)

        st.subheader("False positive comparison")
        fig_fp = px.bar(
            dollar_df,
            x="mode",
            y="false_positives",
            title="False positives by detection mode",
            color_discrete_sequence=["#6366f1"],
        )
        fig_fp.update_layout(xaxis_title="Detection Mode", yaxis_title="Count")
        st.plotly_chart(fig_fp, use_container_width=True)
        st.caption(
            "Higher false positives = more legitimate transactions incorrectly flagged. "
            "There is always a tradeoff between catching more fraud and generating more false alarms."
        )
