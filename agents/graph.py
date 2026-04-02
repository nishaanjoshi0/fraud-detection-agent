from __future__ import annotations

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"
OUT_EDA_DIR = OUT_DIR / "eda"
os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCFG = OUT_EDA_DIR / ".mplconfig"
_MPLCACHE = OUT_EDA_DIR / ".cache"
_MPLCFG.mkdir(parents=True, exist_ok=True)
_MPLCACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCFG))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPLCACHE))

preprocessor = joblib.load(OUT_DIR / "preprocessor.pkl")
FEATURES: list[str] = preprocessor["features"]
ENCODERS = preprocessor["encoders"]
lgbm_model = joblib.load(OUT_DIR / "lgbm_model.pkl")
iso_model = joblib.load(OUT_DIR / "iso_forest_model.pkl")
scaler = joblib.load(OUT_DIR / "anomaly_scaler.pkl")

import shap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --- Training data (same holdout + split as supervised / anomaly) ---

_train_tx = pd.read_csv(DATA_DIR / "train_transaction.csv", low_memory=False)
_train_id = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)
_merged = pd.merge(_train_tx, _train_id, on="TransactionID", how="left")
_holdout_mask = (_merged["ProductCD"] == "S") & (_merged["isFraud"] == 1)
_remaining = _merged.loc[~_holdout_mask].copy()
_X = _remaining[FEATURES].copy()
_y = _remaining["isFraud"].astype(int).copy()
_X_train_df, _, _, _ = train_test_split(
    _X,
    _y,
    test_size=0.2,
    random_state=42,
    stratify=_y,
)
_train_idx = _X_train_df.index
TRAIN_DF = _remaining.loc[_train_idx].copy().reset_index(drop=True)
TRAIN_LEGIT_DF = TRAIN_DF[TRAIN_DF["isFraud"] == 0].copy().reset_index(drop=True)


def _min_max_anomaly_score(raw: np.ndarray, min_raw: float, max_raw: float) -> np.ndarray:
    denom = max_raw - min_raw
    if denom == 0:
        normalized = np.zeros_like(raw, dtype=float)
    else:
        normalized = (raw - min_raw) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    return 1.0 - normalized


def _encode_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    unk_marker = "__UNK__"
    missing_marker = "__MISSING__"
    X = X[FEATURES].copy()
    for col, enc in ENCODERS.items():
        if unk_marker not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, unk_marker)
        s = X[col].fillna(missing_marker).astype(str)
        s = s.where(s.isin(enc.classes_), unk_marker)
        X[col] = enc.transform(s)
    return X.fillna(-999)


def preprocess_single(transaction: dict) -> np.ndarray:
    """Single-row feature matrix as numpy (1, n_features) in FEATURES order."""
    row = {f: transaction.get(f, np.nan) for f in FEATURES}
    df = pd.DataFrame([row])
    X = _encode_dataframe(df)
    return X.to_numpy(dtype=np.float64)


# Preprocessed training matrix for cosine similarity (aligned with TRAIN_DF rows)
TRAIN_X_MAT = _encode_dataframe(TRAIN_DF).to_numpy(dtype=np.float64)


class FraudState(TypedDict):
    transaction: dict
    supervised_score: float
    anomaly_score: float
    combined_score: float
    investigation: dict
    explanation: str
    decision: str


def scoring_node(state: FraudState) -> dict:
    X_arr = preprocess_single(state["transaction"])
    X_df = pd.DataFrame(X_arr, columns=FEATURES)
    supervised_score = float(lgbm_model.predict_proba(X_df)[0, 1])
    raw = iso_model.decision_function(X_df)[0]
    min_raw = float(scaler["min"])
    max_raw = float(scaler["max"])
    anomaly_score = float(_min_max_anomaly_score(np.array([raw]), min_raw, max_raw)[0])
    combined_score = max(supervised_score, anomaly_score)
    out: dict = {
        "supervised_score": supervised_score,
        "anomaly_score": anomaly_score,
        "combined_score": combined_score,
    }
    if combined_score < 0.3:
        out["decision"] = "APPROVE"
        out["explanation"] = "Transaction score below risk threshold. Auto-approved."
    return out


def route_after_scoring(state: FraudState) -> str:
    if state["combined_score"] < 0.3:
        return "approve"
    return "investigate"


def investigation_node(state: FraudState) -> dict:
    tx = state["transaction"]
    X_arr = preprocess_single(tx)
    X_df = pd.DataFrame(X_arr, columns=FEATURES)
    q = X_arr.flatten()

    card1 = tx.get("card1", np.nan)
    if pd.isna(card1):
        velocity = 0
        amount_deviation = 0.0
    else:
        same_card = TRAIN_DF["card1"] == card1
        velocity = int(same_card.sum())
        grp_amt = TRAIN_DF.loc[same_card, "TransactionAmt"]
        amt = tx.get("TransactionAmt", np.nan)
        if len(grp_amt) == 0 or pd.isna(amt):
            amount_deviation = 0.0
        else:
            mu = float(grp_amt.mean())
            sigma = float(grp_amt.std())
            if sigma == 0.0 or np.isnan(sigma):
                amount_deviation = 0.0
            else:
                amount_deviation = float((float(amt) - mu) / sigma)

    explainer = shap.TreeExplainer(lgbm_model)
    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        sv = sv[1]
    sv_row = np.asarray(sv).reshape(-1)
    order = np.argsort(-np.abs(sv_row))[:5]
    top_shap_features = []
    for i in order:
        name = FEATURES[i]
        val = float(sv_row[i])
        direction = "increases_risk" if val >= 0 else "decreases_risk"
        top_shap_features.append(
            {"feature": name, "shap_value": val, "direction": direction}
        )

    sup = state["supervised_score"]
    ano = state["anomaly_score"]
    pattern_type = "known_pattern" if sup > ano else "novel_anomaly"

    sims = cosine_similarity(q.reshape(1, -1), TRAIN_X_MAT)[0]
    top_idx = np.argsort(-sims)[:5]
    similar_cases = []
    for j in top_idx:
        r = TRAIN_DF.iloc[int(j)]
        similar_cases.append(
            {
                "TransactionAmt": float(r["TransactionAmt"])
                if pd.notna(r["TransactionAmt"])
                else None,
                "ProductCD": r["ProductCD"] if pd.notna(r["ProductCD"]) else None,
                "isFraud": int(r["isFraud"]),
            }
        )

    investigation = {
        "velocity": {
            "Number of prior transactions on this card in training data.": velocity,
        },
        "amount_deviation": amount_deviation,
        "top_shap_features": top_shap_features,
        "pattern_type": pattern_type,
        "similar_cases": similar_cases,
    }
    return {"investigation": investigation}


def explanation_node(state: FraudState) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tx = state["transaction"]
    key_fields = {
        "TransactionAmt": tx.get("TransactionAmt"),
        "ProductCD": tx.get("ProductCD"),
        "card4": tx.get("card4"),
        "card6": tx.get("card6"),
        "addr1": tx.get("addr1"),
        "dist1": tx.get("dist1"),
    }
    inv_json = json.dumps(state["investigation"], indent=2, default=str)
    user_msg = (
        f"Transaction (key fields): {json.dumps(key_fields, default=str)}\n\n"
        f"supervised_score: {state['supervised_score']:.6f}\n"
        f"anomaly_score: {state['anomaly_score']:.6f}\n"
        f"combined_score: {state['combined_score']:.6f}\n\n"
        f"Investigation (JSON):\n{inv_json}"
    )
    system_prompt = """You are a fraud analytics assistant helping a financial institution 
evaluate a flagged transaction. You receive transaction data, model 
scores, and behavioral context. Generate a concise partner-facing 
explanation covering:
1. Why this transaction was flagged
2. Whether it matches a known fraud pattern or appears to be a novel anomaly
3. Which signals most strongly drove the decision
4. Recommended action: APPROVE, DECLINE, or ESCALATE TO HUMAN REVIEW
5. Estimated dollar risk

Be direct and analytical. Write for a fraud operations team, not a 
technical audience. Keep under 150 words. End your response with a 
line that says exactly: DECISION: APPROVE or DECISION: DECLINE or 
DECISION: ESCALATE"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        text = f"Explanation generation failed: {exc}"
        return {"explanation": text, "decision": "ESCALATE"}

    decision = "ESCALATE"
    for line in reversed(text.splitlines()):
        m = re.search(
            r"DECISION:\s*(APPROVE|DECLINE|ESCALATE)\b",
            line.strip(),
            re.IGNORECASE,
        )
        if m:
            decision = m.group(1).upper()
            break
    return {"explanation": text, "decision": decision}


def build_graph():
    graph = StateGraph(FraudState)
    graph.add_node("scoring", scoring_node)
    graph.add_node("investigation", investigation_node)
    graph.add_node("explanation", explanation_node)

    graph.set_entry_point("scoring")
    graph.add_conditional_edges(
        "scoring",
        route_after_scoring,
        {
            "approve": END,
            "investigate": "investigation",
        },
    )
    graph.add_edge("investigation", "explanation")
    graph.add_edge("explanation", END)

    return graph.compile()


agent_app = build_graph()


if __name__ == "__main__":
    test_transaction = {
        "TransactionAmt": 847.50,
        "ProductCD": "W",
        "card1": 4657,
        "card2": 320.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 299.0,
        "addr2": 87.0,
        "dist1": 0.0,
        "dist2": None,
        "C1": 1.0,
        "C2": 1.0,
        "C3": 0.0,
        "C4": 0.0,
        "C5": 0.0,
        "C6": 1.0,
        "C7": 0.0,
        "C8": 0.0,
        "C9": 1.0,
        "C10": 0.0,
        "C11": 1.0,
        "C12": 0.0,
        "C13": 1.0,
        "C14": 1.0,
        "D1": 0.0,
        "D2": None,
        "D3": None,
        "D4": None,
        "D5": None,
        "D6": None,
        "D7": None,
        "D8": None,
        "D9": None,
        "D10": 0.0,
        "D11": None,
        "D12": None,
        "D13": None,
        "D14": None,
        "D15": None,
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M0",
        "M5": "F",
        "M6": "T",
        "M7": None,
        "M8": None,
        "M9": None,
    }

    initial_state: FraudState = {
        "transaction": test_transaction,
        "supervised_score": 0.0,
        "anomaly_score": 0.0,
        "combined_score": 0.0,
        "investigation": {},
        "explanation": "",
        "decision": "",
    }

    print("Running agent pipeline...")
    result = agent_app.invoke(initial_state)
    print(f"\nSupervised score: {result['supervised_score']:.4f}")
    print(f"Anomaly score:    {result['anomaly_score']:.4f}")
    print(f"Combined score:   {result['combined_score']:.4f}")
    print(f"Pattern type:     {result['investigation'].get('pattern_type', 'N/A')}")
    print(f"Decision:         {result['decision']}")
    print(f"\nExplanation:\n{result['explanation']}")
