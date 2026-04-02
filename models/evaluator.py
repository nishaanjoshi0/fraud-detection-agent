from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_EDA_DIR = OUT_DIR / "eda"
os.environ.setdefault("MPLBACKEND", "Agg")
_MPL = OUT_EDA_DIR / ".mplconfig"
_CACHE = OUT_EDA_DIR / ".cache"
_MPL.mkdir(parents=True, exist_ok=True)
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = ROOT / "data"

THRESHOLD = 0.5
AVG_TXN_DOLLARS = 200.0


def load_and_merge() -> pd.DataFrame:
    train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv", low_memory=False)
    train_identity = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)
    return pd.merge(
        train_transaction,
        train_identity,
        on="TransactionID",
        how="left",
    )


def apply_novel_fraud_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_mask = (df["ProductCD"] == "S") & (df["isFraud"] == 1)
    novel_fraud_holdout = df.loc[holdout_mask].copy()
    remaining = df.loc[~holdout_mask].copy()
    return remaining, novel_fraud_holdout


def preprocess_with_loaded_encoders(
    df: pd.DataFrame,
    features: list[str],
    encoders_dict: dict[str, LabelEncoder],
) -> pd.DataFrame:
    unk_marker = "__UNK__"
    missing_marker = "__MISSING__"
    X = df[features].copy()
    for col, enc in encoders_dict.items():
        if unk_marker not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, unk_marker)
        s = X[col].fillna(missing_marker).astype(str)
        s = s.where(s.isin(enc.classes_), unk_marker)
        X[col] = enc.transform(s)
    return X.fillna(-999)


def min_max_anomaly_score(raw: np.ndarray, min_raw: float, max_raw: float) -> np.ndarray:
    denom = max_raw - min_raw
    if denom == 0:
        normalized = np.zeros_like(raw, dtype=float)
    else:
        normalized = (raw - min_raw) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    return 1.0 - normalized


def _compute_score_frame(
    X: pd.DataFrame,
    lgbm_model,
    iso_model,
    scaler: dict[str, float],
) -> pd.DataFrame:
    """Internal: scores from preprocessed feature matrix X."""
    min_raw = float(scaler["min"])
    max_raw = float(scaler["max"])
    supervised_score = lgbm_model.predict_proba(X)[:, 1]
    raw = iso_model.decision_function(X)
    anomaly_score = min_max_anomaly_score(np.asarray(raw), min_raw, max_raw)
    combined_score = np.maximum(supervised_score, anomaly_score)
    return pd.DataFrame(
        {
            "supervised_score": supervised_score,
            "anomaly_score": anomaly_score,
            "combined_score": combined_score,
        },
        index=X.index,
    )


def binary_metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = THRESHOLD
) -> tuple[float, float, float, float]:
    y_pred = (scores >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, scores)
    return prec, rec, f1, auc


def novel_holdout_metrics(
    novel_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float = THRESHOLD,
) -> tuple[float, float, int]:
    """Recall on novel (all fraud), AUC vs test negatives, count caught at threshold."""
    y_novel = np.ones(len(novel_scores), dtype=int)
    caught = int(np.sum(novel_scores >= threshold))
    recall = caught / len(novel_scores) if len(novel_scores) else 0.0

    test_neg_mask = y_test == 0
    if len(novel_scores) == 0 or not np.any(test_neg_mask):
        auc = float("nan")
    else:
        y_eval = np.concatenate([y_novel, np.zeros(np.sum(test_neg_mask), dtype=int)])
        s_eval = np.concatenate([novel_scores, test_scores[test_neg_mask]])
        auc = roc_auc_score(y_eval, s_eval)

    return recall, auc, caught


def dollar_impact(y_true: np.ndarray, scores: np.ndarray, threshold: float = THRESHOLD):
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return {
        "fraud_caught_dollars": tp * AVG_TXN_DOLLARS,
        "fraud_missed_dollars": fn * AVG_TXN_DOLLARS,
        "false_positives": fp,
    }


def print_table1(rows: list[dict]) -> None:
    print("Table 1 — Full test set:")
    print(
        f"{'Detection Mode':<22} | {'Precision':>9} | {'Recall':>6} | {'F1':>5} | {'AUC':>6}"
    )
    print("-" * 62)
    for r in rows:
        print(
            f"{r['mode']:<22} | {r['precision']:>9.4f} | {r['recall']:>6.4f} | "
            f"{r['f1']:>5.4f} | {r['auc']:>6.4f}"
        )
    print()


def print_table2(rows: list[dict]) -> None:
    print("Table 2 — Novel fraud holdout only:")
    print(
        f"{'Detection Mode':<22} | {'Recall':>6} | {'AUC':>6} | {'Transactions caught':>20}"
    )
    print("-" * 62)
    for r in rows:
        auc_s = f"{r['auc']:.4f}" if np.isfinite(r["auc"]) else "nan"
        print(
            f"{r['mode']:<22} | {r['recall']:>6.4f} | {auc_s:>6} | "
            f"{r['transactions_caught']:>20}"
        )
    print()


def print_table3(rows: list[dict]) -> None:
    print("Table 3 — Dollar impact (full test set, avg transaction = $200):")
    print(
        f"{'Detection Mode':<22} | {'Fraud caught ($)':>16} | "
        f"{'Fraud missed ($)':>16} | {'False positives':>15}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['mode']:<22} | {r['fraud_caught_dollars']:>16,.0f} | "
            f"{r['fraud_missed_dollars']:>16,.0f} | {r['false_positives']:>15}"
        )
    print()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "preprocessor": OUT_DIR / "preprocessor.pkl",
        "lgbm": OUT_DIR / "lgbm_model.pkl",
        "iso": OUT_DIR / "iso_forest_model.pkl",
        "scaler": OUT_DIR / "anomaly_scaler.pkl",
        "novel": OUT_DIR / "novel_fraud_holdout.pkl",
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} ({name}). Run supervised.py and anomaly.py first.")

    preprocessor = joblib.load(paths["preprocessor"])
    features: list[str] = preprocessor["features"]
    encoders_dict: dict[str, LabelEncoder] = preprocessor["encoders"]
    lgbm_model = joblib.load(paths["lgbm"])
    iso_model = joblib.load(paths["iso"])
    scaler = joblib.load(paths["scaler"])
    novel_fraud_holdout = joblib.load(paths["novel"])

    def score_transactions(X: pd.DataFrame) -> pd.DataFrame:
        """Return supervised, anomaly (scaled), and max-combined scores for preprocessed X."""
        return _compute_score_frame(X, lgbm_model, iso_model, scaler)

    df = load_and_merge()
    remaining, _ = apply_novel_fraud_holdout(df)

    X = remaining[features].copy()
    y = remaining["isFraud"].astype(int).copy()

    _X_train_df, X_test_df, _y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    test_idx = X_test_df.index
    test_df = remaining.loc[test_idx].copy()
    novel_df = novel_fraud_holdout.copy()

    X_test = preprocess_with_loaded_encoders(test_df, features, encoders_dict)
    X_test = X_test.loc[y_test.index]
    X_novel = preprocess_with_loaded_encoders(novel_df, features, encoders_dict)

    y_test_arr = y_test.to_numpy()
    y_novel_arr = novel_df["isFraud"].astype(int).to_numpy()

    scores_test = score_transactions(X_test)
    scores_novel = score_transactions(X_novel)

    modes = [
        ("Supervised only", "supervised_score"),
        ("Anomaly only", "anomaly_score"),
        ("Combined (max score)", "combined_score"),
    ]

    # ----- Table 1: full test -----
    table1_rows = []
    for mode_name, col in modes:
        s = scores_test[col].to_numpy()
        p, r, f1, auc = binary_metrics_at_threshold(y_test_arr, s)
        table1_rows.append(
            {
                "mode": mode_name,
                "precision": p,
                "recall": r,
                "f1": f1,
                "auc": auc,
            }
        )

    # ----- Table 2: novel holdout -----
    table2_rows = []
    for mode_name, col in modes:
        s_n = scores_novel[col].to_numpy()
        s_t = scores_test[col].to_numpy()
        rec, auc_n, caught = novel_holdout_metrics(s_n, s_t, y_test_arr)
        table2_rows.append(
            {
                "mode": mode_name,
                "recall": rec,
                "auc": auc_n,
                "transactions_caught": caught,
            }
        )

    # ----- Table 3: dollar impact (full test) -----
    table3_rows = []
    for mode_name, col in modes:
        s = scores_test[col].to_numpy()
        d = dollar_impact(y_test_arr, s)
        table3_rows.append({"mode": mode_name, **d})

    print_table1(table1_rows)
    print_table2(table2_rows)
    print_table3(table3_rows)

    # ----- Save CSVs -----
    pd.DataFrame(table1_rows).to_csv(OUT_DIR / "detection_comparison.csv", index=False)
    pd.DataFrame(table2_rows).to_csv(OUT_DIR / "novel_fraud_comparison.csv", index=False)
    pd.DataFrame(table3_rows).to_csv(OUT_DIR / "dollar_impact.csv", index=False)

    # ----- Partner narrative (3 sentences) -----
    # Dollar delta: combined vs supervised-only fraud caught ($) on full test set
    sup_caught = table3_rows[0]["fraud_caught_dollars"]
    comb_caught = table3_rows[2]["fraud_caught_dollars"]
    extra_recovered = max(0.0, comb_caught - sup_caught)

    narrative = (
        f"At a 0.5 alert threshold on the held-out test set, layering anomaly scores on top of the supervised "
        f"model surfaces about ${extra_recovered:,.0f} more in fraud-tagged exposure (at $200 per transaction) "
        f"than supervised scoring alone, by catching additional confirmed fraud without waiting for a label. "
        f"Supervised learning remains strongest where historical fraud patterns repeat, while the combined "
        f"score improves recall on subtle or shifting behavior the classifier under-weights. "
        f"Novel fraud typologies still benefit from analyst review: use model alerts to prioritize queues, not "
        f"to eliminate human judgment on emerging schemes."
    )
    print(narrative)


if __name__ == "__main__":
    main()
