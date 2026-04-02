from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_and_merge() -> pd.DataFrame:
    train_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv", low_memory=False)
    train_identity = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)
    df = pd.merge(
        train_transaction,
        train_identity,
        on="TransactionID",
        how="left",
    )
    return df


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
    """
    Apply the exact same preprocessing scheme as supervised.py:
    - For each categorical column in encoders_dict: label-encode using fitted encoders (no refit)
    - Fill missing categorical values with '__MISSING__' before encoding
    - Map unseen categories to '__UNK__' so transform never errors
    - Fill all remaining missing values with -999
    """
    unk_marker = "__UNK__"
    missing_marker = "__MISSING__"

    X = df[features].copy()

    for col, enc in encoders_dict.items():
        # Ensure we can always encode unseen values.
        if unk_marker not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, unk_marker)

        s = X[col].fillna(missing_marker).astype(str)
        s = s.where(s.isin(enc.classes_), unk_marker)
        X[col] = enc.transform(s)

    X = X.fillna(-999)
    return X


def min_max_anomaly_score(raw: np.ndarray, min_raw: float, max_raw: float) -> np.ndarray:
    denom = (max_raw - min_raw)
    if denom == 0:
        normalized = np.zeros_like(raw, dtype=float)
    else:
        normalized = (raw - min_raw) / denom
    normalized = np.clip(normalized, 0.0, 1.0)
    # Invert so: 1.0 = most anomalous, 0.0 = most normal.
    return 1.0 - normalized


def main() -> None:
    out_model_path = OUT_DIR / "iso_forest_model.pkl"
    out_scaler_path = OUT_DIR / "anomaly_scaler.pkl"

    preprocessor_path = OUT_DIR / "preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Missing {preprocessor_path}. Run models/supervised.py first."
        )

    preprocessor = joblib.load(preprocessor_path)
    features: list[str] = preprocessor["features"]
    encoders_dict: dict[str, LabelEncoder] = preprocessor["encoders"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Data loading / merging --------
    df = load_and_merge()

    # -------- Novel fraud holdout --------
    remaining, novel_fraud_holdout = apply_novel_fraud_holdout(df)

    X = remaining[features].copy()
    y = remaining["isFraud"].astype(int).copy()

    # -------- Train/test split --------
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Restore full-row slices for preprocessing (to access all needed columns consistently).
    train_idx = X_train_df.index
    test_idx = X_test_df.index
    train_df = remaining.loc[train_idx].copy()
    test_df = remaining.loc[test_idx].copy()

    # -------- Preprocessing (no refit) --------
    X_train = preprocess_with_loaded_encoders(train_df, features, encoders_dict)
    X_test = preprocess_with_loaded_encoders(test_df, features, encoders_dict)
    X_novel = preprocess_with_loaded_encoders(
        novel_fraud_holdout, features, encoders_dict
    )

    # Ensure row order matches y_train/y_test order for downstream boolean indexing.
    X_train = X_train.loc[y_train.index]
    X_test = X_test.loc[y_test.index]

    # -------- Training Isolation Forest (legitimate only) --------
    # Use the training split labels to select legitimate transactions.
    legit_train_idx = y_train[y_train == 0].index
    X_legit_train = X_train.loc[legit_train_idx]

    pythonIsolationForest = IsolationForest(
        n_estimators=200,
        contamination=0.035,
        random_state=42,
        n_jobs=-1,
    )
    pythonIsolationForest.fit(X_legit_train)

    # -------- Scoring / anomaly normalization --------
    pythonraw_train_legit = pythonIsolationForest.decision_function(X_legit_train)
    min_raw = float(np.min(pythonraw_train_legit))
    max_raw = float(np.max(pythonraw_train_legit))

    pythonraw_test = pythonIsolationForest.decision_function(X_test)
    test_anomaly_scores = min_max_anomaly_score(
        pythonraw_test, min_raw, max_raw
    )

    pythonraw_novel = pythonIsolationForest.decision_function(X_novel)
    novel_anomaly_scores = min_max_anomaly_score(
        pythonraw_novel, min_raw, max_raw
    )

    # -------- Evaluation --------
    print("Isolation Forest evaluation")
    test_auc = roc_auc_score(y_test, test_anomaly_scores)
    print(f"AUC-ROC (test): {test_auc:.6f}")

    if len(novel_fraud_holdout) == 0:
        print("Novel fraud holdout is empty; skipping novel holdout AUC.")
        novel_auc = float("nan")
        novel_detection_rate = float("nan")
    else:
        y_novel = novel_fraud_holdout["isFraud"].astype(int).to_numpy()

        unique_classes = np.unique(y_novel)
        if unique_classes.size >= 2:
            novel_auc = roc_auc_score(y_novel, novel_anomaly_scores)
        else:
            # Holdout specified as only isFraud==1 => single class.
            # For a valid ROC AUC, pair novel positives with test negatives.
            test_neg_mask = y_test.to_numpy() == 0
            y_eval = np.concatenate(
                [y_novel, np.zeros(np.sum(test_neg_mask), dtype=int)]
            )
            proba_eval = np.concatenate(
                [novel_anomaly_scores, test_anomaly_scores[test_neg_mask]]
            )
            novel_auc = roc_auc_score(y_eval, proba_eval)

        novel_detection_rate = float(np.mean(novel_anomaly_scores > 0.5) * 100.0)

    print(f"AUC-ROC (novel fraud holdout): {novel_auc:.6f}")
    print(
        "Novel fraud detection rate "
        f"(% with anomaly_score > 0.5): {novel_detection_rate:.2f}%"
    )

    # -------- Save artifacts --------
    print("Saving Isolation Forest + scaler...")
    joblib.dump(pythonIsolationForest, out_model_path)
    joblib.dump({"min": min_raw, "max": max_raw}, out_scaler_path)
    print("Done.")


if __name__ == "__main__":
    main()