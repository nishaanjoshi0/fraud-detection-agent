from __future__ import annotations

from pathlib import Path

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_EDA_DIR = OUT_DIR / "eda"

# Matplotlib/SHAP may attempt to use unwritable user cache directories.
# Force it to write under our project outputs folder.
os.environ.setdefault("MPLBACKEND", "Agg")
MPLCONFIGDIR = OUT_EDA_DIR / ".mplconfig"
XDG_CACHE_HOME = OUT_EDA_DIR / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))


pythonFEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

CATEGORICAL_COLS = ["card4", "card6"] + [f"M{i}" for i in range(1, 10)]


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
    """
    Remove rows with ProductCD == 'S' AND isFraud == 1 before splitting.
    Store these removed rows separately as novel_fraud_holdout.
    """
    holdout_mask = (df["ProductCD"] == "S") & (df["isFraud"] == 1)
    novel_fraud_holdout = df.loc[holdout_mask].copy()
    remaining = df.loc[~holdout_mask].copy()
    return remaining, novel_fraud_holdout


def preprocess_with_label_encoders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    novel_holdout_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns:
    - Fit encoders on training only
    - Transform train/test/novel_holdout with the same encoders
    - Fill missing values with -999 after encoding
    """
    encoders_dict: dict[str, LabelEncoder] = {}

    # Work on copies to keep caller data intact.
    X_train = train_df[pythonFEATURES].copy()
    X_test = test_df[pythonFEATURES].copy()
    X_novel = novel_holdout_df[pythonFEATURES].copy()

    unk_marker = "__UNK__"
    missing_marker = "__MISSING__"

    # Convert and encode categoricals.
    for col in CATEGORICAL_COLS:
        # Fill missing + cast to string to get stable LabelEncoder behavior.
        X_train[col] = X_train[col].fillna(missing_marker).astype(str)
        X_test[col] = X_test[col].fillna(missing_marker).astype(str)
        X_novel[col] = X_novel[col].fillna(missing_marker).astype(str)

        enc = LabelEncoder()
        enc.fit(X_train[col])

        # Avoid transform failures if test/novel contains unseen categories.
        if unk_marker not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, unk_marker)

        def encode_with_unk(series: pd.Series) -> np.ndarray:
            # Map unseen labels to unk_marker so transform never errors.
            series = series.where(series.isin(enc.classes_), unk_marker)
            return enc.transform(series)

        X_train[col] = encode_with_unk(X_train[col])
        X_test[col] = encode_with_unk(X_test[col])
        X_novel[col] = encode_with_unk(X_novel[col])

        encoders_dict[col] = enc

    # Fill remaining missing values for numeric columns (and any leftovers).
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    X_novel = X_novel.fillna(-999)

    return X_train, X_test, X_novel, encoders_dict


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_EDA_DIR.mkdir(parents=True, exist_ok=True)

    from lightgbm import LGBMClassifier, early_stopping

    # -------- Data loading / merging --------
    df = load_and_merge()

    # -------- Novel fraud holdout --------
    remaining, novel_fraud_holdout = apply_novel_fraud_holdout(df)

    X = remaining[pythonFEATURES].copy()
    y = remaining["isFraud"].astype(int).copy()

    # -------- Train/test split --------
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Restore full rows for preprocessing helper (it expects dataframes with all columns).
    # (X_train_df/X_test_df contain only FEATURES, so we join back indices from original remaining.)
    # Using indices avoids a costly merge on large dataframes.
    train_idx = X_train_df.index
    test_idx = X_test_df.index

    train_df = remaining.loc[train_idx].copy()
    test_df = remaining.loc[test_idx].copy()

    # -------- Preprocessing --------
    X_train, X_test, X_novel, encoders_dict = preprocess_with_label_encoders(
        train_df=train_df,
        test_df=test_df,
        novel_holdout_df=novel_fraud_holdout,
    )

    # -------- Training --------
    pythonparams = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = LGBMClassifier(**pythonparams)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, first_metric_only=True)],
    )

    # -------- Evaluation (test set) --------
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report (test):")
    print(classification_report(y_test, y_test_pred))
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"AUC-ROC (test): {test_auc:.6f}")

    # -------- Evaluation (novel holdout) --------
    # Note: novel_fraud_holdout is the subset with isFraud==1 (single-class),
    # so ROC AUC may be undefined. We compute it against test-set negatives if needed.
    if len(novel_fraud_holdout) == 0:
        print("Novel fraud holdout is empty; skipping novel holdout evaluation.")
        novel_auc = float("nan")
        novel_mean_pred = float("nan")
    else:
        y_novel = novel_fraud_holdout["isFraud"].astype(int).to_numpy()
        y_novel_proba = model.predict_proba(X_novel)[:, 1]
        novel_mean_pred = float(np.mean(y_novel_proba))

        unique_classes = np.unique(y_novel)
        if unique_classes.size >= 2:
            novel_auc = roc_auc_score(y_novel, y_novel_proba)
        else:
            # Build an evaluation set: novel positives + all test negatives (for a valid AUC).
            test_neg_mask = y_test.to_numpy() == 0
            if np.any(test_neg_mask):
                y_eval = np.concatenate([y_novel, np.zeros(np.sum(test_neg_mask), dtype=int)])
                proba_eval = np.concatenate([y_novel_proba, y_test_proba[test_neg_mask]])
                novel_auc = roc_auc_score(y_eval, proba_eval)
            else:
                novel_auc = float("nan")

    print(f"AUC-ROC (novel fraud holdout): {novel_auc:.6f}")

    # -------- SHAP --------
    print("Computing SHAP values (sample 2,000 test rows)...")
    n_shap = min(2000, len(X_test))
    shap_sample_idx = np.random.RandomState(42).choice(len(X_test), size=n_shap, replace=False)
    X_shap = X_test.iloc[shap_sample_idx]

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        # For binary classification, shap returns per-class at times; use class-1.
        shap_values = shap_values[1]

    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    # Save the current matplotlib figure produced by shap.summary_plot.
    import matplotlib.pyplot as plt  # local import to keep top-level minimal

    plt.tight_layout()
    plt.savefig(OUT_EDA_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -------- Save artifacts --------
    print("Saving model and preprocessing artifacts...")
    joblib.dump(model, OUT_DIR / "lgbm_model.pkl")

    joblib.dump(
        {"features": pythonFEATURES, "encoders": encoders_dict},
        OUT_DIR / "preprocessor.pkl",
    )
    joblib.dump(novel_fraud_holdout, OUT_DIR / "novel_fraud_holdout.pkl")

    print("Done.")


if __name__ == "__main__":
    main()