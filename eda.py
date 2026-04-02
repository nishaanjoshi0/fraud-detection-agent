"""
Exploratory data analysis for fraud detection.
Loads IEEE-CIS-style train_transaction + train_identity, merges, plots, prints summary.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "eda"
MPLCONFIGDIR = OUT_DIR / ".mplconfig"
XDG_CACHE_HOME = OUT_DIR / ".cache"
# Matplotlib sometimes defaults to ~/.matplotlib which may be unwritable.
# Setting MPLCONFIGDIR early avoids hard crashes.
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs (low_memory=False)...")
    train_transaction = pd.read_csv(
        DATA_DIR / "train_transaction.csv", low_memory=False
    )
    train_identity = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)

    df = pd.merge(
        train_transaction,
        train_identity,
        on="TransactionID",
        how="left",
        suffixes=("", "_id"),
    )

    n = len(df)
    fraud_rate = df["isFraud"].mean() * 100
    print(f"Merged dataframe shape: {df.shape}")
    print(f"Overall fraud rate: {fraud_rate:.4f}%")
    print()

    # --- Plot 1: class distribution ---
    print("Generating plot 1/6: class distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["steelblue", "coral"])
    ax.set_title("Transaction class distribution")
    ax.set_ylabel("Count")
    total = counts.sum()
    for bar, c in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        pct = 100.0 * c / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: amount distribution (log x), sample 50k per class ---
    print("Generating plot 2/6: amount distribution...")
    legit = df[df["isFraud"] == 0]
    fraud_df = df[df["isFraud"] == 1]
    n_leg = min(50_000, len(legit))
    n_fr = min(50_000, len(fraud_df))
    samp_leg = legit.sample(n=n_leg, random_state=42)["TransactionAmt"]
    samp_fr = fraud_df.sample(n=n_fr, random_state=42)["TransactionAmt"]
    # positive amounts only for log scale
    samp_leg = samp_leg[samp_leg > 0]
    samp_fr = samp_fr[samp_fr > 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        samp_leg,
        bins=50,
        alpha=0.55,
        label="Legitimate",
        color="steelblue",
        density=True,
    )
    ax.hist(
        samp_fr,
        bins=50,
        alpha=0.55,
        label="Fraud",
        color="coral",
        density=True,
    )
    ax.set_xscale("log")
    ax.set_title("Transaction amount distribution by class (sampled, density)")
    ax.set_xlabel("TransactionAmt (log scale)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "amount_distribution.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: fraud rate by ProductCD ---
    print("Generating plot 3/6: fraud by ProductCD...")
    prod = (
        df.groupby("ProductCD", dropna=False)
        .agg(fraud=("isFraud", "sum"), n=("isFraud", "count"))
        .assign(rate=lambda x: 100.0 * x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(prod.index.astype(str), prod["rate"], color="teal")
    ax.set_title("Fraud rate (%) by ProductCD")
    ax.set_xlabel("ProductCD")
    ax.set_ylabel("Fraud rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraud_by_productcd.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: fraud rate by card4 ---
    print("Generating plot 4/6: fraud by card type...")
    card = (
        df.groupby("card4", dropna=False)
        .agg(fraud=("isFraud", "sum"), n=("isFraud", "count"))
        .assign(rate=lambda x: 100.0 * x["fraud"] / x["n"])
        .sort_values("rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    # Use numeric x positions to avoid matplotlib category-axis type issues.
    x = np.arange(len(card))
    ax.bar(x, card["rate"].to_numpy(dtype=float), color="slateblue")
    ax.set_title("Fraud rate (%) by card4")
    ax.set_xlabel("card4")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in card.index.tolist()], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fraud_by_cardtype.png", dpi=150)
    plt.close(fig)

    # --- Plot 5: missing values top 20 (column-level) ---
    print("Generating plot 5/6: missing values...")
    miss_pct = df.isna().mean().sort_values(ascending=False).head(20) * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(miss_pct))
    ax.barh(y_pos, miss_pct.values, color="gray")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(miss_pct.index)
    ax.invert_yaxis()
    ax.set_title("Top 20 features by missing value %")
    ax.set_xlabel("Missing (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    # --- Plot 6: velocity by hour (sample 50k) ---
    print("Generating plot 6/6: velocity by hour...")
    samp = df.sample(n=min(50_000, len(df)), random_state=42)
    hour = (samp["TransactionDT"] // 3600) % 24
    # Copy to avoid pandas fragmentation warnings from column insertion.
    samp = samp.copy()
    samp["_hour"] = hour
    counts_by = (
        samp.groupby(["_hour", "isFraud"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
    )
    col0 = 0 if 0 in counts_by.columns else counts_by.columns.min()
    col1 = 1 if 1 in counts_by.columns else None
    if col1 is None:
        legit_h = counts_by[col0].values.astype(float)
        fraud_h = np.zeros(24)
    else:
        legit_h = counts_by[col0].values.astype(float)
        fraud_h = counts_by[col1].values.astype(float)
    leg_total = legit_h.sum()
    fr_total = fraud_h.sum()
    leg_pct = np.where(leg_total > 0, 100.0 * legit_h / leg_total, 0.0)
    fr_pct = np.where(fr_total > 0, 100.0 * fraud_h / fr_total, 0.0)
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, leg_pct, marker="o", label="Legitimate (% of legit)")
    ax.plot(hours, fr_pct, marker="s", label="Fraud (% of fraud)")
    ax.set_title("Transaction share by hour of day (sampled, normalized within class)")
    ax.set_xlabel("Hour of day (0–23)")
    ax.set_ylabel("% of class total")
    ax.set_xticks(hours)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "velocity_by_hour.png", dpi=150)
    plt.close(fig)

    # --- Top 3 numeric mean-difference features ---
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    fraud_mask = df["isFraud"] == 1

    fraud_means = df.loc[fraud_mask, numeric_df.columns].mean()
    nonfraud_means = df.loc[~fraud_mask, numeric_df.columns].mean()
    mean_diffs = (fraud_means - nonfraud_means).abs().dropna()
    top3 = mean_diffs.sort_values(ascending=False).head(3).index.tolist()
    top3_safe = top3 + ["(insufficient data)"] * max(0, 3 - len(top3))

    fraud_count = int(df["isFraud"].sum())
    novel_fraud_s = int(((df["ProductCD"] == "S") & (df["isFraud"] == 1)).sum())
    exposure = fraud_count * 200

    key_finding = (
        f"Fraud is sparse ({fraud_rate:.2f}% of volume) yet measurable: the largest mean-value gaps "
        f"appear for {top3_safe[0]}, {top3_safe[1]}, and {top3_safe[2]}, highlighting which numeric fields separate "
        f"the classes most strongly. "
        f"Hour-of-day concentration and product channel (ProductCD) further stratify risk and complement "
        f"amount-based rules for client-facing monitoring."
    )

    print()
    print("========================================")
    print("PARTNER EVALUATION SUMMARY")
    print("========================================")
    print(f"Total transactions analyzed: {n}")
    print(f"Overall fraud rate: {fraud_rate:.4f}%")
    print(
        f"Withheld novel fraud category (ProductCD=S, isFraud=1): {novel_fraud_s} transactions held out from training"
    )
    print(f"Estimated dollar exposure (fraud * $200): ${exposure:,}")
    print(f"Top 3 fraud-correlated features by mean value difference: {top3}")
    print()
    print(f"Key finding: {key_finding}")
    print("========================================")
    print()
    print(
        f"Count of ProductCD == 'S' AND isFraud == 1 rows (this is the novel fraud holdout size): {novel_fraud_s}"
    )
    print("Confirm these will be withheld from training in subsequent steps.")


if __name__ == "__main__":
    main()
