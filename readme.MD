# Agentic Fraud Detection

Research prototype. Supervised fraud detection + unsupervised novel pattern discovery + LangGraph multi-agent investigation layer with GPT-4o explanations.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green) ![LightGBM](https://img.shields.io/badge/LightGBM-gradient--boosting-orange)

---

## Problem

Supervised fraud models catch patterns that match historical fraud. They are structurally blind to fraud they have never been shown. A model trained on labeled data cannot flag a pattern that does not exist in its training set.

This project targets that gap across two layers:

- **Known fraud** handled by LightGBM trained on labeled historical transactions
- **Novel fraud** surfaced by Isolation Forest trained exclusively on legitimate transactions, with no fraud labels
- **Every flagged transaction** investigated and explained by a LangGraph agent pipeline powered by GPT-4o

---

## Novel Fraud Simulation

An entire fraud subcategory (`ProductCD == 'S'`, 686 transactions) is withheld from training and placed only in the test set. The supervised model has never seen these patterns at inference time.

This is not a standard train/test split. It is an out-of-distribution evaluation: proving the system catches fraud categories it was never shown, rather than claiming generalization without evidence.

---

## Results

| Detection Mode | Known Fraud AUC | Novel Fraud AUC | Fraud Caught |
|---|---|---|---|
| Supervised only | 0.9526 | 0.4742 | $582,000 |
| Anomaly only | 0.7037 | 0.8171 | $146,400 |
| Combined | 0.9169 | 0.7985 | $588,000 |

Supervised AUC on novel fraud: **0.47** (worse than random). Anomaly layer on the same transactions: **0.80**. The delta is the business case for the dual-layer architecture.

---

## Architecture

### 1. LangGraph Agent Pipeline

```mermaid
flowchart TD
    subgraph INPUT["INPUT LAYER"]
        TXN["Transaction\nTransactionAmt · ProductCD\ncard1-6 · addr1-2 · dist1-2\nC1-C14 · D1-D15 · M1-M9\n49 features total"]
    end

    subgraph PREPROCESS["PREPROCESSING"]
        LE["Label Encoding\ncard4 · card6 · M1-M9\nfit on train only"]
        FV["Missing Value Imputation\nfill -999"]
        LE --> FV
    end

    subgraph SCORING["SCORING AGENT — Node 1"]
        direction LR
        subgraph LGBM["LightGBM Classifier"]
            LM["500 estimators · lr 0.05\nnum_leaves 64 · scale_pos_weight 10\n471k transactions"]
            LP["predict_proba\nsupervised_score 0-1"]
            LM --> LP
        end
        subgraph ISO["Isolation Forest"]
            IM["200 estimators · contamination 0.035\nTrained on legit only · 455k transactions"]
            IP["decision_function\nmin-max normalize · invert\nanomaly_score 0-1"]
            IM --> IP
        end
        COMB["combined_score = max(supervised, anomaly)"]
        LP --> COMB
        IP --> COMB
    end

    subgraph ROUTE["ROUTING"]
        THRESH{"combined_score\nthreshold check"}
    end

    subgraph AUTOAPPROVE["AUTO-APPROVE"]
        AA["Exit pipeline\nNo LLM call\nDecision = APPROVE\nLatency under 10ms"]
    end

    subgraph INVESTIGATE["INVESTIGATION AGENT — Node 2"]
        VEL["Card Velocity\nCount prior txns same card1"]
        AMT["Amount Deviation\nsigma from card1 group mean"]
        SHAP["SHAP Analysis\nTreeExplainer · Top 5 features\nincreases_risk / decreases_risk"]
        PAT["Pattern Classification\nsupervised > anomaly = known_pattern\nelse = novel_anomaly"]
        SIM["Similar Cases\nCosine similarity vs 471k rows\nTop 5 nearest neighbors"]
        VEL & AMT & SHAP & PAT & SIM --> CTX["Investigation Context Dict"]
    end

    subgraph EXPLAIN["EXPLANATION AGENT — Node 3"]
        PROMPT["Prompt: tx fields + 3 scores\nfull investigation JSON"]
        GPT["GPT-4o · max_tokens=300\nPartner-facing fraud narrative"]
        PARSE["Parse DECISION line\nAPPROVE / DECLINE / ESCALATE\nfallback = ESCALATE"]
        PROMPT --> GPT --> PARSE
    end

    subgraph OUTPUT["DECISION OUTPUT"]
        DEC["supervised_score · anomaly_score\ncombined_score · pattern_type\nexplanation text · decision label"]
    end

    TXN --> LE
    FV --> SCORING
    COMB --> THRESH
    THRESH -->|"score under 0.3"| AA
    THRESH -->|"score 0.3 or above"| INVESTIGATE
    CTX --> PROMPT
    PARSE --> DEC
    AA --> DEC

    style INPUT fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style PREPROCESS fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style SCORING fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style LGBM fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style ISO fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style ROUTE fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style AUTOAPPROVE fill:#0d2010,stroke:#2d9e4e,color:#a0e8b0
    style INVESTIGATE fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style EXPLAIN fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style OUTPUT fill:#0d2010,stroke:#2d9e4e,color:#a0e8b0
```

Low-risk transactions exit after scoring without hitting the LLM. The LLM runs only where a human-readable explanation adds value. This keeps cost and latency minimal across the majority of volume.

---

### 2. Dual-Layer Detection Architecture

```mermaid
flowchart TD
    subgraph DATA["DATA PREPARATION"]
        RAW["train_transaction.csv · 590,540 rows · 394 cols\ntrain_identity.csv · 144,233 rows · 41 cols"]
        MERGE["Left merge on TransactionID\n590,540 rows · 434 cols · fraud rate 3.499%"]
        HOLDOUT["Novel Fraud Holdout\nProductCD=S AND isFraud=1\n686 transactions removed from training"]
        SPLIT["Stratified split · random_state=42\ntrain 471,883 · test 117,971 · holdout test only"]
        RAW --> MERGE --> HOLDOUT --> SPLIT
    end

    subgraph L1["LAYER 1 — SUPERVISED"]
        subgraph LGBM_TRAIN["LightGBM"]
            LF["49 features · LabelEncoding · missing -999"]
            LM["n_estimators=500 · lr=0.05 · num_leaves=64\nsubsample=0.8 · scale_pos_weight=10"]
            LS["SHAP TreeExplainer\nTop 20 features · per-transaction attribution"]
            LF --> LM --> LS
        end
        subgraph LGBM_PERF["Performance"]
            LP1["Test AUC: 0.9526\nPrecision 0.5276 · Recall 0.7284 · F1 0.6119"]
            LP2["Novel Fraud AUC: 0.4742\nWorse than random · blind spot confirmed"]
        end
    end

    subgraph L2["LAYER 2 — ANOMALY"]
        subgraph ISO_TRAIN["Isolation Forest"]
            IF["Trained on legit only · 455,901 transactions\nNo fraud labels used"]
            IM["n_estimators=200 · contamination=0.035"]
            IS["raw = decision_function\nnormalized = min-max · score = 1 - normalized"]
            IF --> IM --> IS
        end
        subgraph ISO_PERF["Performance"]
            IP1["Test AUC: 0.7037\nLower on known fraud · expected without labels"]
            IP2["Novel Fraud AUC: 0.8171\nCatches what supervised misses"]
        end
    end

    subgraph COMBINE["COMBINATION"]
        CS["combined_score = max(supervised_score, anomaly_score)"]
        CR1["Test recall 0.7359 · Fraud caught $588,000"]
        CR2["Novel Fraud AUC 0.7985 · 82 of 686 caught\nRecovery: 0.47 to 0.80 AUC"]
        CR3["False Positives: 6,329\nPrecision-recall tradeoff vs supervised 2,606 FP"]
        CS --> CR1 & CR2 & CR3
    end

    subgraph DOWNSTREAM["ROUTING"]
        THR{"combined_score 0.3 or above?"}
        AGT["LangGraph Agent\nInvestigation + GPT-4o"]
        AUTO["Auto-Approve\nNo LLM overhead"]
        THR -->|"Yes"| AGT
        THR -->|"No"| AUTO
    end

    SPLIT --> L1
    SPLIT --> L2
    L1 --> CS
    L2 --> CS
    CS --> THR

    style DATA fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style L1 fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style LGBM_TRAIN fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style LGBM_PERF fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style L2 fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style ISO_TRAIN fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style ISO_PERF fill:#241c0e,stroke:#c8860a,color:#f5f0e8
    style COMBINE fill:#0d2010,stroke:#2d9e4e,color:#a0e8b0
    style DOWNSTREAM fill:#1c1610,stroke:#c8860a,color:#f5f0e8
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Supervised detection | LightGBM, SHAP |
| Anomaly detection | Isolation Forest |
| Agent orchestration | LangGraph, LangChain |
| LLM explanation | OpenAI GPT-4o |
| API | FastAPI |
| Frontend | Streamlit |
| Data processing | Pandas, NumPy |

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data): `train_transaction.csv` + `train_identity.csv` merged on `TransactionID`.

- 590,540 transactions · 3.5% fraud rate · 434 features after merge
- Novel fraud holdout: 686 transactions (ProductCD=S, isFraud=1) withheld from training entirely

---

## Project Structure

```
agentic-fraud-detection/
├── data/                       place Kaggle CSVs here
├── models/
│   ├── supervised.py           LightGBM training + SHAP
│   ├── anomaly.py              Isolation Forest training
│   └── evaluator.py            detection comparison tables
├── agents/
│   └── graph.py                LangGraph pipeline
├── api/
│   └── main.py                 FastAPI endpoints
├── outputs/
│   └── eda/                    saved plots and CSVs
├── eda.py                      exploratory analysis
├── app.py                      Streamlit frontend
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
```

Add `OPENAI_API_KEY=your_key` to a `.env` file in the project root.

```bash
python eda.py
python models/supervised.py
python models/anomaly.py
python models/evaluator.py
streamlit run app.py
```

---

## Streamlit App

**Tab 1 — Transaction Evaluator**

Loads and scores all flagged transactions on startup. Click any row to run the full agent pipeline in real time: scoring, SHAP investigation, behavioral context, and GPT-4o explanation.

**Tab 2 — Model Performance**

Pre-computed metrics across all three detection modes, dollar impact chart, and false positive comparison. The novel fraud holdout section is the key finding.

---

## Design Decisions

**Isolation Forest over Autoencoder**
Isolation Forest is interpretable and fast for a research prototype. An autoencoder learns richer representations of normal behavior and would improve novel fraud recall at the cost of training complexity. That is the natural next step.

**GPT-4o over rule-based templates**
Templates are fast but rigid. GPT-4o synthesizes SHAP values, velocity signals, amount deviation, and similar case context into a narrative that adapts to each transaction's specific risk profile. For a fraud operations team reviewing hundreds of alerts daily, explanation quality directly affects analyst throughput.

**ESCALATE as a first-class decision**
Novel fraud detection cannot be fully automated. When anomaly score dominates over supervised score, the system is outside its training distribution. The correct output is a human reviewer, not an autonomous decline. ESCALATE is not a fallback. It is the architecturally correct response to genuine uncertainty.

**False positive tradeoff**
Combined model: 6,329 false positives. Supervised only: 2,606. Adding the anomaly layer casts a wider net. In production the threshold gets tuned against the business cost of each error type: missed fraud vs. declined legitimate customer. This project surfaces that tradeoff rather than hiding it behind a single optimized metric.

---

## Limitations

- IEEE-CIS is transaction fraud data. The dual-layer methodology transfers to identity fraud and credit risk domains. The feature engineering does not.
- Novel fraud simulation uses one withheld subcategory. Real novel fraud is more diverse and adversarial than a clean holdout split captures.
- Cosine similarity for similar case retrieval runs against the full training set per query. Production deployment requires approximate nearest neighbor search (FAISS) at scale.
- GPT-4o explanation quality depends on the richness of investigation context. Thin context produces generic output.

---

## Future Work

### Autoencoder for Novel Fraud Detection

Isolation Forest detects statistical outliers at the feature level. A deep autoencoder learns what normal looks like at a structural level and catches deviations that tree-based methods miss.

Trained on legitimate transactions only, the network learns to compress and reconstruct normal behavior. At inference, high reconstruction error signals the transaction deviates from learned normal patterns regardless of whether it matches any known fraud label.

```mermaid
flowchart LR
    subgraph TRAIN["TRAINING — Legit Only · No Fraud Labels"]
        TD["455,901 legitimate transactions\n49 features · preprocessed"]
    end

    subgraph ENCODER["ENCODER"]
        E1["Dense 49 to 32\nReLU · BatchNorm · Dropout 0.2"]
        E2["Dense 32 to 16\nReLU · BatchNorm · Dropout 0.2"]
        E3["Dense 16 to 8\nReLU"]
        E1 --> E2 --> E3
    end

    subgraph BOTTLE["BOTTLENECK"]
        BN["Latent Space · 8 dims\nCompressed normal behavior"]
    end

    subgraph DECODER["DECODER"]
        D1["Dense 8 to 16\nReLU · BatchNorm"]
        D2["Dense 16 to 32\nReLU · BatchNorm"]
        D3["Dense 32 to 49\nSigmoid · Reconstruction"]
        D1 --> D2 --> D3
    end

    subgraph INFERENCE["INFERENCE"]
        INP["New Transaction · 49 features"]
        ERR["Reconstruction Error\nMSE input vs output"]
        LOW["Low Error\nLooks normal · Legitimate"]
        HIGH["High Error\nCannot reconstruct\nNovel Anomaly Flagged"]
        INP --> ERR
        ERR -->|"under threshold"| LOW
        ERR -->|"at or above threshold"| HIGH
    end

    TD --> E1
    E3 --> BN
    BN --> D1
    D3 -.->|"MSE loss · backprop"| E1
    BN --> INP

    style TRAIN fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style ENCODER fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style BOTTLE fill:#3d2000,stroke:#c8860a,color:#f5c842
    style DECODER fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style INFERENCE fill:#0d2010,stroke:#2d9e4e,color:#a0e8b0
```

---

### Adversarial Red Team Agent

Static fraud systems degrade as fraud patterns evolve to evade them. The planned extension is an adversarial agent that probes the detection system for evasion strategies and feeds those strategies back into retraining.

```mermaid
flowchart TD
    subgraph SYSTEM["DETECTION SYSTEM"]
        CM["LightGBM + Isolation Forest\nCurrent decision boundary"]
        SHAP_EXP["SHAP Feature Analysis\nTop features · decision thresholds · weight distribution"]
        CM --> SHAP_EXP
    end

    subgraph REDTEAM["RED TEAM AGENT"]
        RTP["GPT-4o prompt\nGiven these SHAP weights and thresholds:\n1. Evade supervised score threshold\n2. Appear statistically normal\n3. Maximize transaction value\n4. Exploit feature blind spots"]
        RTG["Adversarial Transaction Generator\nManipulates high-weight features\nStays below anomaly threshold"]
        RTP --> RTG
    end

    subgraph EVAL["EVASION EVALUATION"]
        SCORE["Score adversarial transactions\nagainst current system"]
        FILTER["Filter successful evasions\ncombined_score under threshold"]
        AUDIT["Evasion analysis\nFeatures exploited · gaps used · dollar exposure"]
        SCORE --> FILTER --> AUDIT
    end

    subgraph HARDEN["DATASET HARDENING"]
        LABEL["Label evasions isFraud=1\nAdd to training corpus"]
        AUG["Original data + adversarial examples\nHarder · more diverse fraud cases"]
        LABEL --> AUG
    end

    subgraph RETRAIN["RETRAINING"]
        RT["Retrain LightGBM\nNew decision boundary"]
        RI["Retrain Isolation Forest\nUpdated thresholds"]
        EVAL_NEW["Evaluate\nAUC on adversarial holdout\nEvasion rate reduction\nFalse positive check"]
        RT & RI --> EVAL_NEW
    end

    subgraph CONV["CONVERGENCE"]
        C1["Each iteration:\nEvasion surface shrinks\nModel generalizes further\nRaises cost of evasion\nDoes not fully converge\nFraudsters also adapt"]
    end

    SHAP_EXP --> RTP
    RTG --> SCORE
    AUG --> RT & RI
    EVAL_NEW -->|"Loop · new SHAP weights"| RTP
    EVAL_NEW --> C1

    style SYSTEM fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style REDTEAM fill:#200d0d,stroke:#dc2626,color:#fca5a5
    style EVAL fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style HARDEN fill:#1c1610,stroke:#c8860a,color:#f5f0e8
    style RETRAIN fill:#0d2010,stroke:#2d9e4e,color:#a0e8b0
    style CONV fill:#0d1520,stroke:#3b82f6,color:#93c5fd
```

The system learns not just from fraud that happened, but from fraud that could happen given what it currently knows.

---

## Author

Ishan Joshi · [GitHub](https://github.com/nishaanjoshi0) · [LinkedIn](https://linkedin.com/in/ishannjoshi)
