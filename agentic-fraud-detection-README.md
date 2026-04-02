# Agentic Fraud Detection

A research prototype combining supervised fraud detection, unsupervised novel pattern discovery, and a LangGraph multi-agent investigation layer with GPT-4o generated partner-facing explanations.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green) ![LightGBM](https://img.shields.io/badge/LightGBM-gradient--boosting-orange)

---

## The Problem

Most fraud detection systems are strong at catching patterns that resemble historical fraud. They are structurally blind to fraud they have never seen before. A supervised model trained on labeled data cannot, by definition, flag a fraud pattern it was never shown.

This project addresses that gap directly:

- **Known fraud** is handled by a LightGBM classifier trained on labeled historical transactions
- **Novel fraud** — patterns the supervised model has never seen — is surfaced by an Isolation Forest anomaly detector trained exclusively on legitimate transactions
- **Every flagged transaction** is investigated and explained in plain English by a LangGraph agent pipeline powered by GPT-4o

---

## The Novel Fraud Simulation

To rigorously evaluate novel fraud detection, an entire fraud subcategory (`ProductCD == 'S'`, 686 transactions) is withheld from training entirely and included only in the test set. The supervised model has genuinely never seen these patterns at inference time.

This mirrors how real fraud teams evaluate out-of-distribution detection capability — not by claiming a model generalizes, but by proving it on a held-out category it was never shown.

---

## Key Results

| Detection Mode | Known Fraud AUC | Novel Fraud AUC | Fraud Caught ($) |
|---|---|---|---|
| Supervised only | 0.9526 | 0.4742 | $582,000 |
| Anomaly only | 0.7037 | 0.8171 | $146,400 |
| Combined (dual-layer) | 0.9169 | 0.7985 | $588,000 |

**The headline finding:** The supervised model scores AUC 0.47 on novel fraud — worse than random. The anomaly layer recovers this to AUC 0.80 on the exact same transactions, purely by recognizing behavioral deviation from normal rather than matching known fraud patterns.

---

## Architecture

### Diagram 1 — LangGraph Agent Pipeline

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
            LM["500 estimators · lr 0.05\nnum_leaves 64 · scale_pos_weight 10\nTrained on 471k transactions"]
            LP["predict_proba\n→ supervised_score 0–1"]
            LM --> LP
        end
        subgraph ISO["Isolation Forest"]
            IM["200 estimators · contamination 0.035\nTrained on legit only\n455k transactions"]
            IP["decision_function\n→ min-max normalize → invert\n→ anomaly_score 0–1"]
            IM --> IP
        end
        COMB["combined_score = max(supervised, anomaly)"]
        LP --> COMB
        IP --> COMB
    end

    subgraph ROUTE["CONDITIONAL ROUTING"]
        THRESH{"combined_score\nthreshold check"}
    end

    subgraph AUTOAPPROVE["AUTO-APPROVE PATH"]
        AA["Exit pipeline · No LLM call\nDecision = APPROVE · Latency < 10ms\n~76% of flagged volume"]
    end

    subgraph INVESTIGATE["INVESTIGATION AGENT — Node 2"]
        VEL["Card Velocity\nCount prior txns same card1"]
        AMT["Amount Deviation\nσ from card1 group mean"]
        SHAP["SHAP Analysis\nTreeExplainer · Top 5 features\nincreases_risk / decreases_risk"]
        PAT["Pattern Classification\nsupervised > anomaly → known_pattern\nelse → novel_anomaly"]
        SIM["Similar Cases\nCosine similarity vs 471k training rows\nTop 5 nearest neighbors"]
        VEL & AMT & SHAP & PAT & SIM --> CTX["Investigation Context Dict"]
    end

    subgraph EXPLAIN["EXPLANATION AGENT — Node 3"]
        PROMPT["Prompt: tx fields + 3 scores\n+ full investigation JSON"]
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
    THRESH -->|"score < 0.3"| AA
    THRESH -->|"score >= 0.3"| INVESTIGATE
    CTX --> PROMPT
    PARSE --> DEC
    AA --> DEC

    style INPUT fill:#E8F4FD,stroke:#3b82f6
    style PREPROCESS fill:#F5F3FF,stroke:#8b5cf6
    style SCORING fill:#EDE9FE,stroke:#7C3AED
    style LGBM fill:#DDD6FE,stroke:#7C3AED
    style ISO fill:#FEF3C7,stroke:#d97706
    style ROUTE fill:#F3F4F6,stroke:#6b7280
    style AUTOAPPROVE fill:#D1FAE5,stroke:#059669
    style INVESTIGATE fill:#FEF9C3,stroke:#ca8a04
    style EXPLAIN fill:#FCE7F3,stroke:#9d174d
    style OUTPUT fill:#F0FDF4,stroke:#16a34a
```

The three-agent design is deliberate. Low-risk transactions exit after scoring without ever hitting the LLM — keeping latency and cost minimal for the majority of volume. The LLM only runs where human-readable explanation actually adds value.

---

### Diagram 2 — Dual-Layer Detection Architecture

```mermaid
flowchart TD
    subgraph DATA["DATA PREPARATION"]
        RAW["Raw Data\ntrain_transaction.csv · 590,540 rows · 394 cols\ntrain_identity.csv · 144,233 rows · 41 cols"]
        MERGE["Left Merge on TransactionID\n590,540 rows · 434 cols · fraud rate 3.499%"]
        HOLDOUT["Novel Fraud Holdout\nProductCD=S AND isFraud=1\n686 transactions removed from training entirely"]
        SPLIT["Stratified Train/Test Split · random_state=42\ntrain: 471,883 · test: 117,971 · holdout: test set only"]
        RAW --> MERGE --> HOLDOUT --> SPLIT
    end

    subgraph L1["LAYER 1 — SUPERVISED DETECTION"]
        subgraph LGBM_TRAIN["LightGBM Training"]
            LF["49 features · LabelEncoding · Missing → -999"]
            LM["n_estimators=500 · lr=0.05 · num_leaves=64\nsubsample=0.8 · scale_pos_weight=10\nearly_stopping_rounds=50"]
            LS["SHAP TreeExplainer\nTop 20 feature importance\nPer-transaction attribution"]
            LF --> LM --> LS
        end
        subgraph LGBM_PERF["Performance"]
            LP1["Test AUC: 0.9526\nPrecision: 0.5276 · Recall: 0.7284 · F1: 0.6119"]
            LP2["Novel Fraud AUC: 0.4742\nWorse than random — blind spot confirmed"]
        end
    end

    subgraph L2["LAYER 2 — ANOMALY DETECTION"]
        subgraph ISO_TRAIN["Isolation Forest Training"]
            IF["Train on LEGIT ONLY · 455,901 transactions\nNo fraud labels used at any stage"]
            IM["n_estimators=200 · contamination=0.035\nrandom_state=42"]
            IS["Score: raw = decision_function\nnormalized = min-max · anomaly_score = 1 - normalized"]
            IF --> IM --> IS
        end
        subgraph ISO_PERF["Performance"]
            IP1["Test AUC: 0.7037\nLower on known fraud — expected, no labels"]
            IP2["Novel Fraud AUC: 0.8171\nCatches what supervised misses entirely"]
        end
    end

    subgraph COMBINE["DUAL-LAYER COMBINATION"]
        CS["combined_score = max(supervised_score, anomaly_score)"]
        CR1["Test recall: 0.7359 · Fraud caught: $588,000"]
        CR2["Novel Fraud AUC: 0.7985 · 82 of 686 caught\nAUC recovery: 0.47 → 0.80"]
        CR3["False Positives: 6,329\nPrecision-recall tradeoff vs supervised-only 2,606 FP"]
        CS --> CR1 & CR2 & CR3
    end

    subgraph DOWNSTREAM["ROUTING"]
        THR{"combined_score >= 0.3?"}
        AGT["LangGraph Agent Pipeline\nInvestigation + GPT-4o Explanation"]
        AUTO["Auto-Approve · No LLM overhead"]
        THR -->|"Yes"| AGT
        THR -->|"No"| AUTO
    end

    SPLIT --> L1
    SPLIT --> L2
    L1 --> CS
    L2 --> CS
    CS --> THR

    style DATA fill:#E8F4FD,stroke:#3b82f6
    style L1 fill:#EDE9FE,stroke:#7C3AED
    style LGBM_TRAIN fill:#DDD6FE,stroke:#7C3AED
    style LGBM_PERF fill:#F5F3FF,stroke:#7C3AED
    style L2 fill:#FEF3C7,stroke:#d97706
    style ISO_TRAIN fill:#FEF9C3,stroke:#d97706
    style ISO_PERF fill:#FFFBEB,stroke:#d97706
    style COMBINE fill:#D1FAE5,stroke:#059669
    style DOWNSTREAM fill:#F3F4F6,stroke:#6b7280
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Supervised detection | LightGBM, SHAP |
| Anomaly detection | Isolation Forest (scikit-learn) |
| Agent orchestration | LangGraph, LangChain |
| LLM explanation | OpenAI GPT-4o |
| API | FastAPI |
| Frontend | Streamlit |
| Data processing | Pandas, NumPy, PySpark |

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) — `train_transaction.csv` + `train_identity.csv` merged on `TransactionID`.

- 590,540 transactions
- 3.5% fraud rate
- 434 features after merge
- Novel fraud holdout: 686 transactions (ProductCD=S, isFraud=1) withheld from training

---

## Project Structure

```
agentic-fraud-detection/
├── data/                       ← place Kaggle CSVs here
├── models/
│   ├── supervised.py           ← LightGBM training + SHAP
│   ├── anomaly.py              ← Isolation Forest training
│   └── evaluator.py            ← detection comparison tables
├── agents/
│   └── graph.py                ← LangGraph pipeline
├── api/
│   └── main.py                 ← FastAPI endpoints
├── outputs/
│   └── eda/                    ← saved plots and CSVs
├── eda.py                      ← exploratory analysis
├── app.py                      ← Streamlit frontend
└── requirements.txt
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download dataset**

Download `train_transaction.csv` and `train_identity.csv` from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and place in `data/`.

**3. Set environment variables**
```bash
# Create .env file in project root
OPENAI_API_KEY=your_key_here
```

**4. Run in order**
```bash
# Exploratory analysis
python eda.py

# Train models
python models/supervised.py
python models/anomaly.py
python models/evaluator.py

# Launch app
streamlit run app.py

# Optional: API
uvicorn api.main:app --reload
```

---

## Streamlit App

**Tab 1 — Transaction Evaluator**

Loads and scores all flagged transactions (combined score > 0.3) from the test set on startup. Click any row to run the full agent pipeline in real time — scoring, SHAP investigation, behavioral context, and GPT-4o explanation fire instantly for the selected transaction.

**Tab 2 — Model Performance**

Pre-computed detection metrics across all three modes, dollar impact chart, and false positive comparison. The novel fraud holdout results are the key section — they show exactly where supervised-only systems leak value.

---

## Design Decisions Worth Noting

**Why Isolation Forest over an Autoencoder?**
For a research prototype, Isolation Forest is interpretable and fast to train. An Autoencoder would learn richer representations of normal behavior and likely improve novel fraud recall — a natural next step.

**Why GPT-4o for explanation rather than rule-based templates?**
Rule-based templates are fast but brittle. GPT-4o synthesizes SHAP values, velocity signals, amount deviation, and similar case context into a coherent narrative that adapts to each transaction's specific risk profile. For a fraud operations team reviewing hundreds of alerts, explanation quality directly affects analyst efficiency.

**Why ESCALATE as a decision output?**
Novel fraud detection cannot be fully automated. When the anomaly score dominates over the supervised score — indicating the system is outside its training distribution — the correct response is to route to a human reviewer rather than make an autonomous decision. The ESCALATE path is not a fallback; it is the architecturally correct output for genuine uncertainty.

**The false positive tradeoff**
The combined model generates more false positives (6,329) than supervised-only (2,606). This is the fundamental precision-recall tradeoff in fraud detection. In production, the threshold would be tuned based on the business cost of each error type — missing fraud vs. incorrectly declining a legitimate customer. This project surfaces that tradeoff explicitly rather than optimizing a single metric.

---

## Limitations

- IEEE-CIS is transaction fraud data. The dual-layer detection methodology applies equally to identity fraud and credit risk domains — the feature space and model architecture transfer; the domain-specific signal engineering does not.
- The novel fraud simulation uses a single withheld subcategory. Real novel fraud is more diverse and adversarial than a clean holdout split can capture.
- Cosine similarity for similar case retrieval runs against the full training set on each query. In production, approximate nearest neighbor search (FAISS) would be necessary at scale.
- GPT-4o explanation quality depends on the richness of the investigation context passed to it. Thin behavioral context produces generic explanations.

---

## Future Work

### Deep Learning Anomaly Detection — Autoencoder

The current anomaly layer uses Isolation Forest, which detects outliers based on feature-level statistical isolation. The natural upgrade is a **deep autoencoder** trained on legitimate transactions only.

The intuition: an autoencoder learns to compress and reconstruct normal behavior through a neural network bottleneck. At inference, it attempts to reconstruct any incoming transaction. If reconstruction error is high, the network couldn't "understand" the transaction — meaning it deviates from learned normal behavior in ways that go beyond simple statistical outliers.

This is more powerful than Isolation Forest because it learns deep feature interactions and subtle behavioral correlations that tree-based methods miss. A transaction that passes all individual feature checks can still fail reconstruction if its *combination* of signals is unlike anything the autoencoder learned as normal.

```mermaid
flowchart LR
    subgraph TRAIN["TRAINING — Legit Transactions Only · No Fraud Labels"]
        TD["455,901 legitimate transactions\n49 features · preprocessed"]
    end

    subgraph ENCODER["ENCODER — Compression"]
        E1["Dense 49→32\nReLU · BatchNorm · Dropout 0.2"]
        E2["Dense 32→16\nReLU · BatchNorm · Dropout 0.2"]
        E3["Dense 16→8\nReLU activation"]
        E1 --> E2 --> E3
    end

    subgraph BOTTLE["BOTTLENECK"]
        BN["Latent Space · 8 dims\nCompressed representation\nof normal behavior\nEncoder learns what normal looks like"]
    end

    subgraph DECODER["DECODER — Reconstruction"]
        D1["Dense 8→16\nReLU · BatchNorm"]
        D2["Dense 16→32\nReLU · BatchNorm"]
        D3["Dense 32→49\nSigmoid · Reconstruction output"]
        D1 --> D2 --> D3
    end

    subgraph LOSS["TRAINING OBJECTIVE"]
        MSE["Loss = MSE input vs reconstruction\nBackprop through full network\nTrained only on normal behavior"]
    end

    subgraph INFERENCE["INFERENCE — Anomaly Scoring"]
        INP["New Transaction · 49 features"]
        ERR["Reconstruction Error\nMSE input vs reconstruction"]
        LOW["Low Error\nLooks like normal behavior\n→ Legitimate"]
        HIGH["High Error\nNetwork cannot reconstruct\nDeviates from normal patterns\n→ Novel Anomaly Flagged"]
        INP --> ERR
        ERR -->|"error < threshold"| LOW
        ERR -->|"error >= threshold"| HIGH
    end

    TD --> E1
    E3 --> BN
    BN --> D1
    D3 --> MSE
    MSE -.->|"backprop"| E1
    BN --> INP

    style TRAIN fill:#E8F4FD,stroke:#3b82f6
    style ENCODER fill:#EDE9FE,stroke:#7C3AED
    style BOTTLE fill:#7C3AED,stroke:#5b21b6,color:#ffffff
    style DECODER fill:#DDD6FE,stroke:#7C3AED
    style LOSS fill:#FEF3C7,stroke:#d97706
    style INFERENCE fill:#D1FAE5,stroke:#059669
```

### Adversarial Red Team Agent

The deeper limitation of any static fraud detection system is that fraudsters adapt. A model trained today will degrade as fraud patterns evolve to evade it. This is not a theoretical concern — it is the operational reality of every production fraud system.

The planned extension is an **adversarial agent** that actively probes the detection system to find evasion strategies, then feeds those strategies back into retraining.

```mermaid
flowchart TD
    subgraph SYSTEM["CURRENT DETECTION SYSTEM"]
        CM["LightGBM + Isolation Forest\nCurrent decision boundary"]
        SHAP_EXP["SHAP Feature Analysis\nTop features · decision thresholds\nFeature weight distribution exposed"]
        CM --> SHAP_EXP
    end

    subgraph REDTEAM["RED TEAM AGENT — Adversarial Generator"]
        RTP["GPT-4o Prompt\nYou are a sophisticated fraudster.\nGiven these SHAP weights and thresholds:\n1. Evade supervised score threshold\n2. Appear statistically normal to anomaly detector\n3. Maximize transaction value\n4. Exploit current feature blind spots"]
        RTG["Adversarial Transaction Generator\nManipulates high-weight features\nStays below anomaly threshold\nMimics legitimate behavioral patterns"]
        RTP --> RTG
    end

    subgraph EVAL["EVASION EVALUATION"]
        SCORE["Score adversarial transactions\nagainst current detection system"]
        FILTER["Filter successful evasions\ncombined_score < detection threshold\nThese are the dangerous ones"]
        AUDIT["Evasion analysis\nWhich features exploited · What gaps used\nDollar exposure of evasion surface"]
        SCORE --> FILTER --> AUDIT
    end

    subgraph HARDEN["ADVERSARIAL DATASET CONSTRUCTION"]
        LABEL["Label evasions as fraud\nisFraud = 1\nAdded to training corpus"]
        AUG["Original training data\n+ adversarial fraud examples\nHarder · more diverse fraud cases"]
        LABEL --> AUG
    end

    subgraph RETRAIN["MODEL RETRAINING"]
        RT["Retrain LightGBM\nNew decision boundary\nCovers adversarial patterns"]
        RI["Retrain Isolation Forest\nUpdated anomaly thresholds"]
        EVAL_NEW["Evaluate improvement\nAUC on adversarial holdout\nReduction in evasion rate\nFalse positive tradeoff check"]
        RT & RI --> EVAL_NEW
    end

    subgraph CONV["CONVERGENCE PROPERTIES"]
        C1["Each iteration:\nEvasion surface shrinks\nModel generalizes further\nNovel fraud harder to craft\nRaises the cost of evasion\nDoes not fully converge\nFraudsters also adapt"]
    end

    SHAP_EXP --> RTP
    RTG --> SCORE
    AUG --> RT & RI
    EVAL_NEW -->|"Loop repeats\nnew SHAP weights"| RTP
    EVAL_NEW --> C1

    style SYSTEM fill:#EDE9FE,stroke:#7C3AED
    style REDTEAM fill:#FEE2E2,stroke:#dc2626
    style EVAL fill:#FEF3C7,stroke:#d97706
    style HARDEN fill:#FCE7F3,stroke:#9d174d
    style RETRAIN fill:#D1FAE5,stroke:#059669
    style CONV fill:#F0F9FF,stroke:#0284c7
```

This closes the loop the current architecture leaves open: the system learns not just from fraud that happened, but from fraud that *could* happen given what it currently knows.

---

## Author

Ishan Joshi — [GitHub](https://github.com/nishaanjoshi0) · [LinkedIn](https://linkedin.com/in/ishannjoshi)
