# Core Demand Prediction Engine

**An economically calibrated system for predicting recurring enterprise procurement demand, maximizing net savings under fee constraints.**

> 🏆 **#1 on Level 1 leaderboard** at the end of the hackathon — built through disciplined economic modeling, not through overfitting or hyperparameter tricks.

---

## Why This Problem Matters

Enterprise procurement is dominated by the **long tail**: hundreds of one-off purchases that generate noise, obscure patterns, and waste analyst time. Buried inside that long tail is a much smaller set of **core demand** — product categories a buyer purchases repeatedly, predictably, and with real economic value.

Identifying core demand unlocks procurement automation, contract negotiation leverage, and supplier consolidation. But the prediction problem is constrained: **every prediction incurs a fixed monthly fee**. Predict too many categories and the fees erode all savings. Predict too few and you miss consolidation opportunities.

This is not a recommendation problem. This is a **portfolio optimization problem under asymmetric cost**.

---

## The Economic Objective

$$\text{Score} = \sum_{i \in \text{hits}} \text{Savings}_i \;-\; \sum_{j \in \text{all predictions}} \text{Fee}_j$$

Where savings scale approximately as $\sqrt{\text{price}} \times \text{frequency}$ for correctly predicted categories, and each prediction incurs a fixed fee regardless of correctness.

**Key implication:** Precision matters more than recall. Every false positive costs money directly. The system must be *economically disciplined*, not just statistically accurate.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Level 1 Leaderboard** | **#1** |
| Warm-start backtest precision | 92.8% (1,297 / 1,397) |
| Total predictions | 2,205 across 100 buyers |
| Warm buyers served | 47 (+ 1 edge case → cold fallback) |
| Cold buyers served | 52 |
| Level 2 upgrade rate | 33% of L1 predictions |
| Robustness | Optimized for worst-case across plausible scoring parameters |

---

## Architecture Overview

```mermaid
graph TB
    subgraph Input["📊 Input Data"]
        TX[8.37M Transactions<br/>36 months, 65K buyers]
        TEST[100 Test Buyers<br/>47 warm + 53 cold]
        NACE[NACE Industry Codes]
        SKU[SKU Feature Metadata]
    end

    subgraph Router["🔀 Buyer Router"]
        CLASSIFY{Has transaction<br/>history?}
    end

    subgraph Warm["🔥 Warm-Start Pipeline"]
        direction TB
        W1[Recurrence Feature Engineering]
        W2[Spending Velocity + Recency Weighting]
        W3[Economic Threshold Filtering]
        W4[Per-Buyer Adaptive Floor]
    end

    subgraph Cold["❄️ Cold-Start Pipeline"]
        direction TB
        C1[NACE Industry Profile Lookup]
        C2[Size-Bucketed Peer Matching]
        C3[Spend-Weighted Collaborative Filtering]
        C4[Hierarchical Fallback Chain]
    end

    subgraph Calibration["⚙️ Calibration Engine"]
        direction TB
        BT[Temporal Backtest<br/>Train: 2023-2024<br/>Val: Jan-Jun 2025]
        GS[Grid Search<br/>savings_rate × fee × threshold]
        RA[Robustness Analysis<br/>Worst-case optimization]
    end

    subgraph Levels["📐 Multi-Level Expansion"]
        direction TB
        L1[Level 1: E-Class]
        L2[Level 2: E-Class + Manufacturer<br/>90% dominance, 15+ orders, recent]
        L3[Level 3: E-Class + Feature Cluster<br/>TF-IDF + silhouette-optimal k]
    end

    TX --> CLASSIFY
    TEST --> CLASSIFY
    CLASSIFY -->|"47 warm"| Warm
    CLASSIFY -->|"52 cold + 1 edge"| Cold
    NACE --> Cold
    SKU --> L3

    Warm --> L1
    Cold --> L1
    BT --> GS --> RA
    RA -->|"Optimal thresholds"| Warm
    RA -->|"Cold thresholds"| Cold
    L1 --> L2 --> L3

    style Input fill:#e8f4f8,stroke:#2563eb
    style Warm fill:#fef3c7,stroke:#f59e0b
    style Cold fill:#dbeafe,stroke:#3b82f6
    style Calibration fill:#f3e8ff,stroke:#8b5cf6
    style Levels fill:#dcfce7,stroke:#16a34a
```

---

## Warm-Start vs Cold-Start Strategy

The 100 test buyers split into two fundamentally different problems:

### Warm-Start (47 buyers)
Buyers with 2+ years of transaction history. The signal is **purchase recurrence**: categories bought across many distinct calendar months represent stable, ongoing demand.

**Features per (buyer, E-Class) pair:**
- `n_months_active` — distinct months with purchases
- `n_months_recent` — activity in last 6 months (Jan–Jun 2025)
- `n_months_weighted` — recent months weighted 2× to capture *ongoing* vs *stale* demand
- `avg_monthly_spend` — spending velocity in the recent window
- `recency_ratio` — recent spend / total spend
- `rank_score` — composite signal: `avg_monthly_spend × (1 + recency_ratio)`

**Decision rule:** Include a prediction if:
$$\text{savings\_rate} \times \text{expected\_spend} > \text{fee} \times \text{threshold\_mult}$$

This replaces a naive top-k cutoff with an *economically motivated threshold*: only predict categories where expected savings exceed expected cost.

### Cold-Start (52 buyers + 1 edge case)
Buyers with zero transaction history. Predictions rely entirely on industry signals and peer behavior.

**5-level hierarchical fallback:**

```mermaid
graph TD
    START[Cold Buyer] --> N4{NACE 4-digit<br/>+ size bucket<br/>≥10 peers?}
    N4 -->|Yes| P4[Score by penetration × √avg_value]
    N4 -->|No| N4F{NACE 4-digit<br/>any size?}
    N4F -->|Yes| P4F[Score by penetration × √avg_value]
    N4F -->|No| N2{NACE 2-digit?}
    N2 -->|Yes| P2[Broader industry profile]
    N2 -->|No| CF{Collaborative<br/>Filtering?}
    CF -->|Peers found| PCF[Cosine similarity on<br/>eclass purchase vectors]
    CF -->|No peers| GLOBAL[Global popularity<br/>fallback]

    P4 --> MERGE
    P4F --> MERGE
    P2 --> MERGE
    PCF --> MERGE
    GLOBAL --> MERGE

    MERGE[Merge & Deduplicate] --> ECON[Apply economic<br/>threshold]
    ECON --> OUT[Final cold<br/>predictions]

    style START fill:#dbeafe,stroke:#3b82f6
    style MERGE fill:#dcfce7,stroke:#16a34a
    style ECON fill:#fef3c7,stroke:#f59e0b
```

**Conservative portfolio:** 15 E-Classes per cold buyer (vs ~30 for warm). With zero history, the risk of fee waste is higher, so the system is deliberately selective.

---

## End-to-End Pipeline

```mermaid
graph LR
    subgraph Phase1["Phase 1-2"]
        E[Economic Backtest<br/>Metric]
        T[Economic Threshold<br/>Replaces fixed top-k]
    end

    subgraph Phase3["Phase 3"]
        V[Spending Velocity]
        R[Recency Ratio]
        RS[Rank Score]
    end

    subgraph Phase4["Phase 4"]
        CT[Cold Threshold<br/>Calibration]
    end

    subgraph Phase5["Phase 5"]
        GS[Grid Search<br/>savings_rate × fee<br/>× threshold_mult]
        RB[Robustness:<br/>min score across<br/>plausible params]
    end

    subgraph Phase6["Phase 6"]
        CF[Spend-Weighted CF<br/>for Cold-Start]
    end

    subgraph Phase7["Phase 7"]
        CB[Cold Backtest<br/>20% held-out buyers]
    end

    subgraph Phase8["Phase 8"]
        AF[Per-Buyer<br/>Adaptive Floor]
    end

    E --> T --> V --> RS
    R --> RS
    RS --> GS --> RB
    CT --> CB
    CF --> CB
    RB --> AF
    CB --> AF

    style Phase1 fill:#fef3c7
    style Phase5 fill:#f3e8ff
    style Phase7 fill:#dbeafe
```

The pipeline runs 8 phases sequentially, each addressing a specific aspect of the economic optimization:

| Phase | Purpose | Key Insight |
|-------|---------|-------------|
| 1 | Economic backtest metric | Score = savings - fees, not accuracy |
| 2 | Economic threshold | Dynamic inclusion based on expected savings vs fee |
| 3 | Spending velocity ranking | Recent spending rate + recency ratio as ranking signal |
| 4 | Cold-start economic threshold | Same fee-aware logic applied to industry profiles |
| 5 | Full grid search | Sweep over (savings_rate, fee, threshold_mult) space |
| 6 | Spend-weighted CF | Collaborative filtering using log1p(spend) weights on recurring purchases |
| 7 | Cold-start backtest | Hold out 20% of known buyers to validate cold logic |
| 8 | Per-buyer adaptive floor | Every warm buyer gets ≥1 prediction even below threshold |

---

## Economic Decision Logic

```mermaid
graph TD
    ITEM["(Buyer, E-Class) Pair"] --> COMPUTE["Compute expected_savings =<br/>savings_rate × avg_monthly_spend × prediction_months"]
    COMPUTE --> CHECK{"expected_savings ><br/>fee × threshold_mult?"}
    CHECK -->|"Yes: economically<br/>justified"| INCLUDE[✅ Include in portfolio]
    CHECK -->|"No: fee would<br/>exceed benefit"| EXCLUDE["❌ Exclude — fee waste"]

    INCLUDE --> RANK["Rank by<br/>rank_score = avg_monthly_spend<br/>× (1 + recency_ratio)"]

    EXCLUDE --> FLOOR{"Is this buyer's<br/>only candidate?"}
    FLOOR -->|"Yes"| FALLBACK["⚠️ Include top-1<br/>by rank_score<br/>(adaptive floor)"]
    FLOOR -->|"No"| DROP["Drop from portfolio"]

    style INCLUDE fill:#dcfce7,stroke:#16a34a
    style EXCLUDE fill:#fee2e2,stroke:#dc2626
    style FALLBACK fill:#fef3c7,stroke:#f59e0b
```

### Why Overprediction Hurts

```mermaid
graph LR
    subgraph Conservative["Conservative Portfolio (K=15)"]
        C_SAV["Savings: €800K"]
        C_FEE["Fees: €90K"]
        C_NET["Net: €710K ✅"]
    end

    subgraph Moderate["Moderate Portfolio (K=30)"]
        M_SAV["Savings: €950K"]
        M_FEE["Fees: €180K"]
        M_NET["Net: €770K ✅✅"]
    end

    subgraph Aggressive["Aggressive Portfolio (K=100)"]
        A_SAV["Savings: €1,000K"]
        A_FEE["Fees: €600K"]
        A_NET["Net: €400K ❌"]
    end

    Conservative ---|"Marginal gains<br/>diminish"| Moderate
    Moderate ---|"Fees grow<br/>linearly"| Aggressive

    style C_NET fill:#dcfce7,stroke:#16a34a
    style M_NET fill:#dcfce7,stroke:#16a34a
    style A_NET fill:#fee2e2,stroke:#dc2626
```

Savings exhibit diminishing returns (high-value categories are captured first), while fees grow linearly with prediction count. The optimal portfolio size is where marginal savings equal marginal fees — which the economic threshold naturally discovers.

---

## Level 1 → Level 2 → Level 3 Expansion

```mermaid
graph TD
    L1["Level 1: E-Class<br/>2,205 predictions<br/>(buyer_id, eclass)"]

    L1 --> L2_CHECK{"Manufacturer<br/>dominates?<br/>≥90% share<br/>≥15 orders<br/>Recent activity"}

    L2_CHECK -->|"Yes (33%)"| L2["Level 2: E-Class + Manufacturer<br/>732 upgraded predictions<br/>(buyer_id, eclass_manufacturer)"]
    L2_CHECK -->|"No (67%)"| L2_SKIP["Stay at E-Class only<br/>Avoid speculative<br/>brand predictions"]

    L1 --> L3_PROC["Level 3: Feature Clustering"]

    subgraph Clustering["Feature Clustering Pipeline"]
        direction TB
        SKU_FEAT["Load SKU features<br/>(decoded German chars)"]
        TFIDF["TF-IDF per SKU:<br/>key=value tokens"]
        SILO["Select k ∈ {2..6}<br/>via silhouette score"]
        ASSIGN["Assign clusters via<br/>MiniBatchKMeans"]
    end

    L3_PROC --> Clustering

    Clustering --> L3_WARM["Warm: modal cluster<br/>from historical SKUs"]
    Clustering --> L3_COLD["Cold: highest-volume<br/>cluster"]

    L3_WARM --> L3["Level 3: E-Class + Cluster ID<br/>2,205 rows"]
    L3_COLD --> L3

    style L1 fill:#dbeafe,stroke:#3b82f6
    style L2 fill:#fef3c7,stroke:#f59e0b
    style L3 fill:#dcfce7,stroke:#16a34a
    style L2_SKIP fill:#f3f4f6,stroke:#9ca3af
```

**Level 2 philosophy:** Manufacturer specificity is only added when evidence is overwhelming. A 90% dominance threshold with 15+ orders and recent activity ensures the manufacturer prediction reflects actual procurement behavior, not historical accident.

**Level 3 philosophy:** SKU-level features are noisy (suppliers change, catalogs evolve), but *functional need* is stable. TF-IDF clustering captures "nitrile gloves size L" vs "nitrile gloves size M" as distinct demand patterns, even as specific product codes change across contracts.

---

## Backtesting & Calibration

```mermaid
graph TD
    subgraph TrainWindow["Training Window"]
        TRAIN["Jan 2023 — Dec 2024<br/>24 months of transactions"]
    end

    subgraph ValWindow["Validation Window"]
        VAL["Jan 2025 — Jun 2025<br/>6-month holdout"]
    end

    TRAIN --> FE["Build features<br/>on training data"]
    FE --> PREDICT["Generate predictions<br/>for each (sr, fee, tm)"]
    VAL --> TRUTH["Ground truth:<br/>actual purchases"]
    PREDICT --> SCORE["Score = Σ(sr × spend_hit) − fee × n_preds"]
    TRUTH --> SCORE

    SCORE --> GRID["Grid search:<br/>savings_rate ∈ {0.05..0.15}<br/>fee ∈ {3..15}<br/>threshold_mult ∈ {0..3}"]
    GRID --> BEST["Best config by<br/>backtest score"]
    GRID --> ROBUST["Robustness check:<br/>min score across<br/>plausible (sr, fee) combos"]

    ROBUST --> FINAL["Final params:<br/>sr=0.10, fee=€10<br/>tm=robust optimum"]

    style TrainWindow fill:#fef3c7,stroke:#f59e0b
    style ValWindow fill:#dbeafe,stroke:#3b82f6
    style ROBUST fill:#f3e8ff,stroke:#8b5cf6
```

**Robustness analysis:** The true scoring formula's parameters (savings rate, fee) are partially opaque. Rather than optimizing for a single assumed (sr, fee) pair, the system evaluates each threshold across a *plausible range* of parameters and selects the threshold that maximizes the **worst-case** score. This guards against overfitting to incorrect assumptions about the scoring function.

**Cold-start backtest:** 20% of known training buyers are held out as simulated cold-start buyers. Industry profiles are built from the remaining 80%, and cold predictions are evaluated against the held-out buyers' actual purchasing behavior. This provides direct evidence that the cold-start strategy generalizes.

---

## Why This Approach Won on Level 1

1. **Economic framing, not accuracy framing.** The system directly optimizes for net savings minus fees, not precision/recall/F1. This aligns the model with the actual objective.

2. **Recurrence as the primary signal.** Categories bought across 6+ distinct months are almost certain to continue. This signal has 92.8% precision — far better than ML-based prediction alone.

3. **Robustness over point optimization.** Instead of tuning to one assumed scoring formula, the system optimizes for worst-case performance across plausible parameters.

4. **Portfolio discipline.** Conservative cold-start portfolios (15 items), strict manufacturer thresholds (90% dominance), and economic cutoffs prevent fee waste.

5. **Spending velocity ranking.** Recent spending rate, not just historical volume, determines which categories to prioritize — capturing demand shifts.

6. **Adaptive fallbacks.** Per-buyer floors, hierarchical NACE lookups, and CF blending ensure every buyer gets reasonable predictions even with sparse data.

---

## Project Structure

```
core-demand-hackathon/
├── README.md                              # This file
├── report.md                              # Concise methodology report
├── requirements.txt                       # Python dependencies
├── optimize_v4.py                         # Production pipeline (8-phase system)
├── data_visualization/
│   ├── visualise.py                       # 5 analytical figures
│   ├── fig1_long_tail_core_demand.png     # Pareto curve + long tail analysis
│   ├── fig2_dataset_overview.png          # Dataset statistics (4-panel)
│   ├── fig3_recurrence_analysis.png       # Recurrence vs spend patterns
│   ├── fig4_cold_start_industry_profiles.png  # NACE sector demand profiles
│   └── fig5_portfolio_tradeoff.png        # Optimal portfolio size analysis
├── json_approach/
│   ├── v15_surgical.py                    # LightGBM surgical tuning (25 features)
│   └── v17_gameplan.py                    # 3-lever optimization variant
├── docs/
│   ├── architecture.md                    # System architecture deep dive
│   ├── methodology.md                     # Full methodology explanation
│   ├── economic-optimization.md           # Economic reasoning and tradeoffs
│   ├── results.md                         # Results, calibration, and analysis
│   └── interview-notes.md                # CV bullets + interview preparation
├── README_Core_Demand_Challenge.md        # Original challenge specification
└── LICENSE                                # MIT License
```

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

### Data Files (not included — competition-provided)
Place in the repo root:
- `plis_training.csv.gz` — 8.37M transactions (tab-separated, UTF-8 BOM, gzipped)
- `customer_test.csv.gz` — 100 test buyers
- `features_per_sku.csv.gz` — SKU-level feature metadata
- `nace_codes.csv.gz` — NACE industry classification codes

### Run
```bash
# Generate Level 1 + Level 2 submissions
python optimize_v4.py

# Generate visualizations
cd data_visualization && python visualise.py
```

### Output
- `submission_lvl1.csv` — Level 1 predictions (buyer_id, predicted_id)
- `submission_lvl2.csv` — Level 2 manufacturer-specific predictions

---

## Visualizations

The visualization suite produces 5 figures that explain the problem structure and solution rationale:

| Figure | What It Shows |
|--------|---------------|
| **Long Tail vs Core Demand** | Pareto curve: ~15-20% of E-Classes account for ~80% of spend |
| **Dataset Overview** | Monthly volume trends, price distributions, NACE sector breakdown |
| **Recurrence Analysis** | Months-active distribution, spend vs recurrence scatter, economic signal comparison |
| **Cold-Start Industry Profiles** | Top E-Classes per NACE sector — the basis for industry-profile predictions |
| **Portfolio Trade-off** | Cumulative savings vs fees vs net score — shows optimal portfolio size |

---

## Limitations

- **Level 2 and Level 3 were not optimized for leaderboard competition** — the primary effort was on Level 1 where the economic impact is largest.
- **Cold-start predictions are inherently lower precision** — industry priors are useful but cannot match buyer-specific transaction history.
- **The true scoring formula was partially opaque** — robustness analysis mitigates this, but the optimal parameters may differ from assumptions.
- **No ensemble of ML models was used** — the winning approach is primarily rule-based (recurrence + economic thresholds), which proved more robust than gradient-boosted models in this setting.
- **SKU-level feature clustering (Level 3)** uses unsupervised methods; cluster quality depends on feature coverage in the source data.

---

## Future Work

- **Industry expansion for warm buyers** — applying cold-start NACE-matching to warm buyers could surface peer-sourced portfolio gaps (categories that similar companies buy but the target buyer doesn't yet).
- **Prediction source SWAP** — when multiple scoring methods disagree, replacing weak predictions with higher-confidence alternatives from collaborative filtering could improve quality without increasing portfolio size.
- **Association rule mining** — co-purchase patterns (buyers who buy X frequently also buy Y) could provide cross-category expansion signals.
- **Temporal cross-validation** — rolling-window backtesting across multiple time periods would strengthen calibration confidence.
- **Per-buyer threshold optimization** — different buyers may have different optimal threshold multipliers based on their purchasing behavior distribution.


---

## License

MIT — see [LICENSE](LICENSE).
