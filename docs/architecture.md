# System Architecture

## Design Philosophy

This system is not a recommendation engine. It is an **economic decision system** that predicts which procurement categories will recur for a given buyer, subject to the constraint that each prediction carries a fixed monthly fee.

The architecture reflects three principles:
1. **Economic alignment** — every decision uses the objective function (savings − fees), not proxy metrics
2. **Robustness over precision** — optimize for worst-case performance across uncertain parameters
3. **Signal separation** — warm-start and cold-start are fundamentally different problems requiring distinct approaches

---

## System Diagram

```mermaid
graph TB
    subgraph DataLayer["Data Layer"]
        PLI[Transaction Data<br/>8.37M rows, 36 months]
        CUST[Customer Test Set<br/>100 buyers]
        NACE_DATA[NACE Codes<br/>Industry classification]
        SKU_DATA[SKU Features<br/>Product specifications]
    end

    subgraph Preprocessing["Preprocessing"]
        CLEAN[Data Cleaning<br/>• E-Class normalization<br/>• SKU-based imputation<br/>• Total value computation]
        SPLIT[Buyer Classification<br/>warm vs cold routing]
    end

    subgraph WarmEngine["Warm-Start Engine"]
        direction TB
        WFE[Feature Engineering<br/>n_months_active, recent_spend<br/>avg_monthly_spend, recency_ratio]
        WRANK[Economic Scoring<br/>rank_score = velocity × (1 + recency)]
        WTHRESH[Threshold Filter<br/>sr × expected_spend > fee × tm]
        WFLOOR[Adaptive Floor<br/>≥1 prediction per buyer]
    end

    subgraph ColdEngine["Cold-Start Engine"]
        direction TB
        NACE_PROF[NACE Profile Builder<br/>penetration × √avg_value]
        CF_ENGINE[Collaborative Filtering<br/>Sparse cosine similarity<br/>log1p spend weights]
        HIER[Hierarchical Fallback<br/>4-digit → 2-digit → global]
        MERGE[Source Merge<br/>NACE first, CF second]
    end

    subgraph CalibrationLayer["Calibration Layer"]
        BACKTEST[Warm Backtest<br/>Train 2023-2024<br/>Val Jan-Jun 2025]
        GRID[Grid Search<br/>320 configurations<br/>sr × fee × tm]
        ROBUST[Robustness Selection<br/>max(min score across<br/>plausible params)]
        COLD_BT[Cold Backtest<br/>20% held-out buyers]
    end

    subgraph LevelExpansion["Level Expansion"]
        L2_ENGINE[Level 2 Engine<br/>Manufacturer dominance<br/>90% share, 15+ orders, recent]
        L3_ENGINE[Level 3 Engine<br/>TF-IDF + KMeans<br/>Silhouette-optimal k]
    end

    subgraph Output["Output"]
        SUB1[submission_lvl1.csv<br/>2,205 rows]
        SUB2[submission_lvl2.csv<br/>732 manufacturer pairs]
        DIAG[Diagnostic Summary<br/>Per-buyer stats, distributions]
    end

    PLI --> CLEAN --> SPLIT
    CUST --> SPLIT
    SPLIT -->|warm| WarmEngine
    SPLIT -->|cold| ColdEngine
    NACE_DATA --> ColdEngine
    SKU_DATA --> L3_ENGINE

    WFE --> WRANK --> WTHRESH --> WFLOOR
    NACE_PROF --> HIER --> MERGE
    CF_ENGINE --> MERGE

    BACKTEST --> GRID --> ROBUST
    ROBUST -->|thresholds| WTHRESH
    COLD_BT -->|cold_tm| ColdEngine

    WFLOOR --> L2_ENGINE --> L3_ENGINE
    MERGE --> L2_ENGINE

    L3_ENGINE --> SUB1
    L3_ENGINE --> SUB2
    L3_ENGINE --> DIAG

    style DataLayer fill:#e8f4f8,stroke:#2563eb
    style WarmEngine fill:#fef3c7,stroke:#f59e0b
    style ColdEngine fill:#dbeafe,stroke:#3b82f6
    style CalibrationLayer fill:#f3e8ff,stroke:#8b5cf6
    style LevelExpansion fill:#dcfce7,stroke:#16a34a
    style Output fill:#f1f5f9,stroke:#64748b
```

---

## Component Responsibilities

### Data Layer
- **Input parsing**: Tab-separated, UTF-8 BOM, gzipped CSVs
- **E-Class normalization**: Handle missing values via SKU lookup (if a SKU was previously seen with a known E-Class, impute it)
- **Value computation**: `total_value = vk_per_item × quantityvalue`

### Buyer Router
Classification is straightforward:
- `task == "predict future"` → warm-start
- `task == "cold start"` → cold-start
- **Edge case**: buyer 61933687 is labeled warm but has zero transaction history → rerouted to cold-start

### Warm-Start Engine
1. **Feature engineering** (`build_warm_features`): Aggregates per (buyer, E-Class) pair. Computes recurrence across calendar months, spending velocity in the recent 6-month window, and a composite rank score.
2. **Economic threshold** (`predict_warm`): Includes a pair only if expected savings exceed fee × threshold_mult. This replaces fixed top-k with a dynamic, cost-aware cutoff.
3. **Adaptive floor**: Every buyer gets ≥1 prediction, even if all candidates fall below threshold. Fallback selects the highest rank_score candidate.

### Cold-Start Engine
1. **NACE profile builder** (`build_nace_profile`): For each NACE group, computes E-Class penetration (what fraction of buyers in that industry buy this category) and expected value (√avg_value weighting).
2. **Hierarchical lookup** (`cold_predict_buyer`): Tries 4-digit NACE → 2-digit NACE → global fallback. Requires ≥3 peers for statistical significance.
3. **Collaborative filtering** (`get_cf_eclasses_fast`): Builds sparse buyer×E-Class matrix weighted by log1p(spend_per_month). Finds NACE-matched peers via cosine similarity. Aggregates their purchase patterns.
4. **Source merge**: NACE profile predictions interleaved with CF predictions, deduplicated.

### Calibration Layer
1. **Temporal backtest**: Train on 2023-2024, validate on Jan-Jun 2025. Ensures no data leakage.
2. **Grid search**: 5 savings_rates × 5 fees × 8 threshold_mults = 200 warm configurations evaluated.
3. **Robustness analysis**: For each threshold_mult, compute min score across plausible (sr, fee) range. Select the threshold that maximizes worst-case performance.
4. **Cold backtest**: Hold out 20% of training buyers. Build NACE profiles from remaining 80%. Evaluate cold predictions against held-out truth.

### Level Expansion
1. **Level 2**: Manufacturer added only when one manufacturer accounts for ≥90% of orders in that E-Class for that buyer, with ≥15 orders and recent activity. Designed to avoid fee waste on uncertain brand predictions.
2. **Level 3**: TF-IDF vectorization of SKU feature strings, clustered via MiniBatchKMeans with silhouette-optimal k ∈ {2..6}. Warm buyers get their modal cluster; cold buyers get the highest-volume cluster.

---

## Data Flow Summary

| Stage | Input | Output | Key Decision |
|-------|-------|--------|--------------|
| Preprocessing | Raw CSVs | Cleaned DataFrame | E-Class imputation via SKU lookup |
| Warm Features | Transactions | (buyer, eclass) feature matrix | Recency weighting (2×) |
| Grid Search | Feature matrix + val truth | Optimal (sr, fee, tm) | Worst-case robustness |
| Warm Predict | Features + thresholds | Warm L1 predictions | Economic threshold + floor |
| Cold Profile | All transactions + NACE | Industry E-Class profiles | Penetration × √value scoring |
| Cold CF | Sparse matrix | CF E-Classes per buyer | log1p spend weights, cosine similarity |
| Cold Predict | Profiles + CF | Cold L1 predictions | Hierarchical fallback, merge |
| Level 2 | L1 + manufacturer data | L2 predictions | 90% dominance + recency gate |
| Level 3 | L1 + SKU features | L3 predictions | Silhouette-optimal clustering |

---

## Iteration History

The repository contains three pipeline versions reflecting the evolution of the approach:

| Version | File | Approach | Key Innovation |
|---------|------|----------|----------------|
| V15 | `json_approach/v15_surgical.py` | LightGBM with 25 features, surgical tuning | ML-based scoring with expected value thresholds |
| V17 | `json_approach/v17_gameplan.py` | 3-lever optimization (smarter warm, twin matching, spend ranking) | Personalized cold-start via buyer "twins" |
| V4 | `optimize_v4.py` | 8-phase economic optimization | Robustness analysis, spend-weighted CF, economic thresholds |

The final V4 pipeline represents the most rigorous approach — built on learnings from V15/V17 but replacing ML-based scoring with simpler, more robust recurrence-based signals + economic thresholds.
