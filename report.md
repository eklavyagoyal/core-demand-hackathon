# Core Demand Prediction — Methodology Report

> **Result: #1 on Level 1 leaderboard** — achieved through economically calibrated scoring, robust threshold optimization, and disciplined portfolio construction.

## Problem Summary

For 100 buyers (47 warm-start, 53 cold-start), predict which product categories (E-Classes) represent their recurring Core Demand, such that the portfolio maximises `Sum(Savings) - Sum(Fees)`. This is a constrained portfolio optimization problem, not a standard recommendation task — every prediction incurs a fee, making precision and economic alignment critical.

---

## Approach

### Warm-Start Buyers (47 buyers + 1 edge case)

**Signal:** Purchase recurrence across calendar months. A buyer who buys in the same E-Class across many distinct months is highly likely to continue doing so.

**Feature engineering per (buyer, eclass) pair:**
- `n_months_active`: distinct calendar months with at least one purchase
- `n_months_recent`: months active in Jan–Jun 2025 (last 6 months of observation window)
- `n_months_weighted = (n_months_active - n_months_recent) + 2 × n_months_recent` — recent activity weighted double to capture ongoing demand

**Calibration via backtesting:**
We used Jan 2023–Dec 2024 as training and Jan–Jun 2025 as validation. Grid search over:
- `month_threshold` ∈ {1, 2, 3, 4, 5, 6}
- `min_spend` ∈ {0, 50, 100, 200, 500}
- `max_k` (portfolio cap) ∈ {5, 10, 15, 20, 30}

Optimal config: `month_threshold=6, max_k=30, recency_weighted=True`, achieving **92.8% precision** on the held-out validation set (1297/1397 correct predictions).

**Portfolio discipline:** Predictions capped at `max_k=30`. Sorted by total spend to prioritise high-value categories.

One warm buyer (ID 61933687) appeared in the test set as "predict future" but had zero training history; treated as cold-start.

---

### Cold-Start Buyers (52 buyers)

**No transaction history** — predictions rely on industry signals and buyer profiles.

**Hierarchical lookup strategy:**
1. **Size-bucketed NACE profile**: Match buyer to peers with same 2-digit NACE code AND same employee size bucket (1-10, 11-50, 51-250, 251-1000, 1000+). Score = `penetration × √(avg_value)`.
2. **4-digit NACE fallback**: If size bucket has <10 peers, use full NACE group.
3. **2-digit NACE fallback**: Broader industry sector.
4. **Collaborative filtering blend**: Find top-5 training buyers with similar NACE+size profile using cosine similarity on their eclass purchase vectors. Interleave CF predictions with NACE profile predictions.
5. **Global popularity**: Final fallback if no NACE data.

Top 15 eclasses per cold buyer are predicted.

---

### Portfolio Trade-off

The scoring function penalises over-prediction via fixed fees. Key principles applied:
- Warm buyers: require 6+ recurrence months (high confidence threshold)
- Cold buyers: 15 eclasses — conservative given zero history
- Level 2 predictions only added when manufacturer dominance is clear (>60% of orders for warm, >50% penetration in NACE peers for cold) — avoiding speculative brand predictions

---

## Level 2 — E-Class + Manufacturer

For each Level 1 (eclass) prediction, we check whether one manufacturer dominates:
- **Warm buyers**: dominant if manufacturer accounts for >60% of orders in that eclass
- **Cold buyers**: dominant if manufacturer appears in >50% of NACE peers' purchases

33% of Level 1 predictions were upgraded to Level 2 (eclass_manufacturer). The remaining 67% stay at eclass-only to avoid fee waste on uncertain manufacturer predictions.

---

## Level 3 — Feature Clustering

**Methodology:**
1. Load SKU-level features from `features_per_sku.csv.gz` (decoded URL-encoded German chars: `(e4)`→ä, `(f6)`→ö, `(fc)`→ü, `(df)`→ß)
2. For each target eclass, build a TF-IDF document per SKU: `key=value` tokens from `fvalue_set` (normalised values preferred over raw `fvalue`)
3. Select optimal cluster count k ∈ {2..6} via silhouette score on MiniBatchKMeans
4. **Warm buyers**: map their historically purchased SKUs to clusters, take the modal cluster
5. **Cold buyers**: use the highest-volume cluster as the most common functional need

**Rationale:** Level 3 clusters capture "functional product specifications" (e.g. nitrile gloves size L vs size M, single-phase vs three-phase transformers) that are stable even as specific SKUs change across suppliers.

---

## Summary of Predictions

| Segment     | Buyers | Avg eclasses/buyer | Method |
|-------------|--------|--------------------|--------|
| Warm        | 47     | 30                 | Recurrence threshold (6 months, recency-weighted) |
| Cold        | 52     | 15                 | NACE size-profile + CF blend |
| Warm (edge) | 1      | 15                 | Cold-start fallback |

**Total Level 1:** 2205 predictions across 100 buyers
**Total Level 2:** 2205 rows (732 upgraded to eclass_manufacturer)
**Total Level 3:** 2205 rows (cluster IDs derived from feature clustering)

---

## Files

| File | Description |
|------|-------------|
| `optimize_v4.py` | Production pipeline — 8-phase economic optimization (warm + cold + L2) |
| `json_approach/v15_surgical.py` | LightGBM variant with 25 features, surgical EV tuning |
| `json_approach/v17_gameplan.py` | 3-lever optimization: smarter warm + twin matching + spend ranking |
| `data_visualization/visualise.py` | 5 analytical figures (Pareto, recurrence, industry profiles, tradeoffs) |
| `docs/architecture.md` | System architecture deep dive |
| `docs/methodology.md` | Full methodology and design decisions |
| `docs/economic-optimization.md` | Economic reasoning and tradeoff analysis |
| `docs/results.md` | Results, calibration, and pipeline evolution |
| `docs/interview-notes.md` | CV bullets and interview preparation |

For full details, see the [README](README.md) and [docs/](docs/) folder.
