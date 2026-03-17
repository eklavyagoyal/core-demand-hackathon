# Methodology

## Problem Formulation

Given 100 enterprise buyers, predict which product categories (E-Classes) represent their **recurring procurement demand** (Core Demand), such that the portfolio maximizes:

$$\text{Score} = \sum_{i \in \text{correct predictions}} \text{Savings}_i \;-\; \sum_{j \in \text{all predictions}} \text{Fee}_j$$

Where:
- Savings scale approximately as $\sqrt{\text{price}} \times \text{frequency}$ for correctly identified recurring categories
- Each prediction incurs a fixed monthly fee regardless of correctness
- Savings rate (~10%) and fee (~€10) are known approximately but not exactly

The 100 buyers split into:
- **47 warm-start**: 2+ years of transaction history available
- **53 cold-start**: zero transaction history; only company profile (NACE code, employee count, country)

---

## Warm-Start Methodology

### Core Insight: Recurrence Is the Strongest Signal

A product category bought across many *distinct calendar months* almost certainly represents ongoing, structural demand — not a one-off purchase. This signal is far more predictive than total spend or order count alone.

### Feature Engineering

For each (buyer, E-Class) pair, three recurrence features are computed:

| Feature | Definition | Intuition |
|---------|------------|-----------|
| `n_months_active` | Distinct calendar months with ≥1 purchase | Raw recurrence measure |
| `n_months_recent` | Active months in Jan–Jun 2025 | Is this demand still alive? |
| `n_months_weighted` | `(n_months_active - n_months_recent) + 2 × n_months_recent` | Up-weights recent activity |

Three spending features provide economic context:

| Feature | Definition | Intuition |
|---------|------------|-----------|
| `avg_monthly_spend` | `recent_spend / 6` | Spending velocity in recent window |
| `recency_ratio` | `recent_spend / total_spend` | What fraction of lifetime spend is recent? |
| `expected_val_spend` | `avg_monthly_spend × prediction_months` | Projected spend in prediction horizon |

A composite **rank score** combines velocity with recency momentum:

$$\text{rank\_score} = \text{avg\_monthly\_spend} \times (1 + \text{recency\_ratio})$$

### Economic Threshold

Instead of a naive top-k cutoff, predictions are included based on *expected economic contribution*:

$$\text{Include if: } \text{savings\_rate} \times \text{expected\_val\_spend} > \text{fee} \times \text{threshold\_mult}$$

This means:
- High-spend, recurring categories pass easily
- Low-spend or stale categories are filtered proportionally to their expected fee burden
- The threshold adapts to the assumed (savings_rate, fee) parameters

### Per-Buyer Adaptive Floor

Even if all of a buyer's candidates fall below the economic threshold, the system guarantees ≥1 prediction per buyer. The fallback selects the highest rank_score candidate. This prevents the degenerate case of predicting zero categories for an active buyer.

---

## Cold-Start Methodology

### Challenge

52 buyers have zero transaction history. The only available signals are:
- **NACE code**: industry classification (2 to 5 digits)
- **Employee count**: company size
- **Country**: registration country (limited signal for procurement patterns)

### Hierarchical Industry Profile Matching

The cold-start engine builds E-Class demand profiles at multiple NACE aggregation levels:

**Profile construction:**
For each NACE group, compute per E-Class:
- `penetration` = fraction of buyers in that NACE group who purchased this E-Class
- `avg_value` = average transaction value
- `score` = `penetration × √(avg_value)` — balances frequency with economic significance

**Lookup chain:**
1. **NACE 4-digit + size bucket**: Most specific. Requires ≥3 peers in the same NACE code and employee size bracket.
2. **NACE 4-digit (any size)**: Falls back if size-bucketed group is too small.
3. **NACE 2-digit**: Broader industry sector (e.g., all manufacturing vs just "manufacture of electronic components").
4. **Global popularity**: Final fallback using all training buyers.

At each level, predictions are filtered by the cold-start economic threshold (calibrated via held-out backtest).

### Spend-Weighted Collaborative Filtering

In parallel with the NACE lookup, a collaborative filtering (CF) module provides complementary signals:

1. **Build sparse interaction matrix**: buyer × E-Class, weighted by `log1p(spend_per_month)`
2. **Filter to recurring items**: Only include (buyer, E-Class) pairs with `n_months ≥ 2`
3. **Find NACE-matched peers**: For a cold buyer, identify training buyers in the same 2-digit NACE group
4. **Cosine similarity**: Compute similarity between peer group centroid and individual peers
5. **Weighted aggregation**: Top-10 similar peers contribute their E-Class patterns, weighted by similarity
6. **Size boost**: Peers in the same employee size bucket get a +0.2 similarity boost

### Source Merging

NACE profile predictions and CF predictions are interleaved and deduplicated. NACE profiles take priority (more reliable for industry-generic demand), with CF filling gaps (captures peer-specific patterns that NACE averages might miss).

**Cold portfolio size:** 15 E-Classes per buyer — deliberately conservative given the lack of buyer-specific history.

---

## Calibration & Backtesting

### Warm-Start Calibration

**Temporal split:**
- Training: Jan 2023 – Dec 2024 (24 months)
- Validation: Jan 2025 – Jun 2025 (6 months)

**Grid search space:**
- `savings_rate` ∈ {0.05, 0.08, 0.10, 0.12, 0.15}
- `fee` ∈ {€3, €5, €7.50, €10, €15}
- `threshold_mult` ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
- **Total: 200 configurations**

Each configuration is evaluated using the real economic scoring function:

$$\text{Score} = \sum_{\text{hits}} sr \times \text{spend}_i \;-\; fee \times n_{\text{preds}}$$

### Robustness Analysis

The true scoring parameters are partially opaque. Rather than betting on a single (sr, fee) assumption:

1. For each `threshold_mult`, compute the score across all *plausible* (sr, fee) combinations (sr ∈ [0.05, 0.15], fee ∈ [€5, €15])
2. Take the **minimum** score across this range — the worst-case outcome
3. Select the `threshold_mult` that maximizes this worst-case score

This produces a threshold that performs reasonably well regardless of the exact scoring formula — a deliberate choice of robustness over point optimization.

### Cold-Start Calibration

A separate backtest validates cold-start predictions:

1. Hold out 20% of training buyers (randomly sampled, seed=42)
2. Build NACE profiles and CF index from the remaining 80%
3. Treat held-out buyers as simulated cold-start (predict using industry profile only)
4. Evaluate against their actual transaction history
5. Select the cold threshold_mult that performs best at the submission (sr, fee) parameters

---

## Level 2: Manufacturer Predictions

For each Level 1 (E-Class) prediction, check whether one manufacturer dominates that buyer's purchases in that category:

**Inclusion criteria (all must be met):**
- Manufacturer accounts for ≥90% of orders in that (buyer, E-Class) pair
- ≥15 total orders with that manufacturer in that E-Class
- The (buyer, E-Class, manufacturer) triple appears in the recent period (Jan 2025+)

**Result:** 33% of Level 1 predictions upgraded to Level 2 (E-Class + manufacturer). The remaining 67% stay E-Class-only — deliberately conservative to avoid fee waste on uncertain brand predictions.

---

## Level 3: Feature Clustering

For each target E-Class, cluster the associated SKUs by their product features to identify distinct functional demand patterns:

1. **Feature decoding:** URL-encoded German characters in feature values are decoded (`(e4)` → ä, `(f6)` → ö, etc.)
2. **TF-IDF vectorization:** Each SKU's feature set is tokenized as `key=value` strings, then TF-IDF weighted
3. **Cluster count selection:** MiniBatchKMeans with k ∈ {2, 3, 4, 5, 6}, best k selected by silhouette score
4. **Assignment:**
   - Warm buyers: Map historical SKUs to clusters, take the **modal** cluster (most common functional pattern)
   - Cold buyers: Use the **highest-volume** cluster (most commonly purchased specification)

**Why this works:** Individual SKUs change across suppliers and contracts, but functional needs are stable. A buyer who consistently purchases "nitrile gloves, size L, powder-free" will continue to need that specification regardless of which supplier or SKU code fulfills it.

---

## Design Decisions & Tradeoffs

| Decision | Alternative | Why This Choice |
|----------|-------------|-----------------|
| Recurrence features (not ML) for warm scoring | LightGBM (tested in V15) | Recurrence achieves 92.8% precision; ML added complexity without improving net score |
| Economic threshold (not top-k) | Fixed K per buyer | Dynamic threshold aligns with the economic objective; top-k ignores cost |
| Robustness optimization | Point optimization at assumed params | True params are opaque; worst-case optimization is safer |
| 90% dominance for L2 | 50-60% threshold | Higher threshold prevents fee waste; wrong manufacturer prediction is pure cost |
| 15 items for cold buyers | Same as warm (~30) | Zero history means lower confidence; conservative portfolio avoids fee death |
| NACE profile + CF blend | Pure CF or pure NACE | NACE captures industry structure; CF captures peer-specific patterns; blend is stronger |
| Recency 2× weighting | Equal weighting across time | Demand is non-stationary; recent activity is more predictive of future behavior |
