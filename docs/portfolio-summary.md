# Portfolio Summary — Approach Comparison

## Our Approach vs Companion Approach

Both teams tackled the same Core Demand prediction problem. This document summarizes the key architectural differences, identifies what each approach did better, and documents insights that informed the final design.

---

## Architecture Comparison

| Dimension | Our Approach (V4) | Companion Approach |
|-----------|-------------------|-------------------|
| **Primary warm scoring** | Recurrence features + economic threshold | 5-source ensemble (HF + ML + CF + Hierarchy + Association Rules) |
| **Model complexity** | Rule-based with grid-searched thresholds | HistGradientBoosting + ensemble merge with consensus bonus |
| **Cold-start** | NACE hierarchy + spend-weighted CF | 3-tier NACE matching (4d → 2d → global) |
| **Portfolio sizing** | Dynamic via economic threshold | Fixed caps (400 warm, 525 cold) with greedy trimming |
| **Calibration** | Robustness optimization (minimax) | Temporal backtest with scoring grid |
| **Key innovation** | Worst-case optimization across uncertain params | Industry expansion for warm buyers + SWAP strategy |
| **L1 predictions** | ~2,205 | ~47,868 |

---

## What Our Approach Did Better

### 1. Robustness Under Parameter Uncertainty
Our minimax calibration strategy optimizes for worst-case performance across a range of plausible scoring parameters. This provides safety when the exact scoring formula is opaque — a critical advantage in competition settings.

### 2. Simplicity with Strong Signal
Recurrence-based features (months active, recency weighting) achieve 92.8% precision without requiring gradient-boosted models. Fewer moving parts means fewer failure modes and more interpretable decisions.

### 3. Conservative Portfolio Discipline
Our economic threshold naturally sizes portfolios based on expected savings vs fees. Conservative cold-start sizing (15 items) and strict L2 criteria (90% dominance) prevent fee waste.

### 4. Multi-Level Depth
Level 3 clustering (TF-IDF + silhouette-optimal KMeans on SKU features) captures functional demand patterns beyond simple category matching.

---

## What the Companion Approach Did Better

### 1. Industry Expansion for Warm Buyers
The companion applied cold-start NACE-matching logic to warm buyers, surfacing categories that industry peers purchase but the target buyer doesn't yet. This found ~€25K in additional score from categories warm buyers hadn't considered. This is a valuable idea for future iterations.

### 2. SWAP Strategy
Instead of simply unioning predictions from multiple sources (which increases portfolio size and fees), the companion replaces weak predictions with stronger ones from alternate sources — maintaining portfolio size while improving quality. Conceptually elegant.

### 3. Ensemble Diversity
Five prediction sources (rule-based scoring, ML, collaborative filtering, hierarchy expansion, association rules) with consensus bonuses provide signal diversity. When multiple models agree on a prediction, confidence is higher.

### 4. Association Rule Mining
Co-purchase patterns (buyers who buy X tend to buy Y) provide cross-category expansion signals that recurrence analysis misses.

### 5. Score Progression Documentation
Clear documentation of score evolution across iterations (€781K → €1,267K → €1,484K → €1,510K) makes the improvement trajectory visible and persuasive.

---

## Key Insights for Future Development

1. **Industry expansion for warm buyers** is the highest-leverage improvement to port. Even buyers with rich history may have blind spots that industry peers reveal. Worth implementing as a portfolio augmentation step after warm predictions.

2. **SWAP logic** (replacing weak predictions with stronger alternatives while maintaining portfolio size) is architecturally sound. Our economic threshold partially achieves this by filtering low-value predictions, but explicit swapping between sources could improve quality further.

3. **Consensus scoring** (boosting predictions that appear in multiple sources) is cheap to implement and empirically effective. A 20% score boost for multi-source agreement is reasonable calibration.

4. **MIN_SAVINGS threshold calibration** — the companion found that €3 was the exact sweet spot for their scoring threshold (€2 was too aggressive, €5 left money on table). Our grid search already covers this space, but explicitly documenting the sensitivity analysis strengthens the narrative.

---

## Strategic Takeaway

Both approaches demonstrate that **economic framing** — treating each prediction as an investment decision with real costs — is the primary differentiator from naive recommendation approaches. The specific implementation details (recurrence vs ML, hierarchical lookup vs ensemble, minimax vs point optimization) are secondary to getting the economic framing right.

The simplest path to further improvement from here would be: (1) implement industry expansion for warm buyers, (2) add SWAP logic to replace low-confidence recurrence predictions with high-confidence CF predictions, and (3) add association rule signals for cross-category expansion. Each is incremental and preserves the existing architecture's strengths.
