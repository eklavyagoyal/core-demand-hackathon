# Interview & CV Material

## Resume Bullets

Use 4–6 of these depending on the target role:

**For ML / Data Science roles:**
> - Designed and deployed an economically calibrated procurement demand prediction system that ranked **#1 on Level 1** in a competitive optimization hackathon across 100 enterprise buyers (47 warm-start, 52 cold-start)
> - Engineered a recurrence-based feature pipeline achieving **92.8% precision** on temporal backtest, outperforming gradient-boosted ML models through stronger signal selection and economic threshold calibration
> - Built a hybrid warm-start / cold-start architecture using NACE industry profiles, spend-weighted collaborative filtering (sparse matrix cosine similarity), and hierarchical fallback chains for buyers with zero transaction history
> - Implemented robustness-first calibration: grid search across 200 parameter configurations with worst-case optimization across uncertain scoring parameters, replacing point optimization with minimax strategy
> - Developed multi-level prediction system (category → manufacturer → feature-cluster) with economically motivated promotion gates (90% manufacturer dominance, silhouette-optimal feature clustering)

**For optimization / operations research roles:**
> - Built a portfolio optimization system for B2B procurement that maximizes net economic benefit (savings − fees) under asymmetric cost constraints, ranking #1 among competing teams
> - Designed economic threshold filtering that replaces naive top-k recommendation with dynamically sized portfolios calibrated to expected savings vs marginal fee cost per prediction

**For engineering / systems roles:**
> - Architected an 8-phase prediction pipeline processing 8.37M transactions with spend-weighted collaborative filtering (scipy.sparse), temporal backtesting, and cold-start industry profile matching

---

## LinkedIn / Project Summary

### Short version (for project section)
> **Core Demand Prediction Engine** — Built an economically calibrated system to predict recurring procurement categories for enterprise buyers, ranking #1 on Level 1 in a competitive hackathon. Designed a hybrid warm-start/cold-start architecture with economic threshold filtering, robustness optimization, and multi-level predictions (category → manufacturer → feature specification). Achieved 92.8% warm-start precision through recurrence-based features and disciplined portfolio construction.

### Extended version (for posts / articles)
> In a competitive procurement optimization hackathon, I built a system that predicts which product categories represent a buyer's recurring demand — maximizing net savings while managing per-prediction fees. The key insight was that this isn't a recommendation problem; it's a portfolio optimization problem under asymmetric cost. Every false positive costs money directly.
>
> The system handles 47 warm-start buyers (with transaction history) using recurrence-based features and economic thresholds, and 52 cold-start buyers (with zero history) using industry profiles and collaborative filtering. Calibration uses worst-case optimization across uncertain scoring parameters rather than point optimization at assumed values.
>
> Ranked #1 on Level 1 at the end of the hackathon. The approach worked because it aligned model decisions with economic outcomes rather than optimizing proxy metrics.

---

## Interview Answers

### 1. "Tell me about this project"

> This was a hackathon challenge where I had to predict recurring procurement demand for 100 enterprise buyers. The twist is that each prediction costs money — there's a fixed monthly fee per predicted category. So the objective function is savings minus fees, which fundamentally changes the design.
>
> I split the problem into warm-start buyers with 2+ years of history, where I used purchase recurrence across calendar months as the primary signal, and cold-start buyers with zero history, where I built hierarchical industry profiles using NACE codes and collaborative filtering.
>
> The key innovation was replacing naive top-k recommendations with an economic threshold — each prediction must earn more in expected savings than it costs in fees. I calibrated this with a robustness optimization: instead of assuming specific scoring parameters, I optimized for the worst case across a plausible range.
>
> Finished #1 on Level 1 at the end of the competition. The approach worked because it was economically aligned and robust, not because it used the most complex models.

### 2. "Why did this rank #1 on Level 1?"

> Three reasons.
>
> First, I framed it as an economic optimization problem, not an accuracy problem. Most teams optimized for precision or recall; I optimized directly for savings minus fees. That changes portfolio sizing decisions fundamentally.
>
> Second, recurrence across calendar months turned out to be an incredibly strong signal. Categories bought in 6+ distinct months have over 92% probability of continuing. This simple feature outperformed 25-feature gradient-boosted models.
>
> Third, robustness. The scoring formula's exact parameters were partially opaque. Instead of guessing the right savings rate and fee, I calibrated the system to perform well under the *worst-case* combination of plausible parameters. That meant I didn't blow up if my assumptions were slightly wrong, which is where a lot of teams lost points.

### 3. "How did you handle cold-start buyers?"

> 52 of the 100 buyers had zero transaction history, so I couldn't use any buyer-specific signals. I built a hierarchical fallback system.
>
> First, I tried matching the buyer's NACE code (industry classification) at the 4-digit level and looking at what similar-sized companies in that industry purchase. If there weren't enough peers at that specificity, I fell back to 2-digit NACE, then global popularity.
>
> In parallel, I ran a collaborative filtering module — I built a sparse buyer-by-E-Class matrix weighted by log-transformed spending, and found the most similar buyers in the same industry sector using cosine similarity. Then I merged the NACE profiles and CF recommendations, prioritizing NACE for reliability and CF for coverage.
>
> I deliberately kept cold portfolios small — only 15 categories per buyer vs ~30 for warm buyers. With zero history, every prediction is uncertain, and the fee penalty for wrong predictions is the same. Conservative sizing was a deliberate economic choice.
>
> I validated this with a cold-start backtest: held out 20% of known buyers, built profiles from the remaining 80%, and checked prediction quality against the held-out truth. That gave me confidence the cold logic generalized.

### 4. "How did you think about the economic objective?"

> The objective is savings minus fees. That changes three things compared to a standard recommendation problem.
>
> First, **precision matters more than recall** because a false positive costs money. In a typical recommender, if the user ignores a suggestion, nothing bad happens. Here, every wrong prediction incurs a fee.
>
> Second, **portfolio size has a natural optimum**. Savings follow a diminishing returns curve — you capture the high-value categories first. But fees are linear — every additional prediction costs the same. The optimal point is where marginal savings equal marginal cost. I encode this directly through the economic threshold rather than using arbitrary top-k.
>
> Third, **parameter uncertainty matters**. The exact savings rate and fee aren't fully known. I couldn't just optimize for one assumption. So I ran a robustness analysis: evaluate every threshold across a range of plausible scoring parameters, and pick the threshold that maximizes the worst-case outcome. This is a minimax strategy from decision theory — it sacrifices some best-case performance for safety under uncertainty.
>
> The economic framing also drove my cold-start decisions. I could have given cold buyers 50+ predictions to maximize coverage. But each prediction has only ~15-20% chance of being right for a cold buyer, versus ~93% for warm buyers. At 15 items, the expected net contribution is still positive. At 50, the fees overwhelm the likely savings.

### 5. Deeper technical answer — "Walk me through the calibration and robustness approach"

> Sure. The calibration has two stages: warm and cold.
>
> For warm, I split the data temporally — train on 2023-2024, validate on the first half of 2025. I compute features on the training window and evaluate predictions against actual purchases in the validation window. The evaluation uses the economic scoring function: sum of savings minus fees.
>
> I then do a grid search over three parameters: savings_rate (our estimate of how much the buyer saves per correct prediction), fee (the cost per prediction), and threshold_mult (how conservative to be). That's 5 × 5 × 8 = 200 configurations. For each, I measure the backtest score.
>
> Now here's the key insight: I don't just pick the best-scoring configuration. The savings_rate and fee are external parameters that I don't actually control or know precisely. So for each threshold_mult value, I compute the score at *every* plausible (savings_rate, fee) combination, and take the minimum — the worst case. Then I pick the threshold_mult that maximizes that minimum.
>
> This means my threshold is stable. Even if my assumptions about the scoring formula are somewhat wrong, the system still performs reasonably well. A threshold that's optimal under one specific assumption might be catastrophic under another. The robust threshold is good everywhere.
>
> For cold-start calibration, I hold out 20% of training buyers and pretend they're cold. I build NACE industry profiles from the other 80%, predict for the held-out buyers, and evaluate. This gives me a separate cold threshold that accounts for the lower precision of industry-based predictions.

---

## "What would you do differently?" (Post-mortem answer)

> Three things.
>
> First, I'd explore **industry expansion for warm buyers**. Even warm buyers might be missing categories that their industry peers commonly purchase. Applying the cold-start NACE logic to warm buyers could surface these gaps.
>
> Second, I'd implement a **SWAP strategy** for combining prediction sources. Instead of just adding CF predictions on top of recurrence predictions, I'd replace the lowest-confidence recurrence predictions with higher-confidence CF ones. Same portfolio size, better average quality.
>
> Third, I'd add **association rule mining** for cross-category expansion. Buyers who purchase gloves frequently also tend to purchase safety goggles — that kind of co-purchase signal wasn't explicitly modeled.
>
> But honestly, the biggest leverage was in getting the economic framing right and being disciplined about portfolio construction. The model complexity was secondary to the decision framework.
