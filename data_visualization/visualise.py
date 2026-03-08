"""
Data visualisation for Core Demand Hackathon Challenge 2.
Run: python visualise.py
Outputs: figures saved as PNG files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df_train = pd.read_csv("plis_training.csv.gz", compression="gzip", sep="\t", encoding="utf-8-sig", low_memory=False)
df_test  = pd.read_csv("customer_test.csv.gz",  compression="gzip", sep="\t", encoding="utf-8-sig")
df_nace  = pd.read_csv("nace_codes.csv.gz",      compression="gzip", sep="\t", encoding="utf-8-sig")

df_train["orderdate"]  = pd.to_datetime(df_train["orderdate"])
df_train["total_value"] = df_train["vk_per_item"] * df_train["quantityvalue"]

print("Data loaded. Building figures...")

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#F97316"
GREEN  = "#16A34A"
RED    = "#DC2626"
GREY   = "#6B7280"
LBLUE  = "#BFDBFE"

# =============================================================================
# FIGURE 1 — Long Tail vs Core Demand  (the signature curve)
# =============================================================================

warm_buyers = df_test[df_test["task"] == "predict future"]["legal_entity_id"].tolist()
warm_data   = df_train[df_train["legal_entity_id"].isin(warm_buyers)].copy()

# Per (buyer, eclass) aggregate spend
buyer_eclass = (
    warm_data[warm_data["eclass"].notna()]
    .groupby(["legal_entity_id", "eclass"])
    .agg(total_spend=("total_value", "sum"), n_months=("orderdate", lambda x: x.dt.to_period("M").nunique()))
    .reset_index()
)

# All (buyer, eclass) pairs sorted by total_spend descending — one buyer for illustration
sample_buyer = warm_buyers[1]  # pick a buyer with reasonable history
bd = buyer_eclass[buyer_eclass["legal_entity_id"] == sample_buyer].sort_values("total_spend", ascending=False).reset_index(drop=True)

# Also build the aggregate across ALL warm buyers for a global curve
global_eclass = (
    warm_data[warm_data["eclass"].notna()]
    .groupby("eclass")["total_value"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
global_eclass["cum_pct"] = global_eclass["total_value"].cumsum() / global_eclass["total_value"].sum() * 100
global_eclass["rank"]    = np.arange(1, len(global_eclass) + 1)
global_eclass["rank_pct"] = global_eclass["rank"] / len(global_eclass) * 100

# Find the 80/20 point
idx_80 = (global_eclass["cum_pct"] - 80).abs().idxmin()
rank_pct_80 = global_eclass.loc[idx_80, "rank_pct"]

fig1, axes = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Long Tail vs Core Demand", fontsize=16, fontweight="bold", y=1.01)

# ── Left: Pareto / cumulative spend curve ────────────────────────────────────
ax = axes[0]
ax.plot(global_eclass["rank_pct"], global_eclass["cum_pct"], color=BLUE, linewidth=2.5)
ax.axvline(rank_pct_80, color=RED, linestyle="--", linewidth=1.5, label=f"Top {rank_pct_80:.1f}% eclasses\n= 80% of spend")
ax.axhline(80, color=RED, linestyle=":", linewidth=1, alpha=0.6)
ax.fill_between(global_eclass["rank_pct"], global_eclass["cum_pct"],
                where=(global_eclass["rank_pct"] <= rank_pct_80),
                color=LBLUE, alpha=0.5, label="Core Demand zone")
ax.fill_between(global_eclass["rank_pct"], global_eclass["cum_pct"],
                where=(global_eclass["rank_pct"] > rank_pct_80),
                color=ORANGE, alpha=0.2, label="Long Tail zone")
ax.set_xlabel("% of E-Classes (ranked by spend)", fontsize=12)
ax.set_ylabel("Cumulative % of Total Spend", fontsize=12)
ax.set_title("Pareto Curve — E-Class Spend Concentration\n(all warm-start buyers combined)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.text(rank_pct_80 + 1, 5, f"{rank_pct_80:.1f}%", color=RED, fontsize=10, va="bottom")

# ── Right: Per-eclass spend bar (single sample buyer) ─────────────────────────
ax2 = axes[1]
top_n = min(30, len(bd))
colors = [GREEN if row["n_months"] >= 3 else ORANGE for _, row in bd.head(top_n).iterrows()]
bars = ax2.bar(range(top_n), bd["total_spend"].head(top_n), color=colors, edgecolor="white", linewidth=0.5)
ax2.set_xlabel("E-Class rank (by spend)", fontsize=12)
ax2.set_ylabel("Total Spend (EUR)", fontsize=12)
ax2.set_title(f"Long Tail Pattern — Sample Buyer {sample_buyer}\n(green = recurring ≥3 months, orange = long tail)", fontsize=11)
ax2.set_yscale("log")
ax2.set_xticks(range(0, top_n, 5))
ax2.set_xticklabels(range(1, top_n + 1, 5))
ax2.grid(axis="y", alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=GREEN, label="Core Demand (recurring ≥3 months)"),
              Patch(facecolor=ORANGE, label="Long Tail (infrequent)")]
ax2.legend(handles=legend_els, fontsize=10)

plt.tight_layout()
fig1.savefig("fig1_long_tail_core_demand.png", dpi=150, bbox_inches="tight")
print("  Saved fig1_long_tail_core_demand.png")

# =============================================================================
# FIGURE 2 — Dataset Overview (4-panel)
# =============================================================================

fig2 = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.35)
fig2.suptitle("Dataset Overview — plis_training", fontsize=16, fontweight="bold")

# ── 2a: Monthly order volume ──────────────────────────────────────────────────
ax = fig2.add_subplot(gs[0, :2])
monthly = df_train.groupby(df_train["orderdate"].dt.to_period("M")).size()
months  = [str(p) for p in monthly.index]
ax.bar(range(len(monthly)), monthly.values, color=BLUE, alpha=0.8, width=0.8)
# Shade the warm-buyer observation window
cutoff_idx = months.index("2025-06") if "2025-06" in months else None
if cutoff_idx:
    ax.axvline(cutoff_idx, color=RED, linewidth=2, linestyle="--", label="Warm buyer cutoff (2025-06)")
ax.set_xticks(range(0, len(months), 3))
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Order count", fontsize=11)
ax.set_title("Monthly Order Volume (Jan 2023 – Dec 2025)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# ── 2b: Orders per buyer distribution (log scale) ────────────────────────────
ax = fig2.add_subplot(gs[0, 2])
buyer_orders = df_train.groupby("legal_entity_id").size()
ax.hist(np.log10(buyer_orders + 1), bins=50, color=ORANGE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("log10(Orders per buyer)", fontsize=11)
ax.set_ylabel("Number of buyers", fontsize=11)
ax.set_title("Orders per Buyer\n(log scale)", fontsize=12)
ax.axvline(np.log10(buyer_orders.median()), color=RED, linestyle="--", linewidth=1.5, label=f"Median={buyer_orders.median():.0f}")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# ── 2c: Price distribution ───────────────────────────────────────────────────
ax = fig2.add_subplot(gs[1, 0])
prices = df_train["vk_per_item"].clip(0.01, 5000)
ax.hist(np.log10(prices), bins=60, color=GREEN, edgecolor="white", linewidth=0.3)
ax.set_xlabel("log10(Price per item, EUR)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Price Distribution\n(clipped at EUR 5K)", fontsize=12)
ax.axvline(np.log10(df_train["vk_per_item"].median()), color=RED, linestyle="--", linewidth=1.5,
           label=f"Median=EUR {df_train['vk_per_item'].median():.2f}")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ticks = [0.01, 0.1, 1, 10, 100, 1000]
ax.set_xticks(np.log10(ticks))
ax.set_xticklabels([str(t) for t in ticks])

# ── 2d: Top 15 NACE sectors among test buyers ────────────────────────────────
ax = fig2.add_subplot(gs[1, 1])
test_nace = df_test["nace_code"].dropna().astype(int)
nace_map  = df_nace.set_index(df_nace.columns[0])["n_nace_description"].to_dict()
test_nace_labels = test_nace.map(lambda x: nace_map.get(x, str(x)))
top_nace = test_nace_labels.value_counts().head(10)
ax.barh(range(len(top_nace)), top_nace.values, color=BLUE, alpha=0.85)
ax.set_yticks(range(len(top_nace)))
ax.set_yticklabels([t[:35] + "..." if len(t) > 35 else t for t in top_nace.index], fontsize=8)
ax.set_xlabel("Count", fontsize=11)
ax.set_title("Top 10 NACE Sectors\namong Test Buyers", fontsize=12)
ax.grid(axis="x", alpha=0.3)

# ── 2e: Cold vs warm split + null summary ────────────────────────────────────
ax = fig2.add_subplot(gs[1, 2])
categories = ["Cold Start\n(no history)", "Warm Start\n(has history)"]
counts     = [53, 47]
bar_colors = [ORANGE, GREEN]
bars = ax.bar(categories, counts, color=bar_colors, width=0.5, edgecolor="white")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Buyer count", fontsize=11)
ax.set_title("Test Buyer Split\n(100 total)", fontsize=12)
ax.set_ylim(0, 65)
ax.grid(axis="y", alpha=0.3)

plt.savefig("fig2_dataset_overview.png", dpi=150, bbox_inches="tight")
print("  Saved fig2_dataset_overview.png")

# =============================================================================
# FIGURE 3 — Recurrence Analysis (warm buyers)
# =============================================================================

fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle("Recurrence Analysis — Warm-Start Buyers", fontsize=15, fontweight="bold")

# ── 3a: Distribution of n_months_active per (buyer, eclass) ──────────────────
ax = axes[0]
ax.hist(buyer_eclass["n_months"], bins=30, color=BLUE, edgecolor="white", linewidth=0.3)
ax.axvline(3, color=RED, linestyle="--", linewidth=2, label="Threshold: 3 months")
ax.set_xlabel("Months active (per buyer-eclass pair)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Recurrence Distribution\n(buyer × eclass pairs)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
pct_core = (buyer_eclass["n_months"] >= 3).mean() * 100
ax.text(0.97, 0.95, f"{pct_core:.1f}% qualify\nas Core Demand\n(≥3 months)",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=LBLUE, alpha=0.8))

# ── 3b: Spend vs recurrence scatter ──────────────────────────────────────────
ax = axes[1]
sample = buyer_eclass.sample(min(5000, len(buyer_eclass)), random_state=42)
sc = ax.scatter(sample["n_months"], np.log10(sample["total_spend"] + 0.01),
                c=sample["n_months"], cmap="RdYlGn", alpha=0.5, s=15, edgecolors="none")
ax.axvline(3, color=RED, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xlabel("Months active", fontsize=11)
ax.set_ylabel("log10(Total Spend, EUR)", fontsize=11)
ax.set_title("Spend vs Recurrence\n(sample of 5K buyer-eclass pairs)", fontsize=12)
plt.colorbar(sc, ax=ax, label="Months active")
ax.grid(alpha=0.3)

# ── 3c: Economic signal — benefit proxy distribution ─────────────────────────
ax = axes[2]
buyer_eclass["avg_price"] = buyer_eclass["total_spend"] / buyer_eclass["n_months"].clip(1)
buyer_eclass["benefit_proxy"] = np.sqrt(buyer_eclass["avg_price"].clip(0)) * buyer_eclass["n_months"]

core = buyer_eclass[buyer_eclass["n_months"] >= 3]["benefit_proxy"]
tail = buyer_eclass[buyer_eclass["n_months"] < 3]["benefit_proxy"]

bins = np.linspace(0, np.percentile(buyer_eclass["benefit_proxy"], 95), 50)
ax.hist(tail.clip(upper=bins[-1]),  bins=bins, color=ORANGE, alpha=0.7, label=f"Long Tail (n={len(tail):,})", density=True)
ax.hist(core.clip(upper=bins[-1]),  bins=bins, color=GREEN,  alpha=0.7, label=f"Core Demand (n={len(core):,})", density=True)
ax.set_xlabel("Benefit Proxy = sqrt(avg_price) × n_months", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Economic Signal Distribution\nCore Demand vs Long Tail", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
fig3.savefig("fig3_recurrence_analysis.png", dpi=150, bbox_inches="tight")
print("  Saved fig3_recurrence_analysis.png")

# =============================================================================
# FIGURE 4 — Cold Start: Industry Profiles
# =============================================================================

# Build NACE → eclass profile from training data
df_train_nace = df_train[df_train["nace_code"].notna() & df_train["eclass"].notna()].copy()
df_train_nace["nace_2digit"] = (df_train_nace["nace_code"] // 10).astype(int)

# For a selection of top NACE 2-digit codes, show top eclasses
top_nace_2digit = df_train_nace["nace_2digit"].value_counts().head(6).index.tolist()

fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
fig4.suptitle("Cold-Start Strategy: Top E-Classes by Industry (NACE 2-digit)", fontsize=15, fontweight="bold")

nace_2d_map = {}
for _, row in df_nace.iterrows():
    code = row.iloc[0]
    desc = row["n_nace_description"]
    nace_2d_map[int(code) // 10] = desc

for i, nace_2d in enumerate(top_nace_2digit):
    ax = axes[i // 3][i % 3]
    subset = df_train_nace[df_train_nace["nace_2digit"] == nace_2d]
    top_eclass = subset.groupby("eclass")["total_value"].sum().sort_values(ascending=False).head(10)
    sector_desc = nace_2d_map.get(nace_2d, f"NACE {nace_2d}x")
    n_buyers = subset["legal_entity_id"].nunique()

    ax.barh(range(len(top_eclass)), top_eclass.values / 1e3, color=BLUE, alpha=0.8)
    ax.set_yticks(range(len(top_eclass)))
    ax.set_yticklabels(top_eclass.index.tolist(), fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Total Spend (EUR thousands)", fontsize=10)
    label = sector_desc[:45] + "..." if len(sector_desc) > 45 else sector_desc
    ax.set_title(f"NACE {nace_2d}x: {label}\n({n_buyers} buyers)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))

plt.tight_layout()
fig4.savefig("fig4_cold_start_industry_profiles.png", dpi=150, bbox_inches="tight")
print("  Saved fig4_cold_start_industry_profiles.png")

# =============================================================================
# FIGURE 5 — Portfolio Trade-off: Fee vs Savings
# =============================================================================

fig5, axes = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle("Portfolio Optimisation — Fee vs Savings Trade-off", fontsize=15, fontweight="bold")

# Simulate scoring for a single warm buyer with different k values (top-k eclasses)
sample_buyer_data = buyer_eclass[buyer_eclass["legal_entity_id"] == sample_buyer].sort_values("benefit_proxy", ascending=False).reset_index(drop=True)

# Hypothetical fee per item per month
assumed_fee = 20  # EUR (illustrative)

k_values = range(1, min(40, len(sample_buyer_data)) + 1)
scores, savings_list, fees_list = [], [], []

for k in k_values:
    top_k = sample_buyer_data.head(k)
    total_savings = (top_k["benefit_proxy"] * 0.05).sum()  # illustrative savings rate
    total_fees    = k * assumed_fee
    scores.append(total_savings - total_fees)
    savings_list.append(total_savings)
    fees_list.append(total_fees)

best_k = int(np.argmax(scores)) + 1

ax = axes[0]
ax.plot(list(k_values), savings_list, color=GREEN, linewidth=2, label="Cumulative Savings")
ax.plot(list(k_values), fees_list,    color=RED,   linewidth=2, linestyle="--", label="Cumulative Fees")
ax.plot(list(k_values), scores,       color=BLUE,  linewidth=2.5, label="Net Score (Savings - Fees)")
ax.axvline(best_k, color=ORANGE, linewidth=2, linestyle=":", label=f"Optimal K = {best_k}")
ax.axhline(0, color=GREY, linewidth=1, alpha=0.5)
ax.set_xlabel("Number of E-Classes in Portfolio (K)", fontsize=12)
ax.set_ylabel("EUR (illustrative)", fontsize=12)
ax.set_title(f"Score vs Portfolio Size\n(Sample buyer, assumed fee=EUR {assumed_fee}/item/month)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Right: Scatter of benefit_proxy vs rank for sample buyer
ax = axes[1]
colors_bar = [GREEN if row["n_months"] >= 3 else ORANGE for _, row in sample_buyer_data.iterrows()]
ax.bar(range(len(sample_buyer_data)), sample_buyer_data["benefit_proxy"], color=colors_bar, edgecolor="none")
ax.axvline(best_k - 0.5, color=RED, linewidth=2, linestyle="--", label=f"Optimal cut-off (K={best_k})")
ax.axhline(assumed_fee, color=BLUE, linewidth=1.5, linestyle=":", label=f"Fee threshold = {assumed_fee}")
ax.set_xlabel("E-Class rank (by benefit proxy)", fontsize=12)
ax.set_ylabel("Benefit Proxy Score", fontsize=12)
ax.set_title(f"Benefit Proxy per E-Class\n(Sample buyer — green=recurring, orange=long tail)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig5.savefig("fig5_portfolio_tradeoff.png", dpi=150, bbox_inches="tight")
print("  Saved fig5_portfolio_tradeoff.png")

print()
print("All figures saved:")
print("  fig1_long_tail_core_demand.png  — Pareto curve + per-buyer long tail pattern")
print("  fig2_dataset_overview.png       — Monthly volume, buyer distribution, price, test split")
print("  fig3_recurrence_analysis.png    — Recurrence depth, spend vs recurrence, economic signal")
print("  fig4_cold_start_industry_profiles.png  — Top eclasses per NACE industry")
print("  fig5_portfolio_tradeoff.png     — Fee vs savings, optimal K selection")
