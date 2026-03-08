"""
v17_gameplan.py — The 3-Lever Approach

Lever 1: SMARTER warm — rank by ml_prob × max(all spend estimates), require future_pen > 0
Lever 2: PERSONALIZED cold — twin-match each cold buyer to 5 nearest non-test peers
Lever 3: SPEND-PRIORITIZED ranking with better spend estimates
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import warnings, os
warnings.filterwarnings("ignore")
os.chdir("/Users/eklavyagoyal/Projects/hackathons/data-mining-hackathon/Challenge2")

WARM_CUTOFF  = pd.Timestamp("2025-06-30")
TRAIN_END    = pd.Timestamp("2024-12-31")
VAL_START    = pd.Timestamp("2025-01-01")
VAL_END      = pd.Timestamp("2025-06-30")
FUTURE_START = pd.Timestamp("2025-07-01")
SR, FEE = 0.10, 10.0

print("=" * 70)
print("V17 — THREE-LEVER OPTIMIZATION")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv("plis_training.csv.gz", compression="gzip", sep="\t",
                  encoding="utf-8-sig", low_memory=False)
df_test = pd.read_csv("customer_test.csv.gz", compression="gzip", sep="\t",
                       encoding="utf-8-sig")
df["orderdate"] = pd.to_datetime(df["orderdate"])
df["eclass"] = df["eclass"].astype(str).str.strip().replace(
    {"nan": np.nan, "": np.nan, "<NA>": np.nan})
df["total_value"] = df["vk_per_item"] * df["quantityvalue"]
sku_ec = df[df["eclass"].notna()].groupby("sku")["eclass"].agg(lambda x: x.mode().iloc[0])
df.loc[df["eclass"].isna(), "eclass"] = df.loc[df["eclass"].isna(), "sku"].map(sku_ec)
df["year_month"] = df["orderdate"].dt.to_period("M")

warm_ids = df_test[df_test["task"] == "predict future"]["legal_entity_id"].tolist()
cold_ids = df_test[df_test["task"] == "cold start"]["legal_entity_id"].tolist()
warm_set, cold_set = set(warm_ids), set(cold_ids)
all_ids = set(df_test["legal_entity_id"])
test_ids = set(df_test["legal_entity_id"])

future_df = df[(df["orderdate"] >= FUTURE_START) & ~df["legal_entity_id"].isin(test_ids) &
               df["eclass"].notna()].copy()
buyer_nace = df.groupby("legal_entity_id")["nace_code"].first().to_dict()
buyer_emp = df.groupby("legal_entity_id")["estimated_number_employees"].first().to_dict()
ft = future_df["legal_entity_id"].nunique()

# Eclass features
fpop = future_df.groupby("eclass").agg(
    fb=("legal_entity_id", "nunique"), fs=("total_value", "sum")).reset_index()
fpop["fpen"] = fpop["fb"] / ft
fpop["favg"] = fpop["fs"] / fpop["fb"]

hist_all = df[(df["orderdate"] <= WARM_CUTOFF) & df["eclass"].notna()]
hpop = hist_all.groupby("eclass").agg(hb=("legal_entity_id", "nunique")).reset_index()
hpop["hpen"] = hpop["hb"] / hist_all["legal_entity_id"].nunique()

ecf = hpop.merge(fpop, on="eclass", how="outer")
for c in ecf.select_dtypes(include=[np.number]).columns:
    ecf[c] = ecf[c].fillna(0)
ecf["trend"] = (ecf["fpen"] / ecf["hpen"].clip(lower=1e-6)).clip(upper=10)

h2 = df[df["orderdate"].dt.month >= 7].groupby("eclass")["total_value"].count().rename("h2c")
h1 = df[df["orderdate"].dt.month < 7].groupby("eclass")["total_value"].count().rename("h1c")
ss = pd.DataFrame({"h2c": h2, "h1c": h1}).fillna(0)
ss["h2r"] = ss["h2c"] / (ss["h1c"] + ss["h2c"]).clip(lower=1)
ecf = ecf.merge(ss[["h2r"]], left_on="eclass", right_index=True, how="left")
ecf["h2r"] = ecf["h2r"].fillna(0.5)

# NACE profiles
fn = future_df.copy()
fn["nace_code"] = fn["legal_entity_id"].map(buyer_nace)
fn = fn[fn["nace_code"].notna()]
fn["nace_2d"] = fn["nace_code"].astype(int) // 10
n2t = fn.groupby("nace_2d")["legal_entity_id"].nunique().rename("tb")
n2p = fn.groupby(["nace_2d", "eclass"]).agg(
    nb=("legal_entity_id", "nunique"), ts=("total_value", "sum")).reset_index().join(n2t, on="nace_2d")
n2p["pen"] = n2p["nb"] / n2p["tb"]
n2p["aspend"] = n2p["ts"] / n2p["nb"]
nace_pen = n2p.set_index(["nace_2d", "eclass"])["pen"].to_dict()
nace_spend = n2p.set_index(["nace_2d", "eclass"])["aspend"].to_dict()

warm_nace = {}
for bid in warm_ids:
    n = buyer_nace.get(bid)
    if pd.notna(n): warm_nace[bid] = int(n)
    else:
        tr = df_test[df_test["legal_entity_id"] == bid]
        if len(tr) > 0 and pd.notna(tr.iloc[0]["nace_code"]):
            warm_nace[bid] = int(tr.iloc[0]["nace_code"])

print(f"  Warm={len(warm_ids)} Cold={len(cold_ids)}")


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 1: SMARTER WARM — V7a model + better spend estimation + alive filter
# ═══════════════════════════════════════════════════════════════════════════
print("\n[LEVER 1] Smart Warm Predictions")

warm_df = df[df["legal_entity_id"].isin(warm_set) &
             (df["orderdate"] <= WARM_CUTOFF) & df["eclass"].notna()].copy()
train_part = warm_df[warm_df["orderdate"] <= TRAIN_END]
val_part = warm_df[(warm_df["orderdate"] >= VAL_START) & (warm_df["orderdate"] <= VAL_END)]

# Build features (V7a architecture)
def build_feats(buyer_df, ref_date):
    w = buyer_df.copy()
    w["year_month"] = w["orderdate"].dt.to_period("M")
    a = w.groupby(["legal_entity_id", "eclass"]).agg(
        nma=("year_month", "nunique"), nord=("total_value", "count"),
        tsp=("total_value", "sum"), avgp=("vk_per_item", "mean"),
        lo=("orderdate", "max"), fo=("orderdate", "min")).reset_index()
    a["dsl"] = (ref_date - a["lo"]).dt.days
    a["stalled"] = (a["dsl"] > 120).astype(int)
    a["tenure"] = ((a["lo"] - a["fo"]).dt.days / 30.0).clip(lower=1)
    a["freq"] = a["nord"] / a["tenure"]
    rs = ref_date - pd.DateOffset(months=6)
    rec = w[w["orderdate"] >= rs]
    if len(rec) > 0:
        ra = rec.groupby(["legal_entity_id", "eclass"]).agg(
            rsp=("total_value", "sum"), rord=("total_value", "count"),
            rmn=("year_month", "nunique")).reset_index()
        a = a.merge(ra, on=["legal_entity_id", "eclass"], how="left")
    for c in ["rsp", "rord", "rmn"]:
        if c not in a: a[c] = 0
        a[c] = a[c].fillna(0)
    a["rratio"] = a["rsp"] / a["tsp"].clip(lower=1e-9)
    a["ams"] = a["rsp"] / 6
    a["bp"] = np.sqrt(a["avgp"].clip(lower=0.01)) * a["freq"]
    a["evs"] = a["ams"] * 6
    a["mom"] = a["ams"] * (1 + a["rratio"])
    tm = max(1, (ref_date.year - w["orderdate"].min().year) * 12 +
             ref_date.month - w["orderdate"].min().month)
    a["reg"] = a["nma"] / tm
    bt = w.groupby("legal_entity_id")["total_value"].sum().rename("bts")
    a = a.merge(bt, on="legal_entity_id", how="left")
    a["sshare"] = a["tsp"] / a["bts"].clip(lower=1e-9)
    bd = w.groupby("legal_entity_id")["eclass"].nunique().rename("bnec")
    bm = w.groupby("legal_entity_id")["year_month"].nunique().rename("bam")
    a = a.merge(bd, on="legal_entity_id", how="left")
    a = a.merge(bm, on="legal_entity_id", how="left")
    a = a.merge(ecf[["eclass", "fpen", "favg", "trend", "h2r", "hpen"]],
                on="eclass", how="left")
    for c in ["fpen", "favg", "trend", "h2r", "hpen"]: a[c] = a[c].fillna(0)
    a["nace"] = a["legal_entity_id"].map(warm_nace)
    a["n2d"] = a["nace"].apply(lambda x: x // 10 if pd.notna(x) else np.nan)
    a["nep"] = a.apply(lambda r: nace_pen.get((r["n2d"], r["eclass"]), 0)
                        if pd.notna(r["n2d"]) else 0, axis=1)
    # LEVER 3: Better spend estimate — max of all sources
    a["nace_aspend"] = a.apply(
        lambda r: nace_spend.get((r["n2d"], r["eclass"]), 0)
        if pd.notna(r["n2d"]) else 0, axis=1)
    return a

FCOLS = ["nma", "nord", "tsp", "avgp", "dsl", "stalled", "tenure", "freq",
         "rsp", "rord", "rmn", "rratio", "ams", "bp", "evs", "mom", "reg",
         "sshare", "bnec", "bam", "fpen", "favg", "trend", "h2r", "hpen", "nep"]

train_feats = build_feats(train_part, TRAIN_END)
val_truth = set(zip(val_part["legal_entity_id"], val_part["eclass"]))
val_spend = val_part.groupby(["legal_entity_id", "eclass"])["total_value"].sum().to_dict()
train_feats["target"] = train_feats.apply(
    lambda r: 1 if (r["legal_entity_id"], r["eclass"]) in val_truth else 0, axis=1)

X = train_feats[FCOLS].fillna(0).values
y = train_feats["target"].values
sw = train_feats["bp"].clip(lower=0.01).values

params = {"objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
          "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20,
          "verbose": -1, "seed": 42}
model = lgb.train(params, lgb.Dataset(X, label=y, weight=sw, feature_name=FCOLS),
                  num_boost_round=250)

# Score full warm features
ff = build_feats(warm_df, WARM_CUTOFF)
ff["ml_prob"] = model.predict(ff[FCOLS].fillna(0).values)

# LEVER 3: Better spend estimation — max of recent, historical, NACE, future
ff["spend_recent"] = ff["evs"]  # recent 6m monthly × 6
ff["spend_hist"] = ff["tsp"] / ff["nma"].clip(lower=1) * 6  # all-time monthly × 6
ff["spend_nace"] = ff["nace_aspend"]  # NACE average for this eclass
ff["spend_future"] = ff["favg"]  # Future average spend

ff["best_spend"] = np.maximum(
    np.maximum(ff["spend_recent"], ff["spend_hist"]),
    np.maximum(ff["spend_nace"] * 0.5, ff["spend_future"] * 0.3)
).clip(lower=10)

ff["ev"] = ff["ml_prob"] * SR * ff["best_spend"] - FEE

# Smart filter: prefer items that are "alive" (future_pen > 0 or recent)
ff["is_alive"] = ((ff["fpen"] > 0) | (ff["rord"] > 0)).astype(int)
ff["is_recent"] = (ff["dsl"] <= 365).astype(int)

print(f"  Warm pairs: {len(ff):,}")
print(f"  Alive (future_pen>0 or recent): {ff['is_alive'].sum():,}")
print(f"  EV>-2 all: {(ff['ev']>-2).sum():,}")
print(f"  EV>-2 alive: {((ff['ev']>-2) & (ff['is_alive']==1)).sum():,}")


# ═══════════════════════════════════════════════════════════════════════════
# LEVER 2: PERSONALIZED COLD — twin-match to nearest non-test peers
# ═══════════════════════════════════════════════════════════════════════════
print("\n[LEVER 2] Personalized Cold Start")

# Build buyer profiles for matching
cold_meta = df_test[df_test["task"] == "cold start"][
    ["legal_entity_id", "estimated_number_employees", "nace_code", "secondary_nace_code"]].copy()

# Non-test buyers with future data and NACE info
nt_buyers = future_df.groupby("legal_entity_id").agg(
    n_eclasses=("eclass", "nunique"),
    total_spend=("total_value", "sum")
).reset_index()
nt_buyers["nace"] = nt_buyers["legal_entity_id"].map(buyer_nace)
nt_buyers["emp"] = nt_buyers["legal_entity_id"].map(buyer_emp)
nt_buyers["nace_4d"] = nt_buyers["nace"].apply(lambda x: int(x) if pd.notna(x) else None)
nt_buyers["nace_2d"] = nt_buyers["nace"].apply(lambda x: int(x) // 10 if pd.notna(x) else None)
nt_buyers = nt_buyers[nt_buyers["nace_4d"].notna()].copy()

# Get future purchases per non-test buyer
future_by_buyer = future_df.groupby("legal_entity_id")["eclass"].apply(set).to_dict()

print(f"  Non-test buyers with NACE+future: {len(nt_buyers):,}")

# For each cold buyer, find twins and collect their purchases
cold_rows = []
cold_stats = []
for _, row in cold_meta.iterrows():
    bid = row["legal_entity_id"]
    nace = row["nace_code"]
    emp = row["estimated_number_employees"]

    if pd.isna(nace):
        # No NACE → use global top
        eclasses = fpop.nlargest(100, "fpen")["eclass"].tolist()
        cold_stats.append({"bid": bid, "method": "global", "n_twins": 0, "n_preds": len(eclasses)})
        for ec in eclasses:
            cold_rows.append({"buyer_id": bid, "predicted_id": ec})
        continue

    nace_4d = int(nace)
    nace_2d = nace_4d // 10

    # Find twins: same NACE 4-digit, similar employee count
    candidates = nt_buyers[nt_buyers["nace_4d"] == nace_4d].copy()
    if len(candidates) < 5:
        # Fallback to 2-digit NACE
        candidates = nt_buyers[nt_buyers["nace_2d"] == nace_2d].copy()

    if len(candidates) == 0:
        # Global fallback
        eclasses = fpop.nlargest(100, "fpen")["eclass"].tolist()
        cold_stats.append({"bid": bid, "method": "global", "n_twins": 0, "n_preds": len(eclasses)})
        for ec in eclasses:
            cold_rows.append({"buyer_id": bid, "predicted_id": ec})
        continue

    # Sort by employee similarity
    if pd.notna(emp) and emp > 0:
        candidates["emp_sim"] = 1.0 / (1.0 + np.abs(np.log1p(candidates["emp"].fillna(100)) -
                                                       np.log1p(emp)))
    else:
        candidates["emp_sim"] = 1.0

    candidates = candidates.sort_values("emp_sim", ascending=False)
    twins = candidates.head(min(10, len(candidates)))

    # Collect ALL eclasses purchased by twins, weighted by frequency
    eclass_counts = {}
    eclass_spends = {}
    n_twins = len(twins)
    for _, twin_row in twins.iterrows():
        twin_id = twin_row["legal_entity_id"]
        twin_ecs = future_by_buyer.get(twin_id, set())
        for ec in twin_ecs:
            eclass_counts[ec] = eclass_counts.get(ec, 0) + 1
            # Track spend too
            key = (twin_id, ec)
            spend = future_df[(future_df["legal_entity_id"] == twin_id) &
                              (future_df["eclass"] == ec)]["total_value"].sum()
            eclass_spends[ec] = eclass_spends.get(ec, 0) + spend

    # Rank by: (count/n_twins) * log(avg_spend) — penalizes rare, boosts high-spend
    scored = []
    for ec, cnt in eclass_counts.items():
        freq = cnt / n_twins
        avg_sp = eclass_spends[ec] / cnt
        score = freq * np.log1p(avg_sp)
        scored.append((ec, score, freq))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Dynamic K: include items bought by ≥20% of twins, up to 500
    eclasses = [ec for ec, sc, freq in scored if freq >= 0.2][:500]
    if len(eclasses) < 20:
        eclasses = [ec for ec, sc, freq in scored][:200]

    method = f"twin_{nace_4d}" if len(candidates[candidates["nace_4d"] == nace_4d]) >= 5 else f"twin_2d_{nace_2d}"
    cold_stats.append({"bid": bid, "method": method, "n_twins": n_twins, "n_preds": len(eclasses)})

    for ec in eclasses:
        cold_rows.append({"buyer_id": bid, "predicted_id": ec})

cold_preds = pd.DataFrame(cold_rows)
csd = pd.DataFrame(cold_stats)

print(f"  Cold predictions: {len(cold_preds):,}")
print(f"  Per buyer: mean={csd['n_preds'].mean():.0f}, median={csd['n_preds'].median():.0f}, "
      f"min={csd['n_preds'].min()}, max={csd['n_preds'].max()}")
print(f"  Methods: {csd['method'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE SUBMISSIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SUBMISSIONS]")

def save(warm_mask, cold_df, label):
    warm = ff[warm_mask][["legal_entity_id", "eclass"]].rename(
        columns={"legal_entity_id": "buyer_id", "eclass": "predicted_id"})
    # Ensure all warm covered
    for bid in warm_ids:
        if bid not in set(warm["buyer_id"]):
            bf = ff[ff["legal_entity_id"] == bid]
            if len(bf) > 0:
                warm = pd.concat([warm, pd.DataFrame([{
                    "buyer_id": bid, "predicted_id": bf.nlargest(1, "mom").iloc[0]["eclass"]
                }])], ignore_index=True)

    lvl1 = pd.concat([warm, cold_df[["buyer_id", "predicted_id"]]], ignore_index=True).drop_duplicates()
    # Missing buyers
    for bid in all_ids - set(lvl1["buyer_id"].unique()):
        for ec in fpop.nlargest(10, "fpen")["eclass"]:
            lvl1 = pd.concat([lvl1, pd.DataFrame([{"buyer_id": bid, "predicted_id": ec}])],
                             ignore_index=True)
    lvl1 = lvl1.drop_duplicates()

    # Use correct column names for scorer
    lvl1.columns = ["legal_entity_id", "cluster"]
    fn_out = f"json_approach/submission_{label}.csv"
    lvl1.to_csv(fn_out, index=False)
    n = len(lvl1)
    wn = lvl1[lvl1["legal_entity_id"].isin(warm_set)].shape[0]
    cn = lvl1[lvl1["legal_entity_id"].isin(cold_set)].shape[0]
    print(f"  {label:40s}: {n:>6} (w={wn:>5} c={cn:>5}) €{n*FEE:>8,.0f}")

# V17a: Smart warm (EV>-2, alive only) + personalized cold
save((ff["ev"] > -2) & (ff["is_alive"] == 1), cold_preds, "v17a_smart_twin")

# V17b: Smart warm (EV>-2.5, alive) + personalized cold
save((ff["ev"] > -2.5) & (ff["is_alive"] == 1), cold_preds, "v17b_wider_twin")

# V17c: All EV>-2 (no alive filter) + personalized cold
save(ff["ev"] > -2, cold_preds, "v17c_ev2_twin")

# V17d: EV>-3 alive + personalized cold
save((ff["ev"] > -3) & (ff["is_alive"] == 1), cold_preds, "v17d_ev3_twin")

# V17e: Smart warm EV>-2 alive + SMALLER personalized cold (top 200/buyer)
cold_small = cold_preds.groupby("buyer_id").head(200)
save((ff["ev"] > -2) & (ff["is_alive"] == 1), cold_small, "v17e_smart_twin200")

# V17f: V7a warm + personalized cold (test cold quality alone)
v7a_mask = ff["ev"] > -2  # Same as V7a
save(v7a_mask, cold_preds, "v17f_v7a_twin")

print("\nDone!")
