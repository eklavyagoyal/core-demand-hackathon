"""
v15_surgical.py — Surgical tuning around V7a (our proven €1,234k best)

FACT: V7a trained on 48 warm buyers' backtest = €1,234k
FACT: V9/V14 trained on 26k non-test buyers = €997-1,180k (WORSE)
FACT: The test buyers are special — only models trained on THEM work well

Plan: Reproduce V7a EXACTLY, then small perturbations:
- Cold size: 50, 80, 100, 150 per buyer
- EV threshold: -1, -1.5, -2, -2.5, -3
- Per-buyer cap: 400, 500, 600, uncapped
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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
print("V15 — SURGICAL TUNING AROUND V7a")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD (identical to V7a)
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

# Eclass features (same as V7a)
fpop = future_df.groupby("eclass").agg(
    fb=("legal_entity_id", "nunique"), fs=("total_value", "sum")).reset_index()
ft = future_df["legal_entity_id"].nunique()
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

n4d = fn.copy()
n4d["nace_4d"] = n4d["nace_code"].astype(int)
n4t = n4d.groupby("nace_4d")["legal_entity_id"].nunique().rename("tb")
n4p = n4d.groupby(["nace_4d", "eclass"]).agg(
    nb=("legal_entity_id", "nunique"), ts=("total_value", "sum")).reset_index().join(n4t, on="nace_4d")
n4p["pen"] = n4p["nb"] / n4p["tb"]
n4p["aspend"] = n4p["ts"] / n4p["nb"]

all_nace = df[df["eclass"].notna() & df["nace_code"].notna()].copy()
all_nace["nace_2d"] = all_nace["nace_code"].astype(int) // 10
hn2t = all_nace.groupby("nace_2d")["legal_entity_id"].nunique().rename("tb")
hn2p = all_nace.groupby(["nace_2d", "eclass"]).agg(
    nb=("legal_entity_id", "nunique"), ts=("total_value", "sum")).reset_index().join(hn2t, on="nace_2d")
hn2p["pen"] = hn2p["nb"] / hn2p["tb"]
hn2p["aspend"] = hn2p["ts"] / hn2p["nb"]

gfp = future_df.groupby("eclass").agg(
    nb=("legal_entity_id", "nunique"), ts=("total_value", "sum")).reset_index()
gfp["pen"] = gfp["nb"] / ft
gfp["aspend"] = gfp["ts"] / gfp["nb"]

warm_nace = {}
for bid in warm_ids:
    n = buyer_nace.get(bid)
    if pd.notna(n): warm_nace[bid] = int(n)
    else:
        tr = df_test[df_test["legal_entity_id"] == bid]
        if len(tr) > 0 and pd.notna(tr.iloc[0]["nace_code"]):
            warm_nace[bid] = int(tr.iloc[0]["nace_code"])


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN: EXACTLY V7a's approach (48 warm buyers, backtest)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[TRAIN] V7a-identical model (48 warm buyer backtest)")

warm_df = df[df["legal_entity_id"].isin(warm_set) &
             (df["orderdate"] <= WARM_CUTOFF) & df["eclass"].notna()].copy()
train_part = warm_df[warm_df["orderdate"] <= TRAIN_END].copy()
val_part = warm_df[(warm_df["orderdate"] >= VAL_START) & (warm_df["orderdate"] <= VAL_END)]

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

# OOF for backtest
oof = np.zeros(len(X))
for fold, (ti, vi) in enumerate(
    StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    dt = lgb.Dataset(X[ti], label=y[ti], weight=sw[ti], feature_name=FCOLS)
    dv = lgb.Dataset(X[vi], label=y[vi], weight=sw[vi], feature_name=FCOLS, reference=dt)
    m = lgb.train(params, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[vi] = m.predict(X[vi])
train_feats["ml_prob"] = oof

# Final model
model = lgb.train(params, lgb.Dataset(X, label=y, weight=sw, feature_name=FCOLS),
                  num_boost_round=250)

# Backtest
train_feats["est_spend"] = train_feats["evs"].clip(lower=0)
z = train_feats["est_spend"] == 0
if z.any():
    train_feats.loc[z, "est_spend"] = (
        train_feats.loc[z, "tsp"] / train_feats.loc[z, "nma"].clip(lower=1) * 6)
train_feats["est_spend"] = np.maximum(train_feats["est_spend"],
                                       train_feats["favg"] * 0.3)
train_feats["ev"] = train_feats["ml_prob"] * SR * train_feats["est_spend"] - FEE

print("\n  Backtest (warm only):")
for ev_t in [-3, -2.5, -2, -1.5, -1, 0]:
    sub = train_feats[train_feats["ev"] > ev_t]
    pairs = set(zip(sub["legal_entity_id"], sub["eclass"]))
    sav = sum(SR * val_spend.get(k, 0) for k in pairs if k in val_truth)
    h = sum(1 for k in pairs if k in val_truth)
    f = FEE * len(pairs)
    sc = sav - f
    pr = h / max(len(pairs), 1)
    print(f"  EV>{ev_t:>5.1f}: {len(pairs):>6} preds, {h:>5} hits, "
          f"prec={pr:.1%}, score=€{sc:>10,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# SCORE FULL FEATURES (same as V7a)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[SCORE] Full features (≤Jun2025)")
ff = build_feats(warm_df, WARM_CUTOFF)
Xff = ff[FCOLS].fillna(0).values
ff["ml_prob"] = model.predict(Xff)
ff["est_spend"] = ff["evs"].clip(lower=0)
z = ff["est_spend"] == 0
if z.any():
    ff.loc[z, "est_spend"] = ff.loc[z, "tsp"] / ff.loc[z, "nma"].clip(lower=1) * 6
ff["est_spend"] = np.maximum(ff["est_spend"], ff["favg"] * 0.3)
ff["ev"] = ff["ml_prob"] * SR * ff["est_spend"] - FEE

for ev_t in [-3, -2, -1, 0]:
    print(f"  EV>{ev_t}: {(ff['ev']>ev_t).sum():,} warm preds")


# ═══════════════════════════════════════════════════════════════════════════
# COLD START
# ═══════════════════════════════════════════════════════════════════════════
cold_meta = df_test[df_test["task"] == "cold start"][
    ["legal_entity_id", "estimated_number_employees", "nace_code", "secondary_nace_code"]].copy()

def make_cold(K):
    rows = []
    for _, row in cold_meta.iterrows():
        bid = row["legal_entity_id"]
        results, seen = [], set()
        def add(prof, min_pen, col, val):
            if val is not None: p = prof[prof[col] == val]
            else: p = prof
            f = p[p["pen"] >= min_pen].copy()
            f["sc"] = f["pen"] * np.log1p(f["aspend"])
            for _, r in f.sort_values("sc", ascending=False).iterrows():
                if len(results) >= K: break
                ec = r["eclass"]
                if ec not in seen: results.append(ec); seen.add(ec)
        nace = row["nace_code"]
        if pd.notna(nace):
            add(n4p, 0.05, "nace_4d", int(nace))
            add(n2p, 0.03, "nace_2d", int(nace) // 10)
            add(hn2p, 0.03, "nace_2d", int(nace) // 10)
        sec = row["secondary_nace_code"]
        if pd.notna(sec):
            add(n2p, 0.05, "nace_2d", int(sec) // 10)
        if len(results) < K:
            add(gfp, 0.01, None, None)
        for ec in results:
            rows.append({"buyer_id": bid, "predicted_id": ec})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# GENERATE GRID: EV threshold × cold size
# ═══════════════════════════════════════════════════════════════════════════
print("\n[GRID] Generating submissions")

def save(ev_thresh, cap, cold_k, label):
    w = ff[ff["ev"] > ev_thresh].sort_values(["legal_entity_id", "ev"], ascending=[True, False])
    if cap: w = w.groupby("legal_entity_id").head(cap)
    warm = w[["legal_entity_id", "eclass"]].rename(
        columns={"legal_entity_id": "buyer_id", "eclass": "predicted_id"})
    for bid in warm_ids:
        if bid not in set(warm["buyer_id"]):
            bf = ff[ff["legal_entity_id"] == bid]
            if len(bf) > 0:
                warm = pd.concat([warm, pd.DataFrame([{
                    "buyer_id": bid, "predicted_id": bf.nlargest(1, "mom").iloc[0]["eclass"]
                }])], ignore_index=True)
    cold = make_cold(cold_k)
    lvl1 = pd.concat([warm, cold], ignore_index=True).drop_duplicates()
    for bid in all_ids - set(lvl1["buyer_id"].unique()):
        for ec in gfp.nlargest(5, "pen")["eclass"]:
            lvl1 = pd.concat([lvl1, pd.DataFrame([{"buyer_id": bid, "predicted_id": ec}])],
                             ignore_index=True)
    lvl1 = lvl1.drop_duplicates()
    fn_out = f"json_approach/submission_{label}.csv"
    lvl1.to_csv(fn_out, index=False)
    n = len(lvl1)
    wn = lvl1[lvl1["buyer_id"].isin(warm_set)].shape[0]
    cn = lvl1[lvl1["buyer_id"].isin(cold_set)].shape[0]
    print(f"  {label:30s}: {n:>6} (w={wn:>5} c={cn:>5}) €{n*FEE:>8,.0f}")

# V7a reproduction (should match €1,234k)
save(-2, None, 150, "v15_ev2_c150")

# Tighter warm, same cold
save(-1.5, None, 150, "v15_ev1.5_c150")
save(-1, None, 150, "v15_ev1_c150")

# V7a warm, leaner cold
save(-2, None, 100, "v15_ev2_c100")
save(-2, None, 80, "v15_ev2_c80")

# V7a warm, ZERO cold (test if cold helps at all)
print("\n  --- Zero cold test ---")
w = ff[ff["ev"] > -2].sort_values(["legal_entity_id", "ev"], ascending=[True, False])
warm_only = w[["legal_entity_id", "eclass"]].rename(
    columns={"legal_entity_id": "buyer_id", "eclass": "predicted_id"})
for bid in warm_ids:
    if bid not in set(warm_only["buyer_id"]):
        bf = ff[ff["legal_entity_id"] == bid]
        if len(bf) > 0:
            warm_only = pd.concat([warm_only, pd.DataFrame([{
                "buyer_id": bid, "predicted_id": bf.nlargest(1, "mom").iloc[0]["eclass"]
            }])], ignore_index=True)
# Add minimal cold (just 1 item per cold buyer for coverage)
min_cold = make_cold(1)
lvl1_min = pd.concat([warm_only, min_cold], ignore_index=True).drop_duplicates()
for bid in all_ids - set(lvl1_min["buyer_id"].unique()):
    for ec in gfp.nlargest(1, "pen")["eclass"]:
        lvl1_min = pd.concat([lvl1_min, pd.DataFrame([{"buyer_id": bid, "predicted_id": ec}])],
                             ignore_index=True)
lvl1_min = lvl1_min.drop_duplicates()
lvl1_min.to_csv("json_approach/submission_v15_ev2_c1.csv", index=False)
print(f"  {'v15_ev2_c1 (minimal cold)':30s}: {len(lvl1_min):>6} "
      f"(w={lvl1_min[lvl1_min['buyer_id'].isin(warm_set)].shape[0]:>5} "
      f"c={lvl1_min[lvl1_min['buyer_id'].isin(cold_set)].shape[0]:>5}) "
      f"€{len(lvl1_min)*FEE:>8,.0f}")

# Wider warm + same cold
save(-2.5, None, 150, "v15_ev2.5_c150")
save(-3, None, 150, "v15_ev3_c150")

# Capped variants
save(-3, 500, 150, "v15_ev3_cap500_c150")
save(-3, 600, 150, "v15_ev3_cap600_c150")

print("\nDone!")
