
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
    _USE_TABULATE = True
except ImportError:
    _USE_TABULATE = False

# Paths
CSV_PATH = Path("consolidated_app_screens_votes.csv")
OUT_DIR  = Path("analysis_results")
OUT_DIR.mkdir(exist_ok=True)

# Helpers
def parse_labels(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [lab.strip().lower() for lab in str(cell).split(",") if lab.strip()]

def fleiss_kappa_var(counts):
    counts = np.asarray(counts, float)
    n_raters = counts.sum(axis=1)
    P_i = ((counts * (counts - 1)).sum(axis=1)) / (n_raters * (n_raters - 1))
    Pbar = P_i.mean()
    p_j = counts.sum(axis=0) / n_raters.sum()
    Pe = (p_j**2).sum()
    return np.nan if Pe == 1 else (Pbar - Pe) / (1 - Pe)

def items_for_kappa(kappa_target, n_raters, prevalence, alpha=0.05, power=0.90):
    Pe = prevalence**2 + (1-prevalence)**2
    var = 2 * prevalence * (1-prevalence) / (n_raters*(n_raters-1)*(1-Pe)**2)
    z_a = norm.ppf(1-alpha/2)
    z_b = norm.ppf(power)
    return int(math.ceil(((z_a+z_b)**2 * var) / (kappa_target**2)))

def print_df(df, title):
    print(f"\n=== {title} ===")
    if _USE_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    else:
        print(df.to_string(index=False))

# Load & prepare
def load_and_prepare(path):
    df = pd.read_csv(path)
    vote_cols = [c for c in df.columns if c.startswith("vote_")]
    for col in vote_cols + ["true_label"]:
        df[f"{col}_list"] = df[col].apply(parse_labels)
    df["votes_per_item"] = df[[f"{c}_list" for c in vote_cols]].apply(
        lambda row: sum(bool(x) for x in row), axis=1
    )
    labels = set()
    for c in vote_cols:
        for labs in df[f"{c}_list"]:
            labels.update(labs)
    for labs in df["true_label_list"]:
        labels.update(labs)
    return df, vote_cols, sorted(labels)

# Descriptive statistics
def descriptive_stats(df, vote_cols):
    freq = Counter()
    for c in vote_cols:
        for labs in df[f"{c}_list"]:
            freq.update(labs)
    freq_df = pd.DataFrame(freq.items(), columns=["label","count"]) \
                .sort_values("count", ascending=False).reset_index(drop=True)
    dist_df = df["votes_per_item"].value_counts() \
              .sort_index().rename_axis("num_votes") \
              .reset_index(name="num_items")

    # Bar plot
    plt.figure(figsize=(8,4))
    plt.bar(freq_df["label"], freq_df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Vote count")
    plt.title("Label frequency")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"label_frequency.png", dpi=150)
    plt.close()

    print_df(freq_df, "Label frequency across all votes")
    print_df(dist_df, "Distribution of annotator count per screenshot")
    return freq_df, dist_df

# Fleiss’ κ per label
def compute_kappas(df, vote_cols, labels):
    def binary_counts(label):
        rows = []
        for _, r in df.iterrows():
            flags = [int(label in r[f"{c}_list"]) for c in vote_cols if r[f"{c}_list"]]
            if flags:
                rows.append([len(flags)-sum(flags), sum(flags)])
        return np.array(rows)

    results = []
    for lab in labels:
        mat = binary_counts(lab)
        k = fleiss_kappa_var(mat) if mat.size else np.nan
        results.append((lab, round(k,3)))
    kappas_df = pd.DataFrame(results, columns=["label","fleiss_kappa"]) \
                  .sort_values("fleiss_kappa", ascending=False) \
                  .reset_index(drop=True)
    print_df(kappas_df, "Fleiss’ κ per label")

    pooled = fleiss_kappa_var(np.vstack([binary_counts(l) for l in labels]))
    print(f"\n>>> OVERALL pooled Fleiss’ κ: {round(pooled,3)}")

    return kappas_df

# Majority‐vote metrics
def majority_vote_metrics(df, vote_cols, labels):
    def maj_labels(row):
        cnt = Counter()
        voters = 0
        for c in vote_cols:
            labs = row[f"{c}_list"]
            if labs:
                voters += 1
                cnt.update(labs)
        thresh = math.ceil(voters/2) if voters else 1
        return [lab for lab,ct in cnt.items() if ct>=thresh] or ["unlabeled"]

    df["pred_labels"] = df.apply(maj_labels, axis=1)
    N = len(df)
    rows = []
    for lab in labels:
        tp=fp=fn=tn=0
        for preds,true in zip(df["pred_labels"], df["true_label_list"]):
            p,t = lab in preds, lab in true
            tp+=p and t; fp+=p and not t; fn+=t and not p; tn+=not p and not t
        prec = tp/(tp+fp) if tp+fp else np.nan
        rec  = tp/(tp+fn) if tp+fn else np.nan
        acc  = (tp+tn)/N
        rows.append((lab, tp, fp, fn, tn, round(prec,3), round(rec,3), round(acc,3)))
    mdf = pd.DataFrame(rows, columns=[
        "label","TP","FP","FN","TN","precision","recall","accuracy"
    ]).sort_values("precision", ascending=False).reset_index(drop=True)
    print_df(mdf, "Crowd vs ground-truth metrics")
    return mdf

# Power analysis
def power_analysis(df, vote_cols, mdf, labels):
    power_calc = NormalIndPower()
    baseline = 0.5
    rows = []
    for lab in labels:
        rec   = mdf.loc[mdf.label==lab, "recall"].iloc[0]
        gtpos = int(mdf.loc[mdf.label==lab, ["TP","FN"]].sum(axis=1).iloc[0])
        if gtpos==0 or math.isnan(rec):
            continue
        h_obs = 2*(math.asin(math.sqrt(rec)) - math.asin(math.sqrt(baseline)))
        post_p = power_calc.power(effect_size=h_obs, nobs1=gtpos,
                                  alpha=0.05, ratio=1.0, alternative="two-sided")
        h_tgt = 2*(math.asin(math.sqrt(0.70)) - math.asin(math.sqrt(baseline)))
        need_pos = math.ceil(power_calc.solve_power(
            effect_size=h_tgt, power=0.80, alpha=0.05,
            ratio=1.0, alternative="two-sided"
        ))
        rows.append((lab, gtpos, round(rec,3), round(h_obs,3), round(post_p,3), need_pos))

    power_df = pd.DataFrame(rows, columns=[
        "label","GT_Pos","recall","Cohen_h","posthoc_power","npos_for_rec70"
    ]).sort_values("posthoc_power", ascending=False).reset_index(drop=True)
    print_df(power_df, "Post-hoc power for recall")

    needs = []
    for lab in labels:
        prev = power_df.loc[power_df.label==lab, "GT_Pos"].iloc[0] / len(df)
        screens = items_for_kappa(0.60, len(vote_cols), prev)
        needs.append((lab, screens))
    need_df = pd.DataFrame(needs, columns=["label","screens_needed"]) \
                 .sort_values("screens_needed") \
                 .reset_index(drop=True)
    print_df(need_df, "Screens needed per label for κ ≥ 0.60")
    return power_df, need_df

def main():
    df, vote_cols, labels = load_and_prepare(CSV_PATH)
    descriptive_stats(df, vote_cols)
    compute_kappas(df, vote_cols, labels)
    mdf = majority_vote_metrics(df, vote_cols, labels)
    power_analysis(df, vote_cols, mdf, labels)

if __name__ == "__main__":
    main()
