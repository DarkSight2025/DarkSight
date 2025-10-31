"""
MobileViews: Core 2×2 Analysis 

Computes
- Counts per label (GPT vs expert/GT)
- GPT vs GT metrics (per label + macro/micro)
- Majority vote vs GT (overall + per-group micro; ≥3 workers per screen)
- Krippendorff's alpha (binary, per label & group)

Input
- CSV at /resulr.csv with columns: screen_id, group, crowd_annotations, expert_annotations, (optional) gpt_annotations

"""
from __future__ import annotations

import ast
import math
import numpy as np
import pandas as pd

# ---------------------------------
# Config
# ---------------------------------
INPUT_CSV = "/Results.csv"

# Canonical labels (evaluate only these; No_DDP is excluded from per-label metrics)
CODE_LABELS = [
    "FA-G-PRO",
    "II-AM-FH",
    "II-AM-G-SMALL",
    "II-PRE",
    "NG-AD",
]

# Mapping: code -> human-readable
CODE_TO_NAME = {
    "FA-G-PRO": "Pay to Avoid Ads",
    "II-AM-FH": "False Hierarchy",
    "II-AM-G-SMALL": "Small Close Button",
    "II-PRE": "Preselection",
    "NG-AD": "Pop-up Ad",
    "No_DDP": "No_Dark_Patterns",
}

# Reverse mapping (used for normalization)
NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}

# ---------------------------------
# Helpers
# ---------------------------------

def normalize_label_to_code(x: str) -> str:
    if not isinstance(x, str):
        return "No_DDP"
    s = x.strip()
    if not s:
        return "No_DDP"
    if s in CODE_TO_NAME:
        return s
    lower_map = {k.lower(): v for k, v in NAME_TO_CODE.items()}
    return lower_map.get(s.lower(), "No_DDP")


def annolist_to_codes_list(anno_list_str):
    """Parse stringified list[dict] with key 'label' → deduped sorted list of codes (excl. No_DDP)."""
    if not isinstance(anno_list_str, str) or not anno_list_str.strip():
        return []
    try:
        items = ast.literal_eval(anno_list_str)
    except Exception:
        return []
    codes = set()
    if isinstance(items, list):
        for d in items:
            if isinstance(d, dict) and "label" in d:
                code = normalize_label_to_code(d["label"])
                if code != "No_DDP":
                    codes.add(code)
    return sorted(codes)


def crowd_to_voters_lists(crowd_str):
    """
    Convert crowd_annotations (stringified list of dicts) into ballots per worker.
    Returns [list_of_codes_selected_by_worker], one list per distinct user_id (No_DDP excluded).
    """
    if not isinstance(crowd_str, str) or not crowd_str.strip():
        return []
    try:
        items = ast.literal_eval(crowd_str)
    except Exception:
        return []
    per_worker = {}
    if isinstance(items, list):
        for i, d in enumerate(items):
            if not (isinstance(d, dict) and "label" in d):
                continue
            uid = d.get("user_id") or f"anon_{i}"
            code = normalize_label_to_code(d["label"])
            s = per_worker.setdefault(uid, set())
            if code != "No_DDP":
                s.add(code)
    return [sorted(list(s)) for s in per_worker.values()]


def compute_metrics(label, df, pred_col, true_col):
    """Binary metrics for one label in multilabel setting."""
    pred = df[pred_col].apply(lambda lst: label in lst)
    true = df[true_col].apply(lambda lst: label in lst)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tp, fp, fn, tn, precision, recall, f1, fpr


def summarize_macro_micro(metrics_df):
    """Return macro and micro averages from a per-label metrics DataFrame."""
    tp_tot = metrics_df['TP'].sum()
    fp_tot = metrics_df['FP'].sum()
    fn_tot = metrics_df['FN'].sum()
    tn_tot = metrics_df['TN'].sum()
    macro = pd.Series({
        'Precision': metrics_df['Precision'].mean(),
        'Recall':    metrics_df['Recall'].mean(),
        'F1 Score':  metrics_df['F1 Score'].mean(),
        'FPR':       metrics_df['FPR'].mean()
    }, name='Macro Avg')
    prec_micro = tp_tot/(tp_tot+fp_tot) if (tp_tot+fp_tot) else 0.0
    rec_micro  = tp_tot/(tp_tot+fn_tot) if (tp_tot+fn_tot) else 0.0
    f1_micro   = (2*prec_micro*rec_micro/(prec_micro+rec_micro)) if (prec_micro+rec_micro) else 0.0
    fpr_micro  = fp_tot/(fp_tot+tn_tot) if (fp_tot+tn_tot) else 0.0
    micro = pd.Series({
        'Precision': prec_micro,
        'Recall':    rec_micro,
        'F1 Score':  f1_micro,
        'FPR':       fpr_micro
    }, name='Micro Avg')
    return pd.DataFrame([macro, micro])


def strict_majority(voters_lists, label):
    """Half or more of workers must select the label: need = ceil(W/2)."""
    n = len(voters_lists)
    if n == 0:
        return False
    need = math.ceil(n/2)
    votes = sum(label in v for v in voters_lists)
    return votes >= need


def krippendorff_alpha_binary(item_ratings):
    """Krippendorff's alpha for nominal binary data (variable raters per item)."""
    Do_num = 0.0
    Do_den = 0.0
    counts = {0: 0, 1: 0}
    for ratings in item_ratings:
        n = len(ratings)
        if n <= 1:
            continue
        Do_den += n * (n - 1)
        for r in ratings:
            if r not in (0, 1):
                raise ValueError("krippendorff_alpha_binary expects 0/1 ratings only.")
            counts[r] += 1
        for i in range(n):
            for j in range(i + 1, n):
                if ratings[i] != ratings[j]:
                    Do_num += 1
    if Do_den == 0:
        return np.nan
    Do = Do_num / Do_den
    N = counts[0] + counts[1]
    if N == 0:
        return np.nan
    p0 = counts[0] / N
    p1 = counts[1] / N
    De = 1 - (p0**2 + p1**2)
    return 1.0 if De == 0 else 1 - Do / De


# --- Micro P/R/F1 on screen-level majority predictions ---

def micro_counts(df_sub, codes, pred_col='majority_vote', true_col='ground_truth_list'):
    TP = FP = FN = 0
    for true_list, pred_list in zip(df_sub[true_col], df_sub[pred_col]):
        pred_set = set(pred_list); true_set = set(true_list)
        for code in codes:
            actual = code in true_set
            predicted = code in pred_set
            if predicted and actual: TP += 1
            elif predicted and not actual: FP += 1
            elif (not predicted) and actual: FN += 1
    return TP, FP, FN


def micro_metrics(df_sub, codes, pred_col='majority_vote', true_col='ground_truth_list'):
    TP, FP, FN = micro_counts(df_sub, codes, pred_col, true_col)
    prec = TP/(TP+FP) if (TP+FP) else np.nan
    rec  = TP/(TP+FN) if (TP+FN) else np.nan
    f1   = (2*prec*rec/(prec+rec)) if (not np.isnan(prec) and not np.isnan(rec) and (prec+rec)>0) else np.nan
    return {'TP':TP, 'FP':FP, 'FN':FN, 'precision':prec, 'recall':rec, 'f1':f1}


# ---------------------------------
# Main
# ---------------------------------

def main():
    print(f"Reading per-screen file from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    print("Initial DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_cols = ['screen_id', 'group', 'crowd_annotations', 'expert_annotations']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Build lists
    df['Voters_list'] = df['crowd_annotations'].apply(crowd_to_voters_lists)
    df['ground_truth_list'] = df['expert_annotations'].apply(annolist_to_codes_list)
    if 'gpt_annotations' in df.columns:
        df['GPT_label_list'] = df['gpt_annotations'].apply(annolist_to_codes_list)
    else:
        df['GPT_label_list'] = [[] for _ in range(len(df))]

    print("\nUnique groups in data:", df['group'].unique())
    print("Total screens:", len(df))

    # -------------------------
    # COUNTS (sanity)
    # -------------------------
    gpt_counts = (
        pd.Series([l for lst in df['GPT_label_list'] for l in lst]).value_counts()
          .rename_axis('Pattern').reset_index(name='gpt_count')
    ) if df['GPT_label_list'].map(len).sum() > 0 else pd.DataFrame(columns=['Pattern','gpt_count'])

    gt_counts = (
        pd.Series([l for lst in df['ground_truth_list'] for l in lst]).value_counts()
          .rename_axis('Pattern').reset_index(name='ground_truth_count')
    )
    counts = (pd.merge(gpt_counts, gt_counts, on='Pattern', how='outer')
              .fillna(0)
              .assign(
                  gpt_count=lambda d: d['gpt_count'].astype(int) if 'gpt_count' in d else 0,
                  ground_truth_count=lambda d: d['ground_truth_count'].astype(int),
                  PatternFull=lambda d: d['Pattern'].map(CODE_TO_NAME).fillna(d['Pattern'])
              )
              .sort_values('PatternFull')
             )
    print("\n--- Combined counts (codes) ---")
    print(counts[['Pattern','gpt_count','ground_truth_count']].to_string(index=False))

    # -------------------------
    # GPT vs GROUND TRUTH
    # -------------------------
    rows = []
    for code in CODE_LABELS:
        tp, fp, fn, tn, prec, rec, f1, fpr = compute_metrics(code, df, 'GPT_label_list', 'ground_truth_list')
        rows.append({
            'Pattern':   CODE_TO_NAME[code],
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': round(prec, 3),
            'Recall':    round(rec, 3),
            'F1 Score':  round(f1, 3),
            'FPR':       round(fpr, 4)
        })
    metrics_gpt = pd.DataFrame(rows)
    print("\n==================== GPT vs. GROUND TRUTH (OVERALL) ====================")
    print(metrics_gpt.to_string(index=False))
    print("\nAverages (overall):")
    print(summarize_macro_micro(metrics_gpt).round(6).to_string())

    # -------------------------
    # MAJORITY VOTE vs GROUND TRUTH
    # -------------------------
    df_majority = df[df['Voters_list'].apply(len) >= 3].copy()

    def majority_labels(voters_lists, labels=CODE_LABELS):
        return [lbl for lbl in labels if strict_majority(voters_lists, lbl)]

    df_majority['majority_vote'] = df_majority['Voters_list'].apply(majority_labels)

    drop_rate_by_group = (df.assign(has3=df['Voters_list'].apply(len) >= 3)
                            .groupby('group')['has3']
                            .agg(rate=lambda s: 1 - s.mean(), n='count'))
    print("\n--- Screens dropped for <3 voters by group ---")
    print(drop_rate_by_group.to_string())

    rows = []
    for code in CODE_LABELS:
        tp, fp, fn, tn, prec, rec, f1, fpr = compute_metrics(code, df_majority, 'majority_vote', 'ground_truth_list')
        rows.append({
            'Pattern':   CODE_TO_NAME[code],
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': round(prec, 3),
            'Recall':    round(rec, 3),
            'F1 Score':  round(f1, 3),
            'FPR':       round(fpr, 4)
        })
    metrics_maj_overall = pd.DataFrame(rows)
    print("\n================ MAJORITY VOTE vs. GROUND TRUTH (OVERALL, ≥3 WORKERS) ================")
    print(f"Rows evaluated: {len(df_majority)}")
    print(metrics_maj_overall.to_string(index=False))
    print("\nAverages (overall, majority):")
    print(summarize_macro_micro(metrics_maj_overall).round(6).to_string())

    group_rows = []
    for cond in sorted(df_majority['group'].unique()):
        df_grp = df_majority[df_majority['group'] == cond]
        m = micro_metrics(df_grp, CODE_LABELS)
        group_rows.append({
            'group': cond,
            'rows': len(df_grp),
            'Precision_micro': round(m['precision'], 6),
            'Recall_micro':    round(m['recall'], 6),
            'F1_micro':        round(m['f1'], 6),
            'TP': m['TP'], 'FP': m['FP'], 'FN': m['FN']
        })
    group_maj_table = pd.DataFrame(group_rows).sort_values('group')
    print("\n--- Per-group (HUMAN majority) micro Precision/Recall/F1 ---")
    print(group_maj_table.to_string(index=False))

    # -------------------------
    # KRIPPENDORFF'S ALPHA
    # -------------------------
    print("\n### Per-Condition Krippendorff's α (binary, per label) ###")
    alpha_rows = []
    for cond in sorted(df['group'].unique()):
        df_grp = df[df['group'] == cond]
        for code in CODE_LABELS:
            item_ratings = df_grp['Voters_list'].apply(lambda vs: [1 if code in v else 0 for v in vs]).tolist()
            a = krippendorff_alpha_binary(item_ratings)
            alpha_rows.append({
                'group': cond,
                'code': code,
                'pattern': CODE_TO_NAME[code],
                'alpha': round(float(a), 6) if a == a else np.nan
            })
    alpha_table = pd.DataFrame(alpha_rows).sort_values(['group','pattern'])
    print("\n--- Krippendorff's α per label & group ---")
    print(alpha_table.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
