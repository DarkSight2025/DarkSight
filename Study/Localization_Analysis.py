
r"""

Methods (crowd & GPT):

  • SIMPLE
      - Majority gate (≥ MIN_WORKERS on screen; votes_for_label ≥ ceil(W/2))
      - Per-worker mean boxes → average across workers (one fused box per label)

  • WBF_equal
      - Real WBF-style clustering (IoU ≥ WBF_IOU_THR), equal weights
      - Return a single top-1 fused box per label (largest-vote cluster)

TP definition (per screen × label):
  • GPT-TP   : screen has expert label AND GPT predicted that label on the screen
  • Crowd-TP : screen has expert label AND screen has ≥ MIN_WORKERS AND
               votes_for_label ≥ ceil(W/2); fuse ONLY boxes from voters of that label

Outputs & analyses:
  • Instance-level & screen-level IoU summaries (per method)
  • One combined table (per pattern + overall) with mean±SD for GPT & Crowd
    under SIMPLE and WBF_equal
  • IAA among correct detectors (pairwise F1@IoU≥thr with majority gate)
  • OLS regressions (overall + per-pattern) for SIMPLE & WBF_equal with
    cluster-robust SEs clustered by screen_id

Usage
-----
pip install -r requirements.txt
python localization_analysis.py \
  --csv data/Results.csv \
  --outdir outputs \
  --wbf-iou-thr 0.55 \
  --min-workers 3

"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# =========================
# Config (CLI overrides)
# =========================
DEFAULT_CSV = os.environ.get("LOCALIZE_CSV", "data/Results.csv")  # override via --csv
DEFAULT_OUTDIR = os.environ.get("LOCALIZE_OUT", "outputs")

# Canonical labels & names
PATTERN_LABELS = ["FA-G-PRO", "II-AM-FH", "II-AM-G-SMALL", "II-PRE", "NG-AD"]
PATTERN_NAMES = {
    "FA-G-PRO": "Pay to Avoid Ads",
    "II-AM-FH": "False Hierarchy",
    "II-AM-G-SMALL": "Small Close Button",
    "II-PRE": "Preselection",
    "NG-AD": "Pop-up Ad",
}

# Aggregation / detection gates
WBF_IOU_THR = 0.55       # cluster threshold for WBF-style fusion
MIN_WORKERS = 3          # ≥3 workers on screen

# IAA parameters
IAA_IOU_THR = 0.50        # pairwise F1 IoU threshold
IAA_DET_THR = 0.50        # majority fraction gate for IAA
IAA_REQUIRE_MIN_WORKERS = 3

# Printing / options (can be flipped by CLI flags)
USE_MIXEDLM = False
SAVE_CSVS = True

pd.options.display.float_format = "{:0.3f}".format
warnings.filterwarnings(
    "ignore",
    message="covariance of constraints does not have full rank",
)

# =========================
# Parsing & geometry helpers
# =========================
def safe_eval_cell(x: Any) -> Any:
    if pd.isna(x):
        return []
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    if x2 <= x1 or y2 <= y1 or u2 <= u1 or v2 <= v1:
        return 0.0
    ix1, iy1 = max(x1, u1), max(y1, v1)
    ix2, iy2 = min(x2, u2), min(y2, v2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (x2 - x1) * (y2 - y1)
    b_area = (u2 - u1) * (v2 - v1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0

def relative_size(box: Sequence[float], source_wh: Optional[Sequence[float]]) -> float:
    try:
        if not box or not source_wh:
            return 0.0
        w, h = source_wh
        if w is None or h is None or w <= 0 or h <= 0:
            return 0.0
        area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
        scr = w * h
        return float(area / scr) if scr > 0 else 0.0
    except Exception:
        return 0.0

# =========================
# SIMPLE helpers
# =========================
def per_worker_mean_boxes(crowd_anns_for_label: List[dict]) -> Tuple[List[List[float]], set]:
    """Return (list of per-worker mean boxes, set of detector worker_ids)."""
    by_worker: Dict[str, List[Sequence[float]]] = {}
    for ann in crowd_anns_for_label:
        uid = ann.get("user_id")
        bx = ann.get("bbox")
        if uid is None or bx is None:
            continue
        by_worker.setdefault(uid, []).append(bx)
    means, workers = [], set()
    for uid, boxes in by_worker.items():
        arr = np.array(boxes, dtype=float)
        means.append(arr.mean(axis=0).tolist())
        workers.add(uid)
    return means, workers

def simple_consensus_box(worker_means: List[Sequence[float]]) -> Optional[List[float]]:
    if not worker_means:
        return None
    arr = np.array(worker_means, dtype=float)
    return arr.mean(axis=0).tolist()

def simple_average_box(boxes: List[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    arr = np.array(boxes, dtype=float)
    return arr.mean(axis=0).tolist()

# =========================
# WBF fusers
# =========================
def wbf_fuse_true(
    boxes: List[Sequence[float]],
    scores: Optional[List[float]] = None,
    iou_thr: float = WBF_IOU_THR,
) -> List[Tuple[List[float], float]]:
    """True weighted-boxes fusion (scores weight coordinates), returns (fused_box, total_weight) per cluster."""
    if not boxes:
        return []
    if scores is None or len(scores) != len(boxes):
        scores = [1.0] * len(boxes)
    boxes = [tuple(map(float, b)) for b in boxes]
    scores = [float(s) for s in scores]
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    clusters, reps = [], []
    for idx in order:
        b = boxes[idx]
        matched = -1
        for k, rep in enumerate(reps):
            if iou(b, rep) >= iou_thr:
                matched = k
                break
        if matched == -1:
            clusters.append([idx])
            reps.append(b)
        else:
            clusters[matched].append(idx)
            inds = clusters[matched]
            w = np.array([scores[i] for i in inds], dtype=float)
            arr = np.array([boxes[i] for i in inds], dtype=float)
            reps[matched] = tuple((arr * w[:, None]).sum(axis=0) / w.sum())
    out = []
    for inds in clusters:
        w = np.array([scores[i] for i in inds], dtype=float)
        arr = np.array([boxes[i] for i in inds], dtype=float)
        fused = (arr * w[:, None]).sum(axis=0) / w.sum()
        out.append((fused.tolist(), float(w.sum())))
    return out

def wbf_fuse_equal_true(boxes: List[Sequence[float]], iou_thr: float = WBF_IOU_THR) -> List[Tuple[List[float], float]]:
    """
    Canonical WBF-style clustering with equal weights.
    Returns [(fused_box, votes)] for each cluster.
    """
    if not boxes:
        return []
    boxes = [tuple(map(float, b)) for b in boxes]
    boxes.sort(key=lambda b: (b[0], b[1], b[2], b[3]))  

    clusters: List[List[int]] = []
    reps: List[Tuple[float, float, float, float]] = []

    for idx, b in enumerate(boxes):
        matched = -1
        for k, rep in enumerate(reps):
            if iou(b, rep) >= iou_thr:
                matched = k
                break
        if matched == -1:
            clusters.append([idx])
            reps.append(b)
        else:
            clusters[matched].append(idx)
            arr = np.array([boxes[i] for i in clusters[matched]], dtype=float)
            reps[matched] = tuple(arr.mean(axis=0))  # equal-weight update

    out = []
    for k, inds in enumerate(clusters):
        out.append((list(reps[k]), float(len(inds))))  # votes as "confidence"
    return out

def wbf_top1_equal(boxes: List[Sequence[float]], iou_thr: float = WBF_IOU_THR) -> Optional[List[float]]:
    """Return the top-1 box (highest vote count) from WBF equal-weight clustering."""
    fused = wbf_fuse_equal_true(boxes, iou_thr=iou_thr)
    if not fused:
        return None
    fused.sort(key=lambda t: t[1], reverse=True)  # pick largest-vote cluster
    best_box, _votes = fused[0]
    return best_box

# =========================
# Matching
# =========================
def greedy_match_1to1(
    gt_boxes: List[Sequence[float]],
    pred_boxes: List[Sequence[float]],
) -> List[Optional[List[float]]]:
    """
    Greedy 1–1 matching by IoU (descending). IMPORTANT: we DO NOT drop 0-IoU matches.
    If predictions exist, each GT can be matched (up to min(#gt,#pred)) even if best IoU=0.
    If no predictions, returns None for those GTs.
    """
    if not gt_boxes or not pred_boxes:
        return [None] * len(gt_boxes)
    pairs: List[Tuple[float, int, int]] = []
    for i, gt in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            pairs.append((iou(gt, pb), i, j))
    pairs.sort(reverse=True)
    used_gt, used_pb = set(), set()
    out: List[Optional[List[float]]] = [None] * len(gt_boxes)
    for sim, i, j in pairs:
        if i in used_gt or j in used_pb:
            continue
        out[i] = list(map(float, pred_boxes[j]))
        used_gt.add(i)
        used_pb.add(j)
        if len(used_gt) == len(gt_boxes):
            break
    return out

# =========================
# Majority gate
# =========================
def has_strict_majority(n_workers: int, n_votes_for_label: int) -> bool:
    """At least half: votes ≥ ceil(W/2)."""
    need = math.ceil(n_workers / 2)
    return n_votes_for_label >= need

# =========================
# Dataset builders (SIMPLE / WBF_equal)
# =========================
def build_dataset(
    df: pd.DataFrame,
    pattern_labels: List[str],
    agg: str,
    wbf_iou_thr: float,
    min_workers: int = MIN_WORKERS,
) -> pd.DataFrame:
    """
    Build tidy per-expert-instance IoU rows for TPs only, keeping IoU=0.
    GPT TP gate  : any GPT box for that label on that screen.
    Crowd TP gate: ≥ min_workers AND votes_for_label ≥ ceil(W/2); fuse only voters of that label.
    """
    rows = []
    instance_id = 0
    for _, R in df.iterrows():
        screen_id = R.get("screen_id")
        group = R.get("group", "unknown")

        experts = [
            a
            for a in R["expert_annotations"]
            if a and a.get("bbox") is not None and a.get("label") in pattern_labels
        ]
        crowds = [
            a
            for a in R["crowd_annotations"]
            if a and a.get("bbox") is not None and a.get("label") in pattern_labels
        ]
        gpts = [
            a
            for a in R["gpt_annotations"]
            if a and a.get("bbox") is not None and a.get("label") in pattern_labels
        ]

        all_workers = {a.get("user_id") for a in R["crowd_annotations"] if a.get("user_id") is not None}
        n_workers = len(all_workers)

        for lab in pattern_labels:
            expert_instances = [a for a in experts if a["label"] == lab]
            if not expert_instances:
                continue

            gt_list = [e["bbox"] for e in expert_instances]
            size_rel_list = [relative_size(e["bbox"], e.get("source_wh")) for e in expert_instances]

            # GPT candidates
            gpt_boxes = [a["bbox"] for a in gpts if a["label"] == lab]
            if gpt_boxes:
                if agg == "wbf_eq":
                    top = wbf_top1_equal(gpt_boxes, iou_thr=wbf_iou_thr)
                    gpt_cand = [top] if top is not None else []
                elif agg == "simple":
                    avg = simple_average_box(gpt_boxes)
                    gpt_cand = [avg] if avg is not None else []
                else:
                    gpt_cand = gpt_boxes
            else:
                gpt_cand = []

            # Crowd candidates: only voters for this label
            crowd_for_label = [a for a in crowds if a["label"] == lab]
            voters_for_label = {a.get("user_id") for a in crowd_for_label if a.get("user_id") is not None}
            crowd_ok = (n_workers >= min_workers) and has_strict_majority(n_workers, len(voters_for_label))

            if agg == "wbf_eq":
                if crowd_ok:
                    c_boxes = [a["bbox"] for a in crowd_for_label if a.get("bbox") is not None]
                    top = wbf_top1_equal(c_boxes, iou_thr=wbf_iou_thr)
                    crowd_cand = [top] if top is not None else []
                else:
                    crowd_cand = []
            else:  # simple
                if crowd_ok:
                    worker_means, _ = per_worker_mean_boxes(crowd_for_label)
                    consensus = simple_consensus_box(worker_means)
                    crowd_cand = [consensus] if consensus is not None else []
                else:
                    crowd_cand = []

            # 1–1 matching; KEEP IoU=0
            if gpt_cand:
                matches = greedy_match_1to1(gt_list, gpt_cand)
                for gt, sz, mb in zip(gt_list, size_rel_list, matches):
                    if mb is not None:
                        instance_id += 1
                        rows.append(
                            {
                                "instance_id": instance_id,
                                "screen_id": screen_id,
                                "group": group,
                                "label": lab,
                                "annotator": "gpt",
                                "iou": iou(gt, mb),
                                "size_rel": sz,
                                "agg_method": agg,
                            }
                        )
            if crowd_cand:
                matches = greedy_match_1to1(gt_list, crowd_cand)
                for gt, sz, mb in zip(gt_list, size_rel_list, matches):
                    if mb is not None:
                        instance_id += 1
                        rows.append(
                            {
                                "instance_id": instance_id,
                                "screen_id": screen_id,
                                "group": group,
                                "label": lab,
                                "annotator": "crowd",
                                "iou": iou(gt, mb),
                                "size_rel": sz,
                                "agg_method": agg,
                            }
                        )

    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return tidy
    tidy["is_crowd"] = (tidy["annotator"] == "crowd").astype(int)
    tidy["label"] = pd.Categorical(tidy["label"], categories=PATTERN_LABELS, ordered=False)
    tidy["group"] = tidy["group"].astype("category")
    mu = tidy["size_rel"].mean()
    sd = tidy["size_rel"].std(ddof=0)
    tidy["size_z"] = 0.0 if (not np.isfinite(sd) or sd == 0) else (tidy["size_rel"] - mu) / sd
    return tidy

# =========================
# OLS helpers
# =========================
def run_overall_ols(D: pd.DataFrame):
    if D.empty:
        return None
    formula = "iou ~ size_z + is_crowd + is_crowd:size_z + C(label) + is_crowd:C(group)"
    m = smf.ols(formula, data=D).fit(cov_type="cluster", cov_kwds={"groups": D["screen_id"]})
    return m

def run_per_pattern_ols_full(D: pd.DataFrame, tag: str):
    print("\n" + "=" * 80)
    print(f"PER-PATTERN OLS — {tag}")
    print("=" * 80)
    any_printed = False
    for code in PATTERN_LABELS:
        sub = D[D["label"] == code].copy()
        if sub.empty or sub["screen_id"].nunique() < 2:
            print(f"[skip] {PATTERN_NAMES.get(code, code)} — insufficient data")
            continue
        m = smf.ols(
            "iou ~ size_z + is_crowd + is_crowd:size_z + is_crowd:C(group)",
            data=sub,
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["screen_id"]})
        print("\n" + "-" * 80)
        print(
            f"Pattern: {PATTERN_NAMES.get(code, code)} ({code}) | rows={len(sub)} | screens={sub['screen_id'].nunique()}"
        )
        print("-" * 80)
        print(m.summary())
        any_printed = True
    if not any_printed:
        print("No per-pattern fits could be estimated.")

def maybe_mixedlm_screen(D: pd.DataFrame, tag: str):
    if not USE_MIXEDLM or D.empty or D["screen_id"].nunique() < 2:
        return
    try:
        import statsmodels.api as sm
        print("\n" + "=" * 80)
        print(f"MIXED EFFECTS (optional) — Random intercept by screen_id — {tag}")
        print("=" * 80)
        md = sm.MixedLM.from_formula(
            "iou ~ size_z + is_crowd + is_crowd:size_z + C(label) + is_crowd:C(group)",
            data=D,
            groups=D["screen_id"],
            re_formula="1",
        )
        mdf = md.fit(method="lbfgs", reml=True)
        print(mdf.summary())
        print("[Info] Worker RE not included: aggregation removes worker IDs.")
    except Exception as e:
        print(f"[MixedLM skipped due to error] {e}")

def pattern_ranking_at_means(m, D: pd.DataFrame) -> pd.DataFrame:
    if m is None or D.empty:
        return pd.DataFrame()
    ref_group = D["group"].cat.categories[0] if hasattr(D["group"], "cat") else D["group"].unique()[0]
    rows = []
    for lab in D["label"].cat.categories:
        for annot in ["gpt", "crowd"]:
            is_crowd = 1 if annot == "crowd" else 0
            X = pd.DataFrame({"size_z": [0], "is_crowd": [is_crowd], "label": [lab], "group": [ref_group]})
            mu_pred = m.get_prediction(X).predicted_mean
            mu = float(mu_pred.iloc[0]) if hasattr(mu_pred, "iloc") else float(np.array(mu_pred).ravel()[0])
            rows.append({"label": lab, "annotator": annot, "pred_mu": mu})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pivot = out.pivot(index="label", columns="annotator", values="pred_mu").reset_index()
    pivot["Pattern"] = pivot["label"].map(PATTERN_NAMES).fillna(pivot["label"])
    pivot["Avg_mu"] = pivot[["gpt", "crowd"]].mean(axis=1)
    return pivot.sort_values("Avg_mu", ascending=False)

def quick_desc_generic(title: str, D: pd.DataFrame, iou_col: str = "iou"):
    print("\n" + "=" * 80)
    print(f"DATASET SUMMARY — {title}")
    print("=" * 80)
    if D.empty:
        print("[No data]")
        return
    inst = D[iou_col]
    c_inst = D[D["annotator"] == "crowd"][iou_col]
    g_inst = D[D["annotator"] == "gpt"][iou_col]
    c_scr_means = D[D["annotator"] == "crowd"].groupby("screen_id")[iou_col].mean()
    g_scr_means = D[D["annotator"] == "gpt"].groupby("screen_id")[iou_col].mean()
    t, p = stats.ttest_ind(c_inst, g_inst, equal_var=False, nan_policy="omit")
    print(
        f"Rows: {len(D)} | Screens: {D['screen_id'].nunique()} | "
        f"Expert instances: {D['instance_id'].nunique() if 'instance_id' in D.columns else 'n/a'}"
    )
    print(f"Crowd rows: {len(c_inst)} | GPT rows: {len(g_inst)}\n")
    print("IoU (mean ± SD) — instance-level")
    print(f"  Crowd: {c_inst.mean():.3f} ± {c_inst.std(ddof=1):.3f}")
    print(f"  GPT:   {g_inst.mean():.3f} ± {g_inst.std(ddof=1):.3f}")
    print(f"Unpaired t-test (instance means, Crowd vs GPT): t={t:.2f}, p={p:.3g}\n")
    print("IoU (mean ± SD) — screen-level (avg within screen first)")
    print(f"  Crowd: {c_scr_means.mean():.3f} ± {c_scr_means.std(ddof=1):.3f}  (screens={len(c_scr_means)})")
    print(f"  GPT:   {g_scr_means.mean():.3f} ± {g_scr_means.std(ddof=1):.3f}  (screens={len(g_scr_means)})")

# =========================
# IAA among correct detectors (pairwise F1)
# =========================
def _parse_list(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

def _iou_pair(a: List[float], b: List[float]) -> float:
    return iou(a, b)

def _greedy_match_count(A: List[List[float]], B: List[List[float]], thr: float) -> int:
    if not A or not B:
        return 0
    used = [False] * len(B)
    m = 0
    for a in A:
        best_j, best_iou = -1, -1.0
        for j, b in enumerate(B):
            if used[j]:
                continue
            i = _iou_pair(a, b)
            if i >= thr and i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0:
            used[best_j] = True
            m += 1
    return m

def _pairwise_f1(A: List[List[float]], B: List[List[float]], thr: float) -> Tuple[float, int, int]:
    nA, nB = len(A), len(B)
    if nA == 0 and nB == 0:
        return 1.0, 0, 0
    m = _greedy_match_count(A, B, thr)
    denom = nA + nB
    f1 = (2.0 * m / denom) if denom > 0 else 0.0
    return f1, nA, nB

def iaa_among_correct_detectors_df(
    df_all: pd.DataFrame,
    labels=PATTERN_LABELS,
    iou_thr: float = IAA_IOU_THR,
    detection_thr: float = IAA_DET_THR,
    min_workers_screen: int = IAA_REQUIRE_MIN_WORKERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_per_screen = []
    stats = {lab: {"matches": 0, "denom": 0, "sum_f1": 0.0, "pairs": 0, "screens": 0} for lab in labels}
    overall = {"matches": 0, "denom": 0, "sum_f1": 0.0, "pairs": 0, "screens": 0}

    for _, row in df_all.iterrows():
        screen_id = row.get("screen_id")
        crowd = _parse_list(row.get("crowd_annotations"))
        expert = _parse_list(row.get("expert_annotations"))

        workers = sorted({it.get("user_id") for it in crowd if it.get("user_id")})
        n_workers = len(workers)
        if n_workers < min_workers_screen:
            continue

        # per-worker boxes
        boxes_by_worker = defaultdict(lambda: {lab: [] for lab in labels})
        for it in crowd:
            uid = it.get("user_id")
            lab = it.get("label")
            bbox = it.get("bbox")
            if (uid is None) or (lab not in labels) or (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
                continue
            boxes_by_worker[uid][lab].append([float(x) for x in bbox])

        # GT presence per label
        gt_present = {lab: False for lab in labels}
        for it in expert:
            lab = it.get("label")
            bbox = it.get("bbox")
            if (lab in labels) and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                gt_present[lab] = True

        for lab in labels:
            if not gt_present[lab]:
                continue
            detectors = [uid for uid in workers if len(boxes_by_worker[uid][lab]) > 0]
            if len(detectors) < 2:
                continue
            if (len(detectors) / n_workers) < detection_thr:
                continue

            pair_f1s = []
            total_matches, total_boxes = 0, 0
            for u, v in combinations(detectors, 2):
                f1, nA, nB = _pairwise_f1(boxes_by_worker[u][lab], boxes_by_worker[v][lab], iou_thr)
                m = f1 * (nA + nB) / 2.0
                total_matches += m
                total_boxes += (nA + nB)
                pair_f1s.append(f1)

            stats[lab]["screens"] += 1
            stats[lab]["matches"] += total_matches
            stats[lab]["denom"] += total_boxes
            stats[lab]["sum_f1"] += sum(pair_f1s)
            stats[lab]["pairs"] += len(pair_f1s)

            overall["screens"] += 1
            overall["matches"] += total_matches
            overall["denom"] += total_boxes
            overall["sum_f1"] += sum(pair_f1s)
            overall["pairs"] += len(pair_f1s)

            rows_per_screen.append(
                {
                    "screen_id": screen_id,
                    "label": lab,
                    "label_name": PATTERN_NAMES.get(lab, lab),
                    "num_workers_on_screen": n_workers,
                    "num_detectors": len(detectors),
                    "detectors_fraction": len(detectors) / n_workers,
                    "pairwise_F1_mean": (np.mean(pair_f1s) if pair_f1s else np.nan),
                    "pairwise_F1_std": (np.std(pair_f1s, ddof=1) if len(pair_f1s) > 1 else np.nan),
                }
            )

    per_screen_df = pd.DataFrame(rows_per_screen)
    rows = []
    for lab in labels:
        s = stats[lab]
        micro = (2.0 * s["matches"] / s["denom"]) if s["denom"] > 0 else np.nan
        macro = (s["sum_f1"] / s["pairs"]) if s["pairs"] > 0 else np.nan
        rows.append([PATTERN_NAMES.get(lab, lab), s["screens"], s["pairs"], micro, macro])
    micro_o = (2.0 * overall["matches"] / overall["denom"]) if overall["denom"] > 0 else np.nan
    macro_o = (overall["sum_f1"] / overall["pairs"]) if overall["pairs"] > 0 else np.nan
    rows.append(["Overall", overall["screens"], overall["pairs"], micro_o, macro_o])
    summary = pd.DataFrame(rows, columns=["Pattern", "Screens_used", "Worker_pairs", "Micro_F1", "Macro_F1"])
    return per_screen_df, summary

# =========================
# Pipeline
# =========================
def run_pipeline(
    csv_path: str = DEFAULT_CSV,
    wbf_iou_thr: float = WBF_IOU_THR,
    min_workers: int = MIN_WORKERS,
    outdir: str | Path = DEFAULT_OUTDIR,
):
    np.random.default_rng(2025)
    outdir = Path(outdir)
    if SAVE_CSVS:
        outdir.mkdir(parents=True, exist_ok=True)

    # Load
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    for col in ["crowd_annotations", "expert_annotations", "gpt_annotations"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = df[col].apply(safe_eval_cell)
    if "screen_id" not in df.columns:
        raise ValueError("Missing required column: screen_id")
    if "group" not in df.columns:
        df["group"] = "unknown"
    df["group"] = df["group"].astype(str)

    # ---------- SIMPLE & WBF_equal ----------
    reg_simple = build_dataset(df, PATTERN_LABELS, agg="simple", wbf_iou_thr=wbf_iou_thr, min_workers=min_workers)
    reg_wbfeq = build_dataset(df, PATTERN_LABELS, agg="wbf_eq", wbf_iou_thr=wbf_iou_thr, min_workers=min_workers)
    for D in (reg_simple, reg_wbfeq):
        if not D.empty:
            D["group"] = D["group"].astype("category")
            D["label"] = pd.Categorical(D["label"], categories=PATTERN_LABELS, ordered=False)

    # Instance+screen summaries
    quick_desc_generic("SIMPLE consensus", reg_simple, iou_col="iou")
    quick_desc_generic("WBF_equal (real WBF, top-1 equal votes)", reg_wbfeq, iou_col="iou")

    # Per-pattern method summaries
    sum_simple = rename_with_tag(summary_from_tidy(reg_simple, iou_col="iou"), "SIMPLE")
    sum_wbfeq = rename_with_tag(summary_from_tidy(reg_wbfeq, iou_col="iou"), "WBF_equal")

    # ---------- ONE COMBINED TABLE ----------
    combined = sum_simple.merge(sum_wbfeq, on="Pattern", how="outer")
    pat_order = [PATTERN_NAMES[p] for p in PATTERN_LABELS] + ["Overall"]
    combined["Pattern"] = pd.Categorical(combined["Pattern"], categories=pat_order, ordered=True)
    combined = combined.sort_values("Pattern").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("ONE TABLE — Per-pattern mean±SD for GPT & Crowd across methods")
    print("=" * 80)
    cols_order = [
        "Pattern",
        "N_SIMPLE",
        "N_WBF_equal",
        "Crowd_IoU_SIMPLE",
        "Crowd_SD_SIMPLE",
        "GPT_IoU_SIMPLE",
        "GPT_SD_SIMPLE",
        "Crowd_IoU_WBF_equal",
        "Crowd_SD_WBF_equal",
        "GPT_IoU_WBF_equal",
        "GPT_SD_WBF_equal",
    ]
    present_cols = [c for c in cols_order if c in combined.columns]
    print(combined[present_cols].to_string(index=False))
    if SAVE_CSVS:
        combined.to_csv(outdir / "summary_per_pattern.csv", index=False)
        print(f"\nSaved: {outdir / 'summary_per_pattern.csv'}")

    # ---------- IAA among correct detectors ----------
    per_screen_iaa, summary_iaa = iaa_among_correct_detectors_df(
        df, labels=PATTERN_LABELS, iou_thr=IAA_IOU_THR, detection_thr=IAA_DET_THR, min_workers_screen=IAA_REQUIRE_MIN_WORKERS
    )
    print(
        "\nIAA among CORRECT detectors (pairwise F1@IoU≥{:.2f}, majority gate, ≥{} workers)".format(
            IAA_IOU_THR, IAA_REQUIRE_MIN_WORKERS
        )
    )
    print(summary_iaa.to_string(index=False))
    if SAVE_CSVS:
        per_screen_iaa.to_csv(outdir / "iaa_correct_detectors_per_screen.csv", index=False)
        summary_iaa.to_csv(outdir / "iaa_correct_detectors_summary.csv", index=False)
        print(f"\nSaved: {outdir / 'iaa_correct_detectors_per_screen.csv'}, {outdir / 'iaa_correct_detectors_summary.csv'}")

    # ---------- OLS (SIMPLE & WBF_equal) ----------
    for tag, D in [("SIMPLE", reg_simple), ("WBF_equal", reg_wbfeq)]:
        if D.empty:
            print(f"\n[{tag}] No rows; skipping regressions.")
            continue
        print("\n" + "=" * 80)
        print(f"OVERALL OLS — {tag}")
        print("=" * 80)
        m = run_overall_ols(D)
        print(m.summary())
        rank = pattern_ranking_at_means(m, D)
        if not rank.empty:
            print("\nPredicted IoU at mean size (by pattern, ref=first group):")
            print(rank[["Pattern", "gpt", "crowd", "Avg_mu"]].to_string(index=False))
        run_per_pattern_ols_full(D, tag)
        maybe_mixedlm_screen(D, tag)

    print("\n✓ Done.")
    return {
        "simple": reg_simple,
        "wbfeq": reg_wbfeq,
        "combined": combined,
        "iaa_summary": summary_iaa,
    }

# =========================
# CLI
# =========================
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="All-in-one: SIMPLE, WBF_equal + IAA + OLS"
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to NORMALIZED.csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory to save CSV outputs")
    ap.add_argument("--wbf-iou-thr", type=float, default=WBF_IOU_THR, help="WBF IoU threshold (clustering)")
    ap.add_argument("--min-workers", type=int, default=MIN_WORKERS, help="Min distinct workers on a screen")
    ap.add_argument("--iaa-iou-thr", type=float, default=IAA_IOU_THR, help="IAA pairwise F1 IoU threshold")
    ap.add_argument("--iaa-det-frac", type=float, default=IAA_DET_THR, help="IAA majority fraction (detectors/workers)")
    ap.add_argument("--iaa-min-workers", type=int, default=IAA_REQUIRE_MIN_WORKERS, help="IAA: min workers on screen")
    ap.add_argument("--no-save", action="store_true", help="Do not save CSV outputs")
    ap.add_argument("--mixedlm", action="store_true", help="Also print MixedLM (random intercept by screen_id)")
    args = ap.parse_args()

    global SAVE_CSVS, USE_MIXEDLM, IAA_IOU_THR, IAA_DET_THR, IAA_REQUIRE_MIN_WORKERS
    SAVE_CSVS = not args.no_save
    USE_MIXEDLM = args.mixedlm
    IAA_IOU_THR = args.iaa_iou_thr
    IAA_DET_THR = args.iaa_det_frac
    IAA_REQUIRE_MIN_WORKERS = args.iaa_min_workers

    run_pipeline(
        csv_path=args.csv,
        wbf_iou_thr=args.wbf_iou_thr,
        min_workers=args.min_workers,
        outdir=args.outdir,
    )

if __name__ == "__main__":
    main()
