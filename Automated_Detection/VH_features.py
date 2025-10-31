
"""
Collision-Free View Hierarchy (VH) Feature Extraction
====================================================

Extracts per-screen, collision-free VH tokens + heuristics from MobileViews & RICO
JSONs. Matches the logic used in your training pipeline:

Features emitted (as set of strings per screen):
  - class=WidgetClass
  - area_{≤0.01, ≤0.025, ≤0.05, ≤0.1, ≤0.2, ≤0.4, >0.4}
  - xquad_{0..3}, yquad_{0..3} (center-based 4×4 grid)
  - tok=word   (letters-only, a–z/A–Z; no digits/underscores)

Heuristics (extra numeric columns):
  - small_close: 1 if element with "close" semantics AND (≤60×60 px OR ≤1.2% area)
  - popup_ad:    1 if foreground activity looks like an ad OR an ad view ≥30% screen

Output files:
  - metadata_with_tokens.pkl  : DataFrame with columns:
      ['screen','dataset','vh_tokens','small_close','popup_ad']
      vh_tokens is a Python set of strings (collision-free)
  - (optional) vh_vocab_collision_free.csv : Reference vocabulary over all screens


Usage:
  python extract_vh_tokens.py \
    --mobile-root data/cleaned_dataset \
    --rico-root data/rico_semantic_annotations \
    --screens-csv data/GPT_features_39_ALL_WITH_JSON_NOHASH.csv \
    --out-pkl outputs/metadata_with_tokens.pkl \
    --out-vocab outputs/vh_vocab_collision_free.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer


# ==================== CONSTANTS / REGEX ====================


TOKEN_RE = re.compile(r"[a-zA-Z]+")
_AD_ACTIVITY = re.compile(r"(?:adactivity|reward(?:ed)?video|unityads)", re.I)
_AD_VIEW = re.compile(r"(interstitial|unityads|reward)", re.I)


# ==================== HELPERS ====================

def rect(b) -> Tuple[float, float, float, float]:
    """Normalize bounds to (x0, y0, x1, y1) for both formats."""
    if isinstance(b[0], list):
        (x0, y0), (x1, y1) = b
    else:
        x0, y0, x1, y1 = b
    return float(x0), float(y0), float(x1), float(y1)


def flatten_rico(node: Dict, acc: List[Dict]) -> None:
    """DFS-flatten a RICO tree into a MobileViews-like list of nodes."""
    d = {k: node.get(k) for k in (
        "class", "resource-id", "resource_id", "text", "content_description", "bounds"
    )}
    rid = d.pop("resource-id", None)
    if rid is not None and d.get("resource_id") is None:
        d["resource_id"] = rid
    acc.append(d)
    for ch in node.get("children", []) or []:
        flatten_rico(ch, acc)


def bucket_area_ratio(v: float, edges=(0.01, 0.025, 0.05, 0.10, 0.20, 0.40)) -> str:
    """Bucket area ratios for interpretability."""
    for e in edges:
        if v <= e:
            return f"≤{e}"
    return f">{edges[-1]}"


def quad_bins(x0: float, y0: float, x1: float, y1: float, W: float, H: float) -> Tuple[int, int]:
    """
    Center-based quadrant bins (4×4), clipped to [0,3].
    Matches the training pipeline exactly.
    """
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    qx = min(int(max(0, min(cx, W - 1)) / max(W, 1) * 4), 3)
    qy = min(int(max(0, min(cy, H - 1)) / max(H, 1) * 4), 3)
    return qx, qy


def detect_small_close(v: Dict, W: float, H: float) -> bool:
    """
    Detect small close buttons using both absolute size and relative area.
    - Must have close semantics (resource_id/text/content_description)
    - AND (≤60×60 px OR ≤1.2% of screen area)
    """
    b = v.get("bounds")
    if not b:
        return False

    rid = (v.get("resource_id") or "").lower()
    has_close_semantic = (
        any(k in rid for k in ("close", "btn_close", "icon_smallclose", "ic_close", "close_btn"))
        or "close" in (v.get("content_description") or "").lower()
        or "close" in (v.get("text") or "").lower()
    )
    if not has_close_semantic:
        return False

    x0, y0, x1, y1 = rect(b)
    w, h = abs(x1 - x0), abs(y1 - y0)
    area_ratio = (w * h) / max(W * H, 1.0)
    return (w <= 60 and h <= 60) or (area_ratio <= 0.012)


def detect_popup_ad(state: Dict, views: List[Dict], W: float, H: float) -> int:
    """Detect popup ads via foreground activity or large ad views (≥30% screen)."""
    if _AD_ACTIVITY.search(state.get("foreground_activity", "") or ""):
        return 1
    for v in views:
        cand = f"{v.get('resource_id','')} {v.get('class','')}"
        if "popup" in cand.lower() and not _AD_VIEW.search(cand):
            continue
        if _AD_VIEW.search(cand) and v.get("bounds"):
            x0, y0, x1, y1 = rect(v["bounds"])
            if abs(x1 - x0) * abs(y1 - y0) >= 0.30 * W * H:
                return 1
    return 0


def extract_tokens_from_state(state: Dict) -> Tuple[Set[str], int, int]:
    """
    Core extractor — EXACTLY mirrors the training pipeline logic.
    Returns: (tokens_set, small_close_flag, popup_ad_flag)
    """
    # Determine format & canvas size
    if isinstance(state.get("views"), list):
        views = state["views"]
        W = int(state.get("width", 1080) or 1080)
        H = int(state.get("height", 1920) or 1920)
    else:
        views = []
        flatten_rico(state, views)
        try:
            x0, y0, x1, y1 = rect(state["bounds"])
            W = int(abs(x1 - x0)) or 1080
            H = int(abs(y1 - y0)) or 1920
        except Exception:
            W, H = 1080, 1920

    if W <= 0 or H <= 0:
        W, H = 1080, 1920

    toks: Set[str] = set()
    small_flag = 0
    for v in views:
        b = v.get("bounds")
        if not b:
            continue

        if detect_small_close(v, W, H):
            small_flag = 1

        # class
        toks.add(f"class={v.get('class', 'NONE')}")

        # area bucket
        x0, y0, x1, y1 = rect(b)
        w, h = abs(x1 - x0), abs(y1 - y0)
        area_ratio = (w * h) / max(W * H, 1.0)
        toks.add(f"area_{bucket_area_ratio(area_ratio)}")

        # center-based quadrants
        qx, qy = quad_bins(x0, y0, x1, y1, W, H)
        toks.add(f"xquad_{qx}")
        toks.add(f"yquad_{qy}")

        # text/content/resource tokens
        for k in ("text", "content_description", "resource_id"):
            val = (v.get(k) or "")
            for m in TOKEN_RE.findall(str(val).lower()):
                toks.add(f"tok={m}")

    pflag = detect_popup_ad(state, views, W, H)
    return toks, int(small_flag), int(pflag)


# ==================== DATA DISCOVERY ====================

def iter_mobile_jsons(root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (screen_id, path) for MobileViews dataset."""
    if not root.exists():
        return
    for app_dir in root.iterdir():
        states = app_dir / "states"
        if not states.is_dir():
            continue
        for p in states.glob("state_*.json"):
            num = p.stem.split("_", 1)[-1]
            screen = f"{app_dir.name}_{num}"
            yield screen, p


def iter_rico_jsons(root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (screen_id, path) for RICO dataset."""
    if not root.exists():
        return
    for p in root.glob("*.json"):
        yield p.stem, p


def infer_dataset_from_screen(screen: str) -> str:
    """RICO screens are digits-only; MV screens contain an underscore."""
    return "RICO" if str(screen).isdigit() else "MV"


def path_for_screen(screen: str, mobile_root: Path, rico_root: Path) -> Path:
    """Resolve JSON path from screen id."""
    if infer_dataset_from_screen(screen) == "RICO":
        return rico_root / f"{screen}.json"
    app, num = str(screen).rsplit("_", 1)
    return mobile_root / app / "states" / f"state_{num}.json"


# ==================== MAIN ====================

def write_run_info(path: Path):
    def v(pkg):
        try:
            return __import__(pkg).__version__
        except Exception:
            return "unknown"
    txt = (
        f"python_version  : {__import__('sys').version}\n"
        f"pandas          : {v('pandas')}\n"
        f"numpy           : {v('numpy')}\n"
        f"tqdm            : {v('tqdm')}\n"
        f"sklearn         : {v('sklearn')}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")


def build_reference_vocab(df: pd.DataFrame, out_csv: Path):
    """Optional: document the learned VH vocabulary."""
    bags = [{t: 1 for t in toks} for toks in df["vh_tokens"]]
    dv = DictVectorizer(sparse=True, sort=True)  # deterministic order
    dv.fit(bags)
    names = list(dv.get_feature_names_out())
    vocab_df = pd.DataFrame({
        "feature_name": [f"vh::{n}" for n in names],
        "raw_token": names,
        "group": [
            "class" if n.startswith("class=") else
            "tok" if n.startswith("tok=") else
            "area" if n.startswith("area_") else
            "xquad" if n.startswith("xquad_") else
            "yquad" if n.startswith("yquad_") else
            "other"
            for n in names
        ]
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    vocab_df.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Extract collision-free VH tokens + heuristics (MobileViews & RICO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--mobile-root", type=Path, default=Path("data/cleaned_dataset"),
                    help="MobileViews root directory")
    ap.add_argument("--rico-root", type=Path, default=Path("data/rico_semantic_annotations"),
                    help="RICO semantic annotations directory")
    ap.add_argument("--screens-csv", type=Path, default=None,
                    help="Optional CSV with column 'screen' (and optional 'dataset'). "
                         "If omitted, auto-discovers all JSONs from both roots.")
    ap.add_argument("--out-pkl", type=Path, default=Path("outputs/metadata_with_tokens.pkl"),
                    help="Output pickle with columns: screen,dataset,vh_tokens,small_close,popup_ad")
    ap.add_argument("--out-vocab", type=Path, default=None,
                    help="Optional CSV to write a reference vocabulary (DictVectorizer over tokens)")
    ap.add_argument("--run-info", type=Path, default=Path("outputs/RUN_INFO.txt"),
                    help="Write library versions for reproducibility")
    args = ap.parse_args()

    # Build job list
    jobs: List[Tuple[str, Path]] = []

    if args.screens_csv:
        if not args.screens_csv.exists():
            raise FileNotFoundError(f"screens-csv not found: {args.screens_csv}")
        df = pd.read_csv(args.screens_csv)
        if "screen" not in df.columns:
            raise ValueError("screens-csv must include a 'screen' column")
        # Resolve paths; keep alignment even if missing (we'll mark later)
        for scr in df["screen"].astype(str).tolist():
            p = path_for_screen(scr, args.mobile_root, args.rico_root)
            jobs.append((scr, p))
    else:
        jobs.extend(iter_mobile_jsons(args.mobile_root))
        jobs.extend(iter_rico_jsons(args.rico_root))

    if not jobs:
        raise SystemExit("No JSON files found. Check --mobile-root/--rico-root or provide --screens-csv.")

    # Extract
    screens, datasets, tokens_col, small_flags, popup_flags = [], [], [], [], []
    miss_mv = miss_ri = 0

    print(f"Extracting VH tokens from {len(jobs)} screens...")
    for scr, path in tqdm(jobs, desc="VH"):
        ds = infer_dataset_from_screen(scr)
        screens.append(scr)
        datasets.append(ds)

        if not path.exists():
            tokens_col.append(set())
            small_flags.append(0)
            popup_flags.append(0)
            if ds == "MV":
                miss_mv += 1
            else:
                miss_ri += 1
            continue

        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            toks, sflag, pflag = extract_tokens_from_state(state)
            tokens_col.append(toks)
            small_flags.append(sflag)
            popup_flags.append(pflag)
        except Exception:
            tokens_col.append(set())
            small_flags.append(0)
            popup_flags.append(0)
            if ds == "MV":
                miss_mv += 1
            else:
                miss_ri += 1

    # Build dataframe (sorted for reproducibility)
    out_df = pd.DataFrame({
        "screen": screens,
        "dataset": datasets,
        "vh_tokens": tokens_col,
        "small_close": np.array(small_flags, dtype=np.int8),
        "popup_ad": np.array(popup_flags, dtype=np.int8),
    }).sort_values("screen").reset_index(drop=True)

    # Save outputs
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(args.out_pkl)
    print(f"Wrote {len(out_df)} rows to {args.out_pkl}")

    if args.out_vocab is not None:
        build_reference_vocab(out_df, args.out_vocab)
        print(f"Wrote reference vocab to {args.out_vocab}")

    write_run_info(args.run_info)
    print(f"Saved run info to {args.run_info}")
    print(f"Missing JSONs — MV: {miss_mv} | RICO: {miss_ri}")


if __name__ == "__main__":
    main()
