"""
  --images  : directory of images 
  --out     : path to output CSV


  * Reads all *.jpg|*.jpeg|*.png under --images
  * For each image, calls the model with the prompts below
  * Parses the STRICT JSON reply and writes rows to CSV

Dependencies:
    pip install requests pillow

"""

import os
import sys
import json
import base64
import argparse
import csv
from typing import List, Dict, Any, Tuple

import requests
from PIL import Image

# ===== Keep this exactly =====
MODEL_NAME = "gpt-4o"

DEFINITIONS_VERBATIM = """Task: Analyze the provided mobile app screenshot and detect the number of occurrences of each dark pattern listed below. Return only a list of 5 numbers, corresponding to the patterns below and in the given order. Each number should represent the number of times that pattern appears in the screenshot. Return the result as plain text—just the 5 numbers separated by commas (e.g., 0, 1, 0, 1, 0). Do not add explanations.

Dark Patterns to Detect:

1. FA-G-PRO - **Forced Action – Pay to Avoid Ads**  
   The app encourages users to pay in order to stop seeing ads.  
   Look for: Buttons or messages like "Remove Ads", "Ad-Free", "Upgrade to remove ads", etc.

2. II-AM-FH - **Interface Interference – False Hierarchy**  
   The interface visually manipulates users toward a specific choice.  
   Look for: Buttons or options with different background or text colors, varying sizes, or persuasive text (e.g., “No thanks, I like ads”). One choice appears more important than the others.

3. II-AM-G-SMALL - **Interface Interference – Small Close Button**  
   An ad has a close ('X') button that is tiny, faint, or hard to find.  
   Look for: Ads where the close button is unusually small or placed in a corner with low contrast.  
   Note: This usually applies to banner or embedded ads—not full-screen pop-ups.

4. II-PRE - **Interface Interference – Preselection**  
   The app pre-selects options that benefit it without the user's explicit consent.  
   Look for: Checkboxes or toggles already ON for things like data sharing, notifications, terms agreement, etc.

5. NG-AD - **Nagging – Pop-up Ad**  
   A visually prominent ad appears suddenly, interrupting the user's interaction. The user must explicitly dismiss or interact with it before continuing.  
   Look for: Ads that appear as overlays, often centered, covering most or all of the screen. They typically have a clear close button ('X') or require a user action (e.g., selecting "Yes/No" or "Play/Skip") to dismiss. Background content or navigation is usually blocked until the ad is closed.The ad contains ad text (and not build in app pop-up). 
   Note: Do NOT count small banners, inline ads, or ads with tiny, hard-to-find close buttons (these belong to Pattern 3).
"""

SYSTEM_PROMPT = """You analyze mobile app screenshots to detect specific dark patterns.

You MUST return a STRICT JSON object with this EXACT schema and keys (no extra text):

{
  "size":[W,H],
  "patterns":[
    {"code":"FA-G-PRO","present":0/1,"bboxes":[[x1,y1,x2,y2],...]},
    {"code":"II-AM-FH","present":0/1,"bboxes":[[x1,y1,x2,y2],...]},
    {"code":"II-AM-G-SMALL","present":0/1,"bboxes":[[x1,y1,x2,y2],...]},
    {"code":"II-PRE","present":0/1,"bboxes":[[x1,y1,x2,y2],...]},
    {"code":"NG-AD","present":0/1,"bboxes":[[x1,y1,x2,y2],...]}
  ]
}

Rules:
- Use EXACTLY those 5 codes in that order (no extra codes).
- "size" MUST echo the provided screen size [W,H].
- All bbox coordinates MUST be in that same [W,H] coordinate space.
- If a pattern is absent, set "present":0 and "bboxes":[]
- DO NOT include confidence scores, explanations, or any text/markdown around the JSON.
"""

USER_PROMPT_TEMPLATE = """Use these exact definitions (verbatim) to guide detection:

{definitions}

Now, for the attached screenshot with Size: [{W},{H}],
return STRICT JSON ONLY using the schema from the system prompt.
Remember: use the provided size [{W},{H}] and the exact 5 codes in that order."""
# ===== Keep this exactly =====

# Your API key must be in variable `api_key`
local_api_key = api_key  

OUT_COLS = [
    "screen", "size_w", "size_h", "code", "present",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "raw_json"
]

# --- Minimal helpers (no regex) ------------------------------------------------

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def extract_json_block(text: str) -> str:
    """Strip ``` fences if present, then return the JSON substring from first '{' to last '}'."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if lines:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "
".join(lines).strip()
    # Find JSON object
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end+1]
    return t


def parse_model_json(text: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    obj = json.loads(text)
    if "size" not in obj or "patterns" not in obj:
        raise ValueError("Missing 'size' or 'patterns'.")
    W, H = int(obj["size"][0]), int(obj["size"][1])
    pats = obj["patterns"]
    if not isinstance(pats, list) or len(pats) != 5:
        raise ValueError("'patterns' must be length 5.")
    expected = ["FA-G-PRO","II-AM-FH","II-AM-G-SMALL","II-PRE","NG-AD"]
    for i, p in enumerate(pats):
        if p.get("code") != expected[i]:
            raise ValueError("Codes not in required order or incorrect.")
        if "present" not in p or "bboxes" not in p:
            raise ValueError("Each pattern needs 'present' and 'bboxes'.")
    return W, H, pats


def rows_from_patterns(screen: str, W: int, H: int, patterns: List[Dict[str, Any]], raw_json: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for p in patterns:
        code = p["code"]
        present = int(p.get("present", 0))
        bboxes = p.get("bboxes", []) or []
        if present == 1 and bboxes:
            for b in bboxes:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    rows.append([screen, W, H, code, 1, x1, y1, x2, y2, raw_json])
        else:
            rows.append([screen, W, H, code, 0, None, None, None, None, raw_json])
    return rows


def call_model(b64_img: str, W: int, H: int) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {local_api_key}"
    }
    user_prompt = USER_PROMPT_TEMPLATE.format(definitions=DEFINITIONS_VERBATIM, W=W, H=H)
    payload = {
        "model": MODEL_NAME,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "high"}}
            ]}
        ]
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    return json.loads(extract_json_block(text))


def find_images(root: str) -> List[str]:
    allowed = {".jpg", ".jpeg", ".png"}
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in allowed:
                paths.append(os.path.join(dirpath, fn))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT on a directory of screenshots and save long CSV of detections")
    parser.add_argument("--images", required=True, help="Directory containing screenshots (recursively scanned)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Prepare output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Scan images
    imgs = find_images(args.images)
    print(f"Found {len(imgs)} images under {args.images}")

    # Write header
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUT_COLS)

        for idx, path in enumerate(imgs, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                W, H = image_size(path)
                b64 = encode_image_b64(path)
                result = call_model(b64, W, H)
                raw_json = json.dumps(result, ensure_ascii=False)
                size_w, size_h, patterns = parse_model_json(raw_json)
                # If model size mismatches, trust local
                if (size_w, size_h) != (W, H):
                    size_w, size_h = W, H
                rows = rows_from_patterns(name, size_w, size_h, patterns, raw_json)
            except Exception as e:
                # On failure, write absent rows for all codes
                print(f"[WARN] {name}: {e}")
                rows = []
                for code in ["FA-G-PRO","II-AM-FH","II-AM-G-SMALL","II-PRE","NG-AD"]:
                    rows.append([name, W if 'W' in locals() else None, H if 'H' in locals() else None, code, 0, None, None, None, None, "{}"])
            for r in rows:
                writer.writerow(r)

            if idx % 20 == 0:
                print(f"Processed {idx}/{len(imgs)}")

    print(f"Done. Saved CSV to {args.out}")


if __name__ == "__main__":
    local_api_key = api_key  
    main()
