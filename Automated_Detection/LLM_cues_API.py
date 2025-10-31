"""
GPT-4 Vision Feature Extraction for Dark Pattern Detection

Extracts 39 visual cues from mobile UI screenshots using GPT-4 Vision API.
Supports batch processing with checkpointing and retry logic for robustness.

Features:
- Automatic checkpointing every N images
- Retry logic with exponential backoff
- Resume capability (skips already processed images)
- Single image mode for testing
- Configurable via environment variables

Environment Variables:
    OPENAI_API_KEY: Required. Your OpenAI API key
    OPENAI_MODEL: Optional. Default is "gpt-4o"
"""

import os
import base64
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# Configuration
CSV_LABELS_PATH = Path("data/filtered_labels.csv")
IMAGE_DIR = Path("images")
OUTPUT_FILE = Path("outputs/GPT_features_39_RICO.xlsx")
SHEET_NAME = "Sheet1"

# Processing controls
SINGLE_IMAGE_PATH = None  
FIRST_N = None            
CHECKPOINT_EVERY = 20    
RETRY_ATTEMPTS = 3       
RETRY_DELAY = 30        

# API configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise SystemExit(
        "Error: OPENAI_API_KEY environment variable not set.\n"
        "Please export your API key: export OPENAI_API_KEY='your-key-here'"
    )


# Column schema: screen ID + 39 cue columns + ground truth label
COLUMNS = ["screen"] + [str(i) for i in range(1, 40)] + ["true_label"]


# System prompt defining the 39 visual cues
SYSTEM_PROMPT = """
You are an expert mobile-UI auditor.

**Task**
• Look at the image you are given.
• Decide, for each of cues c1-c39 (defined below), whether the cue is present (1) or absent (0).
• After thinking, respond with **exactly one line**: 39 comma-separated 0/1 integers, no spaces, no other text.

**Contract**
– Think step-by-step *silently*; do not reveal any reasoning or explanation.
– Output nothing except the single 39-integer line.
– If a cue is only partially visible, count as present.
– If you are not certain, default to 0.
– Keep the order c1…c39 exactly as given.

Cue definitions
c1  text such as "Remove Ads", "Ad-Free", "Upgrade to remove ads"
c2  ad-free badge/icon with an "×" overlay
c3  full-screen overlay blocks taps
c4  overlay dims or blurs background
c5  overlay shows video controls
c6  overlay shows countdown / "Skip in …"
c7  overlay has **no** close button
c8  overlay close button is tiny
c9  tiny close icon top-right
c10 tiny close icon in any other corner
c11 text "Ad", "Advertisement", or "Sponsored"
c12 store logo / "Install" / price badge / star strip
c13 banner ad ≤ 25 % of screen height
c14 banner blue-triangle ad logo
c15 banner tiny "×" inside
c16 tiny "×" left edge of banner
c17 tiny "×" right edge of banner
c18 ≥ 2 choices with different **background** colours
c19 ≥ 2 choices with different **text** colours
c20 choices differ only by border / outline
c21 preferred option filled/coloured vs grey/outline
c22 preferred ≥ 25 % larger than alternative
c23  ≥ 2 choices where the positive option helps the app (payments, data, tracking, rating, etc.)
c24 decline wording negative/shaming ("No thanks", "I like ads")
c25 preferred tagged "Recommended" / "Best Value"
c26 checkbox visible
c27 toggle switch visible
c28 any checkbox/toggle is **checked by default**
c29 checked mentions notifications
c30 checked mentions privacy/data
c31 checked mentions marketing/emails
c32 email **and** password fields present
c33 OAuth buttons (Google / Facebook / Apple)
c34 register / sign-up button
c35 generic login / sign-in button
c36 banner with **product-image strip**
c37 small "AdChoices" text label
c38 full-screen ad overlay with no border
c39 overlay covers Android navigation bar
"""

USER_PROMPT = """
STEP 1 – Inspect the screenshot
Carefully inspect the attached screenshot for all 39 cues.

STEP 2 – Decide 0/1 for each cue
Remember: 1 = present, 0 = absent, 0 if unsure.

STEP 3 – Internal check
Silently review all 39 decisions once more to catch mistakes.

STEP 4 – Output
Respond with **one line only**:
c1,c2,…,c39   (39 comma-separated integers, no spaces, no labels)
"""


def encode_image(image_path: Path) -> str:
    """
    Encode image file to base64 string for API transmission.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def check_internet_connectivity() -> bool:
    """
    Test internet connectivity with a quick HTTP request.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except Exception:
        return False


def call_gpt_vision(image_b64: str) -> str:
    """
    Call GPT-4 Vision API with encoded image.
    
    Args:
        image_b64: Base64-encoded image string
        
    Returns:
        Raw API response text
        
    Raises:
        requests.HTTPError: If API request fails
    """
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high"
                }}
            ]}
        ]
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()


def parse_cue_vector(response_text: str) -> list:
    """
    Parse 39-element cue vector from GPT response.
    
    Expected format: "0,1,0,1,..." (39 comma-separated integers)
    
    Args:
        response_text: Raw GPT response
        
    Returns:
        List of 39 strings (empty strings if parsing fails)
    """
    for line in response_text.splitlines():
        if line.count(",") >= 38:
            vector = [element.strip() for element in line.split(",")]
            return (vector + [""] * 39)[:39]
    
    # Return empty vector if parsing fails
    return [""] * 39


def load_or_create_results(output_path: Path, sheet_name: str) -> tuple[pd.DataFrame, set]:
    """
    Load existing results file or create new one.
    
    Args:
        output_path: Path to Excel output file
        sheet_name: Excel sheet name
        
    Returns:
        Tuple of (DataFrame with results, set of processed screen IDs)
    """
    if output_path.exists():
        df = pd.read_excel(output_path, sheet_name=sheet_name)
        
        # Ensure all required columns exist
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        
        df = df[COLUMNS]
        processed = set(df["screen"].astype(str))
        
        print(f"Loaded existing results: {len(df)} rows already processed")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=COLUMNS)
        processed = set()
        
        print("Created new results file")
    
    return df, processed


def build_job_queue(labels_path: Path, image_dir: Path, 
                    processed_screens: set) -> list[tuple[str, str, str]]:
    """
    Build list of images to process.
    
    Args:
        labels_path: Path to CSV with screen IDs and labels
        image_dir: Directory containing images
        processed_screens: Set of already-processed screen IDs
        
    Returns:
        List of (screen_id, image_path, label) tuples
    """
    jobs = []
    
    # Single image mode for testing
    if SINGLE_IMAGE_PATH:
        screen_id = Path(SINGLE_IMAGE_PATH).stem
        if screen_id not in processed_screens:
            jobs.append((screen_id, str(SINGLE_IMAGE_PATH), "external"))
            print(f"Single image mode: {SINGLE_IMAGE_PATH}")
        else:
            print(f"Single image already processed: {screen_id}")
        return jobs
    
    # Batch mode from CSV
    labels_df = pd.read_csv(labels_path)
    
    for _, row in labels_df.iterrows():
        screen_id = str(row["screen"]).strip()
        label = str(row.get("true_label", ""))
        image_path = image_dir / f"{screen_id}.jpg"
        
        if image_path.is_file() and screen_id not in processed_screens:
            jobs.append((screen_id, str(image_path), label))
            
            # Respect FIRST_N limit
            if FIRST_N is not None and len(jobs) >= FIRST_N:
                break
    
    return jobs


def save_checkpoint(df: pd.DataFrame, output_path: Path, sheet_name: str):
    """Save current results to Excel file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, sheet_name=sheet_name, index=False)


def process_image_batch(jobs: list, results_df: pd.DataFrame, 
                       processed_screens: set) -> pd.DataFrame:
    """
    Process batch of images with GPT-4 Vision.
    
    Args:
        jobs: List of (screen_id, image_path, label) tuples
        results_df: DataFrame to append results to
        processed_screens: Set to track processed screens
        
    Returns:
        Updated DataFrame with new results
    """
    checkpoint_counter = 0
    
    for screen_id, image_path, label in tqdm(jobs, desc=f"Processing ({MODEL})"):
        image_b64 = encode_image(Path(image_path))
        cue_vector = [""] * 39
        
        # Retry loop with backoff
        for attempt in range(RETRY_ATTEMPTS):
            if not check_internet_connectivity():
                print(f"  No internet connection, waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            
            try:
                response = call_gpt_vision(image_b64)
                cue_vector = parse_cue_vector(response)
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    print(f"  Attempt {attempt + 1} failed for {screen_id}, retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  Failed after {RETRY_ATTEMPTS} attempts: {screen_id} - {e}")
        
        # Append result
        results_df.loc[len(results_df)] = [screen_id] + cue_vector + [label]
        processed_screens.add(screen_id)
        checkpoint_counter += 1
        
        # Periodic checkpoint
        if checkpoint_counter % CHECKPOINT_EVERY == 0:
            save_checkpoint(results_df, OUTPUT_FILE, SHEET_NAME)
            print(f"  Checkpoint: {len(results_df)} total rows saved")
    
    return results_df


def main():
    """Main execution flow."""
    print("="*80)
    print("GPT-4 Vision Feature Extraction")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Input: {CSV_LABELS_PATH}")
    print(f"Images: {IMAGE_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Checkpoint frequency: every {CHECKPOINT_EVERY} images")
    print("="*80 + "\n")
    
    # Load existing results
    results_df, processed_screens = load_or_create_results(OUTPUT_FILE, SHEET_NAME)
    
    # Build job queue
    jobs = build_job_queue(CSV_LABELS_PATH, IMAGE_DIR, processed_screens)
    
    if not jobs:
        print("No new images to process. All done!")
        return
    
    print(f"Found {len(jobs)} images to process\n")
    
    # Process images
    results_df = process_image_batch(jobs, results_df, processed_screens)
    
    # Final save with proper screen ID type
    results_df["screen"] = pd.to_numeric(results_df["screen"], errors="ignore")
    save_checkpoint(results_df, OUTPUT_FILE, SHEET_NAME)
    
    print("\n" + "="*80)
    print(f"Completed: {len(results_df)} total rows saved to {OUTPUT_FILE}")
    print("="*80)


if __name__ == "__main__":
    main()
