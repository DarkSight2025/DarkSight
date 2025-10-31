# Hybrid Human–AI Detection of Deceptive Design Patterns in Mobile Applications

Companion repository for the paper "Hybrid Human-AI Detection of Deceptive Design Patterns in Mobile Applications."

This repo contains the datasets and code used to run the analyses, reproduce figures/tables, and evaluate automated detectors.

## Contents

**Results.csv** – Full study outputs, including:
- `screenshot_id`
- expert annotations (label and bbox)
- GPT-4o annotations (label and bbox)
- crowd annotations (labels and bboxes) for screenshots shown to workers

Crowd worker identities are anonymized. Column names in the file header are the source of truth.

### /study
- **pilot_analysis.py** – Pilot study analysis as reported in the paper.
- **LLM_API_Call.py** – GPT-4o API invocation and the exact prompt used to obtain model judgments.
- **Detection_Analysis.py** – Main analysis and metrics for the 2×2 experiment design.
- **GLMM_Analysis.R** – Generalized linear mixed models testing pre-filter and hints effects (RQ2).

### /Automated_Detection
- **LLM_cues_API.py** – GPT-4o API call and prompt used to extract LLM-based cues/features.
- **VH_features.py** – Feature extraction from View Hierarchy files.
- **Multi_Label_Classifier.py** – XGBoost multi-label classifier and metric computation.

