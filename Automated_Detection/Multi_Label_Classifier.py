"""
XGBoost Multi-Label Classifier for Dark Pattern Detection

Trains XGBoost models with:
- Multi-label binary classification (5 dark pattern types)
- Per-fold vocabulary construction (prevents data leakage)
- Early stopping with validation set
- Threshold optimization per pattern
- 5-fold stratified cross-validation
- Cross-domain evaluation
- SHAP interpretability analysis

Input Requirements:
    - CSV with image features (39 columns: img_1 to img_39)
    - Preprocessed tokens pickle (metadata_with_tokens.pkl)
    - Ground truth labels in 'true_label' column

Output Files:
    - CV_results_with_variance.csv
    - Cross_domain_results_with_variance.csv
    - Per_Pattern_Performance_with_variance.csv
    - thresholds_per_fold.csv
    - SHAP_top20_*.csv (per pattern)
    - SHAP_families_*.csv (feature family aggregates)
    - SHAP_*_beeswarm.png (visualization plots)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack as sparse_hstack, vstack as sparse_vstack, csr_matrix

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction import DictVectorizer

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
np.random.seed(42)


# ==================== CONFIGURATION ====================

# Paths
BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
METADATA_PKL = BASE_DIR / "metadata_with_tokens.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

# Dark pattern taxonomy
PATTERN_LABELS = ["II-AM-FH", "II-AM-G-SMALL", "FA-G-PRO", "II-PRE", "NG-AD"]
PATTERN_NAMES = {
    "II-AM-FH": "False Hierarchy",
    "II-AM-G-SMALL": "Small Close Button",
    "FA-G-PRO": "Pay to Avoid Ads",
    "II-PRE": "Preselection",
    "NG-AD": "Pop-up Ad"
}

# Cross-validation settings
FOLDS = 5
INNER_FOLDS = 4
SEED = 42

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "learning_rate": 0.07,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_estimators": 800,
    "early_stopping_rounds": 20,
    "n_jobs": -1,
    "eval_metric": "aucpr"
}

# SHAP configuration
SHAP_SAMPLES = 400
SHAP_MAX_DISPLAY = 20

# Global storage
THRESHOLD_LOG = []


# ==================== UTILITY FUNCTIONS ====================

def get_library_versions():
    """Get installed library versions for reproducibility."""
    def safe_version(pkg_name):
        try:
            return getattr(__import__(pkg_name), "__version__")
        except Exception:
            return "unknown"
    
    return {
        "python": sys.version,
        "numpy": safe_version("numpy"),
        "pandas": safe_version("pandas"),
        "scipy": safe_version("scipy"),
        "sklearn": safe_version("sklearn"),
        "xgboost": safe_version("xgboost"),
        "shap": safe_version("shap"),
        "iterstrat": safe_version("iterstrat"),
        "seed": SEED,
        "n_jobs": XGB_PARAMS["n_jobs"]
    }


def write_run_info(output_path: Path):
    """Write library versions and configuration to file."""
    versions = get_library_versions()
    with open(output_path, "w") as f:
        for key, value in versions.items():
            f.write(f"{key} {value}\n")
    print(f"Saved run information: {output_path}")


# ==================== DATA LOADING ====================

def load_preprocessed_data(pkl_path: Path) -> pd.DataFrame:
    """
    Load preprocessed metadata with VH tokens.
    
    Expected columns:
        - screen: Screen identifier
        - img_1 to img_39: Image features from GPT-4 Vision
        - vh_tokens: Set of VH tokens per screen
        - small_close: Heuristic flag (0/1)
        - popup_ad: Heuristic flag (0/1)
        - dataset: 'MV' or 'RICO'
        - true_label: Ground truth labels (comma-separated)
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {pkl_path}\n"
            "Please run VH feature extraction first."
        )
    
    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} samples from {pkl_path}")
    
    # Validate required columns
    required = ["screen", "vh_tokens", "small_close", "popup_ad", "dataset", "true_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate image features
    img_cols = sorted([c for c in df.columns if c.startswith("img_")])
    if len(img_cols) != 39:
        raise ValueError(f"Expected 39 image features, found {len(img_cols)}")
    
    return df


def prepare_label_matrices(df: pd.DataFrame, label_col: str = "true_label") -> dict:
    """
    Convert string labels to binary matrices per dataset.
    
    Returns:
        Dictionary with keys 'MV' and 'RICO', each containing:
            - df: DataFrame subset
            - Y: Binary label matrix (n_samples, n_labels)
    """
    mlb = MultiLabelBinarizer(classes=PATTERN_LABELS)
    mlb.fit([[]])
    
    def labels_to_matrix(series):
        return mlb.transform(
            series.fillna("").apply(
                lambda s: [t.strip() for t in str(s).split(",") 
                          if t.strip() and t.strip() != "No_DDP"]
            )
        ).astype(np.int8)
    
    mv_df = df[df['dataset'] == 'MV'].reset_index(drop=True)
    rico_df = df[df['dataset'] == 'RICO'].reset_index(drop=True)
    
    result = {
        'MV': {'df': mv_df, 'Y': labels_to_matrix(mv_df[label_col])},
        'RICO': {'df': rico_df, 'Y': labels_to_matrix(rico_df[label_col])}
    }
    
    print(f"Dataset splits: MV={len(mv_df)} samples, RICO={len(rico_df)} samples")
    print(f"Label matrices: MV {result['MV']['Y'].shape}, RICO {result['RICO']['Y'].shape}")
    
    return result, mlb


# ==================== FEATURE MATRIX CONSTRUCTION ====================

def vectorize_vh_tokens(train_df, val_df, test_df):
    """
    Fit DictVectorizer on training data only to prevent leakage.
    
    Returns:
        Tuple of (X_train, X_val, X_test, feature_names)
    """
    train_bags = [{t: 1 for t in row['vh_tokens']} for _, row in train_df.iterrows()]
    val_bags = [{t: 1 for t in row['vh_tokens']} for _, row in val_df.iterrows()]
    test_bags = [{t: 1 for t in row['vh_tokens']} for _, row in test_df.iterrows()]
    
    dv = DictVectorizer(sparse=True, sort=True)
    X_train = dv.fit_transform(train_bags)
    X_val = dv.transform(val_bags)
    X_test = dv.transform(test_bags)
    
    feature_names = [f"vh::{name}" for name in dv.get_feature_names_out()]
    
    return X_train, X_val, X_test, feature_names


def to_sparse(arr):
    """Convert dense array to sparse CSR matrix."""
    if isinstance(arr, csr_matrix):
        return arr
    return csr_matrix(arr, dtype=np.float32)


def build_feature_matrices(feature_set, train_df, val_df, test_df, img_cols):
    """
    Build feature matrices for specified feature combination.
    
    Args:
        feature_set: One of "IMG-only", "VH+heur", "Combined"
        train_df, val_df, test_df: Data splits
        img_cols: List of image feature column names
        
    Returns:
        Tuple of (X_train, X_val, X_test, feature_names)
    """
    # Extract image features
    X_train_img = to_sparse(
        train_df[img_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    )
    X_val_img = to_sparse(
        val_df[img_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    )
    X_test_img = to_sparse(
        test_df[img_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    )
    
    if feature_set == "IMG-only":
        return X_train_img, X_val_img, X_test_img, list(img_cols)
    
    # Extract VH features
    X_train_vh, X_val_vh, X_test_vh, vh_names = vectorize_vh_tokens(
        train_df, val_df, test_df
    )
    
    # Extract heuristic features
    X_train_heur = to_sparse(train_df[['small_close', 'popup_ad']].astype(np.float32).values)
    X_val_heur = to_sparse(val_df[['small_close', 'popup_ad']].astype(np.float32).values)
    X_test_heur = to_sparse(test_df[['small_close', 'popup_ad']].astype(np.float32).values)
    
    if feature_set == "VH+heur":
        return (
            sparse_hstack([X_train_vh, X_train_heur]),
            sparse_hstack([X_val_vh, X_val_heur]),
            sparse_hstack([X_test_vh, X_test_heur]),
            vh_names + ['small_close', 'popup_ad']
        )
    
    if feature_set == "Combined":
        return (
            sparse_hstack([X_train_img, X_train_vh, X_train_heur]),
            sparse_hstack([X_val_img, X_val_vh, X_val_heur]),
            sparse_hstack([X_test_img, X_test_vh, X_test_heur]),
            list(img_cols) + vh_names + ['small_close', 'popup_ad']
        )
    
    raise ValueError(f"Unknown feature_set: {feature_set}")


# ==================== MODEL TRAINING ====================

def find_optimal_threshold(y_true, y_prob):
    """
    Find decision threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Optimal threshold value
    """
    y_true = y_true.astype(np.int8)
    pos_count = int(y_true.sum())
    neg_count = len(y_true) - pos_count
    
    if pos_count == 0 or neg_count == 0:
        return 0.5
    
    thresholds = np.linspace(0.0, 1.0, 501)
    best_threshold, best_f1 = 0.5, -1.0
    
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(np.int8), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    
    return best_threshold


def train_multilabel_classifier(X_train, Y_train, X_val, Y_val, X_full, Y_full, X_test,
                                seed_base, fold_id, dataset_tag, mlb):
    """
    Train multi-label classifier with early stopping and threshold tuning.
    
    Process:
        1. Train initial model with early stopping on validation set
        2. Find optimal threshold per label using validation predictions
        3. Retrain on full training data (train+val) using optimal iterations
        4. Apply thresholded predictions on test set
    
    Args:
        X_train, Y_train: Training features and labels
        X_val, Y_val: Validation features and labels
        X_full, Y_full: Combined train+val for final retraining
        X_test: Test features
        seed_base: Base random seed
        fold_id: Fold identifier
        dataset_tag: Dataset identifier for logging
        mlb: MultiLabelBinarizer instance
        
    Returns:
        Binary predictions (n_test_samples, n_labels)
    """
    n_labels = Y_train.shape[1]
    Y_pred = np.zeros((X_test.shape[0], n_labels), dtype=np.int8)
    
    if X_train.shape[1] == 0:
        return Y_pred
    
    # Calculate class weights
    pos_train = Y_train.sum(axis=0)
    neg_train = len(Y_train) - pos_train
    pos_full = Y_full.sum(axis=0)
    neg_full = len(Y_full) - pos_full
    
    for j in range(n_labels):
        # Skip if single-class problem
        if (pos_train[j] == 0 or pos_train[j] == len(Y_train) or
            pos_full[j] == 0 or pos_full[j] == len(Y_full)):
            continue
        
        # Phase 1: Train with early stopping
        clf = XGBClassifier(
            **{**XGB_PARAMS,
               "scale_pos_weight": float(neg_train[j] / max(pos_train[j], 1)),
               "random_state": seed_base + j}
        )
        clf.fit(X_train, Y_train[:, j], 
                eval_set=[(X_val, Y_val[:, j])], 
                verbose=False)
        best_iteration = getattr(clf, "best_iteration", XGB_PARAMS["n_estimators"] - 1)
        
        # Phase 2: Find optimal threshold
        val_probs = clf.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(Y_val[:, j], val_probs)
        
        # Log threshold for reproducibility
        THRESHOLD_LOG.append({
            'dataset': dataset_tag,
            'fold': fold_id,
            'label': mlb.classes_[j],
            'label_idx': j,
            'threshold': threshold,
            'best_iteration': best_iteration,
            'seed': seed_base + j,
            'refit_seed': seed_base + j + 777
        })
        
        # Phase 3: Retrain on full data
        refit_params = {
            **XGB_PARAMS,
            "n_estimators": int(best_iteration) + 1,
            "scale_pos_weight": float(neg_full[j] / max(pos_full[j], 1)),
            "random_state": seed_base + j + 777
        }
        refit_params.pop("early_stopping_rounds", None)
        
        clf_final = XGBClassifier(**refit_params)
        clf_final.fit(X_full, Y_full[:, j], verbose=False)
        
        # Phase 4: Predict with optimal threshold
        test_probs = clf_final.predict_proba(X_test)[:, 1]
        Y_pred[:, j] = (test_probs >= threshold).astype(np.int8)
    
    return Y_pred


# ==================== EVALUATION METRICS ====================

def compute_metrics(y_true, y_pred, mlb):
    """
    Compute micro and macro averaged metrics.
    
    Returns:
        Dictionary with precision, recall, F1 for micro and macro averaging
    """
    report = classification_report(
        y_true, y_pred,
        target_names=mlb.classes_,
        digits=3,
        zero_division=0,
        output_dict=True
    )
    
    micro = report.get("micro avg", report.get("micro", {}))
    macro = report.get("macro avg", report.get("macro", {}))
    
    return {
        "micro_P": micro.get("precision", 0.0),
        "micro_R": micro.get("recall", 0.0),
        "micro_F1": micro.get("f1-score", 0.0),
        "macro_P": macro.get("precision", 0.0),
        "macro_R": macro.get("recall", 0.0),
        "macro_F1": macro.get("f1-score", 0.0),
    }


def compute_per_label_metrics(y_true, y_pred, mlb):
    """Compute metrics for each label independently."""
    n_labels = y_true.shape[1]
    results = []
    
    for j in range(n_labels):
        prec = precision_score(y_true[:, j], y_pred[:, j], zero_division=0)
        rec = recall_score(y_true[:, j], y_pred[:, j], zero_division=0)
        f1 = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        
        results.append({
            'label_idx': j,
            'label': mlb.classes_[j],
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1)
        })
    
    return results


def aggregate_fold_metrics(fold_results):
    """Compute mean and std across folds."""
    aggregated = {}
    
    for key in fold_results[0].keys():
        values = [fold[key] for fold in fold_results]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    
    return aggregated


def aggregate_per_label_metrics(fold_results):
    """Compute mean and std per label across folds."""
    n_labels = len(fold_results[0])
    aggregated = []
    
    for j in range(n_labels):
        label_metrics = [fold[j] for fold in fold_results]
        aggregated.append({
            'label_idx': j,
            'label': label_metrics[0]['label'],
            'precision_mean': float(np.mean([m['precision'] for m in label_metrics])),
            'precision_std': float(np.std([m['precision'] for m in label_metrics])),
            'recall_mean': float(np.mean([m['recall'] for m in label_metrics])),
            'recall_std': float(np.std([m['recall'] for m in label_metrics])),
            'f1_mean': float(np.mean([m['f1'] for m in label_metrics])),
            'f1_std': float(np.std([m['f1'] for m in label_metrics]))
        })
    
    return aggregated


def format_metrics_table(metrics, row_info):
    """Format metrics with mean ± std for display."""
    row = row_info.copy()
    
    for metric in ['micro_P', 'micro_R', 'micro_F1', 'macro_P', 'macro_R', 'macro_F1']:
        row[f"{metric}_mean"] = metrics[f"{metric}_mean"]
        row[f"{metric}_std"] = metrics[f"{metric}_std"]
        row[metric] = f"{metrics[f'{metric}_mean']:.3f} ± {metrics[f'{metric}_std']:.3f}"
    
    return row


# ==================== CROSS-VALIDATION ====================

def run_cross_validation(df, Y, dataset_name, img_cols, mlb):
    """
    Run 5-fold stratified cross-validation for all feature combinations.
    
    Args:
        df: DataFrame with features
        Y: Binary label matrix
        dataset_name: Dataset identifier
        img_cols: List of image feature column names
        mlb: MultiLabelBinarizer instance
        
    Returns:
        Tuple of (results_df, per_label_metrics for Combined setting)
    """
    def evaluate_feature_set(feature_set):
        mskf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        fold_metrics = []
        fold_per_label = []
        X_dummy = np.zeros((len(df), 1), dtype=np.int8)
        
        for fold_id, (train_all_idx, test_idx) in enumerate(mskf.split(X_dummy, Y), 1):
            train_all_df = df.iloc[train_all_idx]
            test_df = df.iloc[test_idx]
            Y_train_all = Y[train_all_idx]
            Y_test = Y[test_idx]
            
            # Inner split for validation
            inner_cv = MultilabelStratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 17 + fold_id
            )
            train_idx, val_idx = next(inner_cv.split(
                np.zeros((len(Y_train_all), 1)), Y_train_all
            ))
            
            train_df = train_all_df.iloc[train_idx]
            val_df = train_all_df.iloc[val_idx]
            Y_train = Y_train_all[train_idx]
            Y_val = Y_train_all[val_idx]
            
            # Build feature matrices
            X_train, X_val, X_test, _ = build_feature_matrices(
                feature_set, train_df, val_df, test_df, img_cols
            )
            X_full = sparse_vstack([X_train, X_val])
            Y_full = np.vstack([Y_train, Y_val])
            
            # Train and predict
            Y_pred = train_multilabel_classifier(
                X_train, Y_train, X_val, Y_val, X_full, Y_full, X_test,
                seed_base=2025 + 100 * fold_id,
                fold_id=fold_id,
                dataset_tag=f"{dataset_name}_{feature_set}",
                mlb=mlb
            )
            
            fold_metrics.append(compute_metrics(Y_test, Y_pred, mlb))
            fold_per_label.append(compute_per_label_metrics(Y_test, Y_pred, mlb))
        
        return aggregate_fold_metrics(fold_metrics), aggregate_per_label_metrics(fold_per_label)
    
    # Evaluate all feature combinations
    results = []
    per_label_combined = None
    
    for feature_set in ["IMG-only", "VH+heur", "Combined"]:
        metrics, per_label = evaluate_feature_set(feature_set)
        row = format_metrics_table(metrics, {
            "Dataset": dataset_name,
            "Setting": feature_set
        })
        results.append(row)
        
        if feature_set == "Combined":
            per_label_combined = per_label
    
    return pd.DataFrame(results), per_label_combined


# ==================== CROSS-DOMAIN EVALUATION ====================

def run_cross_domain_evaluation(train_df, Y_train, test_df, Y_test, 
                                evaluation_name, img_cols, mlb):
    """
    Evaluate cross-domain generalization.
    
    Trains on one dataset and tests on another using multiple folds
    to get variance estimates.
    """
    def evaluate_feature_set(feature_set):
        mskf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + 123)
        fold_metrics = []
        X_dummy = np.zeros((len(train_df), 1), dtype=np.int8)
        
        for fold_id, (train_subset_idx, _) in enumerate(mskf.split(X_dummy, Y_train), 1):
            train_subset_df = train_df.iloc[train_subset_idx]
            Y_train_subset = Y_train[train_subset_idx]
            
            # Inner validation split
            inner_cv = MultilabelStratifiedKFold(
                n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 33 + fold_id
            )
            train_idx, val_idx = next(inner_cv.split(
                np.zeros((len(Y_train_subset), 1)), Y_train_subset
            ))
            
            train_df_final = train_subset_df.iloc[train_idx]
            val_df = train_subset_df.iloc[val_idx]
            Y_train_final = Y_train_subset[train_idx]
            Y_val = Y_train_subset[val_idx]
            
            # Build features (test set uses same vocabulary as train)
            X_train, X_val, X_test, _ = build_feature_matrices(
                feature_set, train_df_final, val_df, test_df, img_cols
            )
            X_full = sparse_vstack([X_train, X_val])
            Y_full = np.vstack([Y_train_final, Y_val])
            
            # Train and evaluate
            Y_pred = train_multilabel_classifier(
                X_train, Y_train_final, X_val, Y_val, X_full, Y_full, X_test,
                seed_base=4242 + 100 * fold_id,
                fold_id=fold_id,
                dataset_tag=f"{evaluation_name}_{feature_set}",
                mlb=mlb
            )
            
            fold_metrics.append(compute_metrics(Y_test, Y_pred, mlb))
        
        return aggregate_fold_metrics(fold_metrics)
    
    results = []
    for feature_set in ["IMG-only", "VH+heur", "Combined"]:
        metrics = evaluate_feature_set(feature_set)
        row = format_metrics_table(metrics, {
            "Evaluation": evaluation_name,
            "Setting": feature_set
        })
        results.append(row)
    
    return pd.DataFrame(results)


# ==================== SHAP ANALYSIS ====================

def train_models_for_shap(df, Y, dataset_name, img_cols, mlb):
    """
    Train final models for SHAP analysis.
    
    Uses single train/val split and refits on all data with optimal iterations.
    """
    inner_cv = MultilabelStratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 777)
    train_idx, val_idx = next(inner_cv.split(np.zeros((len(Y), 1)), Y))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    
    # Build full feature matrix
    X_train, X_val, X_all, feat_names = build_feature_matrices(
        "Combined", train_df, val_df, df, img_cols
    )
    
    models = [None] * Y.shape[1]
    pos_train = Y_train.sum(axis=0)
    neg_train = len(Y_train) - pos_train
    pos_all = Y.sum(axis=0)
    neg_all = len(Y) - pos_all
    
    for j in range(Y.shape[1]):
        if (pos_train[j] == 0 or pos_train[j] == len(Y_train) or
            pos_all[j] == 0 or pos_all[j] == len(Y)):
            print(f"[{dataset_name}] Skipping {mlb.classes_[j]}: single-class problem")
            continue
        
        # Train with early stopping
        clf = XGBClassifier(
            **{**XGB_PARAMS,
               "scale_pos_weight": float(neg_train[j] / max(pos_train[j], 1)),
               "random_state": 7000 + j}
        )
        clf.fit(X_train, Y_train[:, j],
                eval_set=[(X_val, Y_val[:, j])],
                verbose=False)
        best_iteration = getattr(clf, "best_iteration", XGB_PARAMS["n_estimators"] - 1)
        
        # Refit on all data
        refit_params = {
            **XGB_PARAMS,
            "n_estimators": int(best_iteration) + 1,
            "scale_pos_weight": float(neg_all[j] / max(pos_all[j], 1)),
            "random_state": 7100 + j
        }
        refit_params.pop("early_stopping_rounds", None)
        
        clf_final = XGBClassifier(**refit_params)
        clf_final.fit(X_all, Y[:, j], verbose=False)
        models[j] = clf_final
    
    return models, feat_names, X_all


def classify_feature(name):
    """Classify feature into family for aggregation."""
    if name.startswith("img_"):
        return "IMG"
    if name in {"small_close", "popup_ad"}:
        return "HEUR"
    if name.startswith("vh::"):
        return "VH"
    return "OTHER"


def export_shap_features(shap_values, feature_names, label, dataset, output_dir, k=20):
    """Export top-k SHAP features to CSV."""
    abs_mean = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(abs_mean)[-k:][::-1]
    
    results = pd.DataFrame({
        'rank': range(1, k + 1),
        'feature': [feature_names[i] for i in top_indices],
        'mean_abs_shap': abs_mean[top_indices],
        'family': [classify_feature(feature_names[i]) for i in top_indices]
    })
    
    filename = f"SHAP_top{k}_{dataset}_{label}.csv".replace(" ", "_")
    output_path = output_dir / filename
    results.to_csv(output_path, index=False)
    print(f"  Exported: {filename}")
    
    return results


def aggregate_shap_by_family(shap_values, feature_names, output_dir, label, dataset):
    """Aggregate SHAP values by feature family."""
    abs_mean = np.abs(shap_values).mean(axis=0)
    families = [classify_feature(f) for f in feature_names]
    
    df = pd.DataFrame({
        "feature": feature_names,
        "abs_shap": abs_mean,
        "family": families
    })
    
    aggregated = df.groupby("family", as_index=False)["abs_shap"].sum().sort_values(
        "abs_shap", ascending=False
    )
    
    filename = f"SHAP_families_{dataset}_{label}.csv".replace(" ", "_")
    output_path = output_dir / filename
    aggregated.to_csv(output_path, index=False)
    
    return aggregated


def generate_shap_plots(models, feature_names, X_all, dataset_name, mlb, output_dir,
                       n_samples=400, max_display=20):
    """
    Generate SHAP beeswarm plots for interpretability.
    
    Args:
        models: List of trained XGBoost models
        feature_names: List of feature names
        X_all: Full feature matrix
        dataset_name: Dataset identifier
        mlb: MultiLabelBinarizer instance
        output_dir: Directory for saving outputs
        n_samples: Number of samples for SHAP computation
        max_display: Number of features to display in plot
    """
    rng = np.random.RandomState(123)
    n_total = X_all.shape[0]
    sample_indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    
    # Convert to dense for SHAP
    X_sample = X_all[sample_indices].toarray() if hasattr(X_all, 'toarray') else X_all[sample_indices]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    
    for j, label in enumerate(mlb.classes_):
        model = models[j]
        if model is None:
            continue
        
        print(f"\n[{dataset_name}] Generating SHAP analysis for: {label}")
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model.get_booster())
        shap_values = explainer.shap_values(X_sample)
        
        # Export top features
        export_shap_features(shap_values, feature_names, label, dataset_name, output_dir, k=20)
        
        # Create beeswarm plot
        shap.summary_plot(
            shap_values, X_sample_df,
            show=False,
            plot_type="dot",
            max_display=max_display
        )
        plt.title(
            f"{dataset_name} - {label}\nSHAP Feature Importance (top {max_display})",
            fontsize=14, pad=20
        )
        plt.tight_layout()
        
        # Save figure
        plot_filename = f"SHAP_{dataset_name}_{label}_beeswarm.png".replace(" ", "_")
        plot_path = output_dir / plot_filename
        plt.savefig(str(plot_path), bbox_inches="tight", dpi=200)
        print(f"  Saved: {plot_filename}")
        plt.close()
        
        # Aggregate by feature family
        family_agg = aggregate_shap_by_family(shap_values, feature_names, output_dir, label, dataset_name)
        print(f"\n  Feature family importance:")
        print(family_agg.to_string(index=False))
        print("-" * 80)


# ==================== MAIN PIPELINE ====================

def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("XGBoost Multi-Label Classifier for Dark Pattern Detection")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write run information
    write_run_info(OUTPUT_DIR / "RUN_INFO.txt")
    
    # Load data
    print("\nLoading preprocessed data...")
    df_meta = load_preprocessed_data(METADATA_PKL)
    
    # Prepare labels
    print("\nPreparing label matrices...")
    datasets, mlb = prepare_label_matrices(df_meta)
    
    # Get image column names
    img_cols = sorted([c for c in df_meta.columns if c.startswith("img_")])
    
    # Section A: Within-dataset cross-validation
    print("\n" + "=" * 80)
    print("SECTION A: 5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    print("\nEvaluating MobileViews...")
    mv_results, mv_per_label = run_cross_validation(
        datasets['MV']['df'], datasets['MV']['Y'], "MobileViews", img_cols, mlb
    )
    print("\nMobileViews Results:")
    print(mv_results.to_string(index=False))
    
    print("\nEvaluating RICO...")
    rico_results, rico_per_label = run_cross_validation(
        datasets['RICO']['df'], datasets['RICO']['Y'], "RICO", img_cols, mlb
    )
    print("\nRICO Results:")
    print(rico_results.to_string(index=False))
    
    # Save CV results
    combined_cv = pd.concat([mv_results, rico_results], ignore_index=True)
    combined_cv.to_csv(OUTPUT_DIR / "CV_results_with_variance.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'CV_results_with_variance.csv'}")
    
    # Save thresholds
    threshold_df = pd.DataFrame(THRESHOLD_LOG)
    threshold_df.to_csv(OUTPUT_DIR / "thresholds_per_fold.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'thresholds_per_fold.csv'} ({len(THRESHOLD_LOG)} entries)")
    
    # Save per-pattern metrics
    per_pattern_rows = []
    for metric in mv_per_label:
        per_pattern_rows.append({
            'Dataset': 'MobileViews',
            'Pattern': PATTERN_NAMES.get(metric['label'], metric['label']),
            'Precision': f"{metric['precision_mean']:.3f} ± {metric['precision_std']:.3f}",
            'Recall': f"{metric['recall_mean']:.3f} ± {metric['recall_std']:.3f}",
            'F1': f"{metric['f1_mean']:.3f} ± {metric['f1_std']:.3f}",
            'Precision_mean': metric['precision_mean'],
            'Precision_std': metric['precision_std'],
            'Recall_mean': metric['recall_mean'],
            'Recall_std': metric['recall_std'],
            'F1_mean': metric['f1_mean'],
            'F1_std': metric['f1_std']
        })
    
    for metric in rico_per_label:
        per_pattern_rows.append({
            'Dataset': 'RICO',
            'Pattern': PATTERN_NAMES.get(metric['label'], metric['label']),
            'Precision': f"{metric['precision_mean']:.3f} ± {metric['precision_std']:.3f}",
            'Recall': f"{metric['recall_mean']:.3f} ± {metric['recall_std']:.3f}",
            'F1': f"{metric['f1_mean']:.3f} ± {metric['f1_std']:.3f}",
            'Precision_mean': metric['precision_mean'],
            'Precision_std': metric['precision_std'],
            'Recall_mean': metric['recall_mean'],
            'Recall_std': metric['recall_std'],
            'F1_mean': metric['f1_mean'],
            'F1_std': metric['f1_std']
        })
    
    per_pattern_df = pd.DataFrame(per_pattern_rows)
    per_pattern_df.to_csv(OUTPUT_DIR / "Per_Pattern_Performance_with_variance.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'Per_Pattern_Performance_with_variance.csv'}")
    
    # Section B: Cross-domain evaluation
    print("\n" + "=" * 80)
    print("SECTION B: CROSS-DOMAIN EVALUATION")
    print("=" * 80)
    
    print("\nTrain: MobileViews | Test: RICO")
    mv_to_rico = run_cross_domain_evaluation(
        datasets['MV']['df'], datasets['MV']['Y'],
        datasets['RICO']['df'], datasets['RICO']['Y'],
        "MV_to_RICO", img_cols, mlb
    )
    print(mv_to_rico[['Evaluation', 'Setting', 'micro_P', 'micro_R', 'micro_F1']].to_string(index=False))
    
    print("\nTrain: RICO | Test: MobileViews")
    rico_to_mv = run_cross_domain_evaluation(
        datasets['RICO']['df'], datasets['RICO']['Y'],
        datasets['MV']['df'], datasets['MV']['Y'],
        "RICO_to_MV", img_cols, mlb
    )
    print(rico_to_mv[['Evaluation', 'Setting', 'micro_P', 'micro_R', 'micro_F1']].to_string(index=False))
    
    # Combined evaluations
    print("\nCombined: 80/20 Random Split")
    combined_df = pd.concat([datasets['MV']['df'], datasets['RICO']['df']], ignore_index=True)
    Y_combined = np.vstack([datasets['MV']['Y'], datasets['RICO']['Y']])
    X_train, X_test, Y_train, Y_test = train_test_split(
        combined_df, Y_combined, test_size=0.2, random_state=4242
    )
    combined_random = run_cross_domain_evaluation(
        X_train, Y_train, X_test, Y_test,
        "Combined_Random", img_cols, mlb
    )
    print(combined_random[['Evaluation', 'Setting', 'micro_P', 'micro_R', 'micro_F1']].to_string(index=False))
    
    # Save cross-domain results
    all_cross_domain = pd.concat([mv_to_rico, rico_to_mv, combined_random], ignore_index=True)
    all_cross_domain.to_csv(OUTPUT_DIR / "Cross_domain_results_with_variance.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'Cross_domain_results_with_variance.csv'}")
    
    # Section C: SHAP interpretability
    print("\n" + "=" * 80)
    print("SECTION C: SHAP INTERPRETABILITY")
    print("=" * 80)
    
    print("\nTraining models for SHAP (MobileViews)...")
    models_mv, feat_names_mv, X_all_mv = train_models_for_shap(
        datasets['MV']['df'], datasets['MV']['Y'], "MobileViews", img_cols, mlb
    )
    
    print("\nTraining models for SHAP (RICO)...")
    models_rico, feat_names_rico, X_all_rico = train_models_for_shap(
        datasets['RICO']['df'], datasets['RICO']['Y'], "RICO", img_cols, mlb
    )
    
    print("\nGenerating SHAP plots...")
    generate_shap_plots(models_mv, feat_names_mv, X_all_mv, "MobileViews", mlb, OUTPUT_DIR,
                       n_samples=SHAP_SAMPLES, max_display=SHAP_MAX_DISPLAY)
    
    generate_shap_plots(models_rico, feat_names_rico, X_all_rico, "RICO", mlb, OUTPUT_DIR,
                       n_samples=SHAP_SAMPLES, max_display=SHAP_MAX_DISPLAY)
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nGenerated outputs:")
    print(f"  1. RUN_INFO.txt - Library versions and configuration")
    print(f"  2. CV_results_with_variance.csv - Cross-validation metrics")
    print(f"  3. Cross_domain_results_with_variance.csv - Cross-domain evaluation")
    print(f"  4. Per_Pattern_Performance_with_variance.csv - Per-pattern results")
    print(f"  5. thresholds_per_fold.csv - Decision thresholds")
    print(f"  6. SHAP_top20_*.csv - Top features per pattern")
    print(f"  7. SHAP_families_*.csv - Feature family aggregates")
    print(f"  8. SHAP_*_beeswarm.png - Visualization plots")
    print("\nAll outputs saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
