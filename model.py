# =============================================================================
# Student Performance Risk Predictor — Training Pipeline
# =============================================================================
# Dataset: UCI Student Performance Dataset
# Source:  Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary
#          School Student Performance. EUROSIS. https://doi.org/10.1145/3519012
# Data:    395 students, 33 columns, zero missing values, Portuguese maths exam.
# Task:    Binary classification — predict whether a student will fail G3.
# =============================================================================

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_PATH = Path("data/student-mat.csv")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
TARGET_COLUMN = "G3"
PASS_THRESHOLD = 10


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def setup_directories():
    """Create output directories if they do not already exist.

    Returns
    -------
    None
    """
    MODELS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"[setup] Directories ready: '{MODELS_DIR}/', '{PLOTS_DIR}/'")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load the UCI Student Performance CSV file.

    Parameters
    ----------
    data_path : Path
        Path to the semicolon-delimited CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Download student-mat.csv from the UCI repository and place it in data/."
        )
    df = pd.read_csv(data_path, sep=";")
    print(f"[load]  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame) -> None:
    """Print data quality diagnostics without modifying the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    None
    """
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"[valid] Missing values: {total_missing} total")
    if total_missing > 0:
        print(missing[missing > 0])

    g3_zero = (df[TARGET_COLUMN] == 0).sum()
    print(
        f"[valid] Students with G3=0: {g3_zero} "
        "(likely non-completers — retained as legitimate fail cases)"
    )


# =============================================================================
# TARGET ENGINEERING
# =============================================================================

def engineer_target(df: pd.DataFrame):
    """Binarise G3 into pass/fail and separate features from target.

    # DESIGN DECISION: G1 and G2 intentionally retained as features.
    # Mid-year monitoring tool — educators have period grades at point of use.
    # This is deliberate design, not data leakage.
    # Threshold of 10 reflects official Portuguese secondary school passing grade.
    # 38 students with G3=0 treated as legitimate fail cases (likely non-completers).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset including G3 column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (G3 dropped).
    y : pd.Series
        Binary target: 1 = pass (G3 >= 10), 0 = fail (G3 < 10).
    """
    df = df.copy()
    df["pass_fail"] = (df[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)
    y = df["pass_fail"]
    X = df.drop(columns=[TARGET_COLUMN, "pass_fail"])

    n_pass = (y == 1).sum()
    n_fail = (y == 0).sum()
    print(
        f"[target] Class distribution — Pass: {n_pass} ({n_pass/len(y)*100:.1f}%)  "
        f"Fail: {n_fail} ({n_fail/len(y)*100:.1f}%)"
    )
    return X, y


# =============================================================================
# EDA PLOTS
# =============================================================================

def plot_class_distribution(y: pd.Series, plots_dir: Path) -> None:
    """Bar chart showing absolute counts and percentages for each class.

    Parameters
    ----------
    y : pd.Series
        Binary target series (0 = fail, 1 = pass).
    plots_dir : Path
        Directory to save the plot.

    Returns
    -------
    None
    """
    counts = y.value_counts().sort_index()
    labels = ["Fail (0)", "Pass (1)"]
    colours = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colours, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts.values):
        pct = count / len(y) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Class Distribution: Pass vs Fail", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() * 1.20)
    plt.tight_layout()
    out = plots_dir / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {out}")


def plot_correlation_heatmap(df: pd.DataFrame, plots_dir: Path) -> None:
    """Pearson correlation heatmap for all numeric columns plus the pass_fail target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (before G3 is dropped) — numeric columns used.
    plots_dir : Path
        Directory to save the plot.

    Returns
    -------
    None
    """
    df_copy = df.copy()
    df_copy["pass_fail"] = (df_copy[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_copy[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.4,
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Pearson Correlation Heatmap (Numeric Features + pass_fail Target)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = plots_dir / "correlation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {out}")


def run_eda(df: pd.DataFrame, y: pd.Series, plots_dir: Path) -> None:
    """Orchestrate all EDA visualisations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    y : pd.Series
        Binary target series.
    plots_dir : Path
        Directory to save plots.

    Returns
    -------
    None
    """
    print("[eda]   Generating EDA plots...")
    plot_class_distribution(y, plots_dir)
    plot_correlation_heatmap(df, plots_dir)
    print("[eda]   EDA complete.")


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

def define_features(X: pd.DataFrame):
    """Return the fixed numerical and categorical column lists.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (G3 already removed).

    Returns
    -------
    numerical_cols : list[str]
        15 numeric feature names.
    categorical_cols : list[str]
        17 categorical feature names.
    """
    numerical_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2",
    ]
    categorical_cols = [
        "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
        "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic",
    ]
    print(f"[feat]  Numerical  ({len(numerical_cols)}): {numerical_cols}")
    print(f"[feat]  Categorical ({len(categorical_cols)}): {categorical_cols}")
    return numerical_cols, categorical_cols


# =============================================================================
# PREPROCESSOR
# =============================================================================

def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Construct a ColumnTransformer that scales numerics and encodes categoricals.

    # handle_unknown='ignore' prevents app crashes on unseen input combinations
    # (e.g. a school code present in the UI but absent from the training set).

    Parameters
    ----------
    numerical_cols : list[str]
        Columns to standardise with StandardScaler.
    categorical_cols : list[str]
        Columns to encode with OneHotEncoder.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )
    print("[prep]  Preprocessor built (not yet fitted).")
    return preprocessor


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_models() -> dict:
    """Instantiate the three candidate classifiers.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name to unfitted scikit-learn estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=100
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE, n_estimators=100
        ),
    }
    print(f"[model] {len(models)} models defined: {list(models.keys())}")
    return models


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(
    models_dict: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
) -> dict:
    """Run Stratified K-Fold cross-validation for every candidate model.

    # Pipeline ensures preprocessor refits on each fold's training data independently.
    # Stratified K-Fold used because 395 rows is too small for a single split.
    # K=5 averages across 5 non-overlapping splits for stable metric estimates.
    # Stratification preserves 67/33 class ratio in every fold.

    Parameters
    ----------
    models_dict : dict
        Candidate models from build_models().
    preprocessor : ColumnTransformer
        Unfitted preprocessor from build_preprocessor().
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    dict[str, dict]
        Nested dict: model name → metric name → array of per-fold scores.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}

    print(f"[cv]    Running {n_folds}-Fold Stratified CV on {len(X_train)} training samples...")
    for name, clf in models_dict.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = scores
        print(f"[cv]    {name} — F1: {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")

    return results


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_cv_results(cv_results: dict) -> None:
    """Print a formatted summary table of cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Output of run_cross_validation().

    Returns
    -------
    None
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'Model':<25}" + "".join(f"{'  '+m.upper():<18}" for m in metrics)
    print("\n" + "=" * len(header))
    print("CROSS-VALIDATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, scores in cv_results.items():
        row = f"{name:<25}"
        for m in metrics:
            key = f"test_{m}"
            mean = scores[key].mean()
            std = scores[key].std()
            row += f"  {mean:.3f}±{std:.3f}   "
        print(row)
    print("=" * len(header) + "\n")


# =============================================================================
# MODEL SELECTION
# =============================================================================

def select_best_model(
    cv_results: dict, models_dict: dict, preprocessor: ColumnTransformer
):
    """Choose the best model: primary F1, tiebreaker ROC-AUC, then recall, then lowest std.

    Parameters
    ----------
    cv_results : dict
        Output of run_cross_validation().
    models_dict : dict
        Original model dict from build_models().
    preprocessor : ColumnTransformer
        Unfitted preprocessor — will be embedded in returned pipeline.

    Returns
    -------
    best_name : str
        Name of the selected model.
    best_pipeline : Pipeline
        Fresh, unfitted pipeline ready for training.
    """
    ranked = sorted(
        cv_results.keys(),
        key=lambda n: (
            -cv_results[n]["test_f1"].mean(),
            -cv_results[n]["test_roc_auc"].mean(),
            -cv_results[n]["test_recall"].mean(),
            cv_results[n]["test_f1"].std(),
        ),
    )
    best_name = ranked[0]
    best_clf = models_dict[best_name]
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", best_clf)])

    f1_mean = cv_results[best_name]["test_f1"].mean()
    roc_mean = cv_results[best_name]["test_roc_auc"].mean()
    print(f"[select] Best model: '{best_name}'")
    print(
        f"[select] Rationale: highest mean CV F1 ({f1_mean:.3f}), "
        f"ROC-AUC ({roc_mean:.3f}). "
        "F1 chosen as primary criterion due to class imbalance (67/33)."
    )
    return best_name, best_pipeline


# =============================================================================
# FINAL TRAINING
# =============================================================================

def train_final_model(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the best pipeline on the full training set.

    Parameters
    ----------
    best_pipeline : Pipeline
        Unfitted pipeline from select_best_model().
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    best_pipeline.fit(X_train, y_train)
    print(f"[train] Final model fitted on {len(X_train)} training samples.")
    return best_pipeline


# =============================================================================
# TEST-SET EVALUATION
# =============================================================================

def evaluate_final_model(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the fitted pipeline on the held-out test set at a given threshold.

    Fail-class probability is obtained via predict_proba, and the fail class
    index is verified from classifier.classes_ rather than assumed by position.
    ROC-AUC is computed from pass-class probability so that higher scores
    correctly correspond to the greater label (class 1), matching sklearn's
    roc_auc_score convention.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted pipeline from train_final_model().
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    threshold : float, optional
        Fail-probability threshold for class assignment. Default 0.5 matches
        the behaviour of pipeline.predict().

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, F1 (all pos_label=1), and ROC-AUC.
    """
    y_prob_all = fitted_pipeline.predict_proba(X_test)
    classes = fitted_pipeline.named_steps["classifier"].classes_
    fail_class_index = int(list(classes).index(0))
    pass_class_index = int(list(classes).index(1))

    fail_probs = y_prob_all[:, fail_class_index]
    pass_probs = y_prob_all[:, pass_class_index]

    # Threshold applied to fail probability — consistent with find_optimal_threshold
    y_pred = np.where(fail_probs >= threshold, 0, 1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, pass_probs),
    }
    print(f"\n[eval]  Test-Set Metrics (threshold={threshold:.2f})")
    for k, v in metrics.items():
        print(f"        {k:<12}: {v:.4f}")
    print(f"\n[eval]  Classification Report (threshold={threshold:.2f}):")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
    return metrics


# =============================================================================
# THRESHOLD OPTIMISATION
# =============================================================================


def find_optimal_threshold(
    best_model_name: str,
    models_dict: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
) -> float:
    """Find the optimal fail-probability threshold using out-of-fold CV predictions.

    Two independent pipeline clones are used: one to verify the fail class
    index from classifier.classes_, one for cross_val_predict. The held-out
    test set is never touched during this process.

    Parameters
    ----------
    best_model_name : str
        Key into models_dict identifying the selected classifier.
    models_dict : dict
        Candidate models from build_models().
    preprocessor : ColumnTransformer
        Unfitted preprocessor from build_preprocessor().
    X_train : pd.DataFrame
        Training features — the only data used here.
    y_train : pd.Series
        Training target.
    n_folds : int
        CV folds — matches the setting used in run_cross_validation().

    Returns
    -------
    float
        Optimal threshold for the fail-class probability.
    """
    # METHODOLOGICAL NOTE: Threshold tuned using aggregated out-of-fold
    # predictions from cross_val_predict on X_train only. A fresh clone
    # of the pipeline is used to avoid state contamination from the
    # already-fitted model. The held-out test set is never used for
    # threshold selection — it remains a clean unbiased estimate of
    # real-world performance at the chosen threshold.
    # Fail class index verified via named_steps['classifier'].classes_
    # rather than assumed by column position.
    # Recall >= 0.80 constraint reflects the asymmetric cost structure
    # of educational welfare tools: missing a failing student is more
    # harmful than an unnecessary intervention.
    # Tiebreak: highest F1, then highest precision, then lowest threshold.

    print("[thresh] Finding optimal classification threshold on training data...")

    base_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", models_dict[best_model_name])]
    )

    # Clone #1 — fit once solely to verify which column index is the fail class.
    # sklearn.base.clone() produces an unfitted copy with identical hyperparameters.
    clone1 = clone(base_pipeline)
    clone1.fit(X_train, y_train)
    classes = clone1.named_steps["classifier"].classes_
    fail_class_index = int(list(classes).index(0))
    print(
        f"[thresh] Classifier classes: {list(classes)} — "
        f"fail class index verified: {fail_class_index}"
    )

    # Clone #2 — separate fresh clone for cross_val_predict out-of-fold probabilities.
    # Using the same StratifiedKFold settings as run_cross_validation for consistency.
    clone2 = clone(base_pipeline)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        clone2, X_train, y_train, cv=cv, method="predict_proba"
    )
    fail_probs = oof_proba[:, fail_class_index]

    # Sweep thresholds 0.20–0.80 in 0.01 increments
    thresholds = np.round(np.arange(0.20, 0.81, 0.01), 2)
    candidates = []
    for t in thresholds:
        y_pred = np.where(fail_probs >= t, 0, 1)
        p = precision_score(y_train, y_pred, pos_label=0, zero_division=0)
        r = recall_score(y_train, y_pred, pos_label=0, zero_division=0)
        f = f1_score(y_train, y_pred, pos_label=0, zero_division=0)
        candidates.append((float(t), float(p), float(r), float(f)))

    # Display top 10 by F1
    display_sorted = sorted(candidates, key=lambda x: (-x[3], -x[2], x[0]))
    print("\n[thresh] Top 10 threshold candidates (sorted by F1 descending):")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    for t, p, r, f in display_sorted[:10]:
        print(f"  {t:>10.2f}  {p:>10.3f}  {r:>10.3f}  {f:>10.3f}")

    # Selection: recall(fail) >= 0.80; maximise F1, tiebreak precision, then lowest t
    eligible = [(t, p, r, f) for t, p, r, f in candidates if r >= 0.80]
    if eligible:
        best_cand = min(eligible, key=lambda x: (-x[3], -x[2], x[0]))
        print("\n[thresh] Constraint recall(fail) >= 0.80 satisfied.")
    else:
        # Fallback: highest recall, then highest F1, then lowest threshold
        best_cand = min(candidates, key=lambda x: (-x[2], -x[3], x[0]))
        print(
            "\n[thresh] Warning: no threshold reached recall(fail) >= 0.80. "
            "Fallback: highest recall."
        )

    opt_t, opt_p, opt_r, opt_f = best_cand
    print(
        f"[thresh] Selected threshold: {opt_t:.2f}  |  "
        f"Precision: {opt_p:.3f}  Recall: {opt_r:.3f}  F1: {opt_f:.3f}\n"
    )
    return opt_t


# =============================================================================
# EVALUATION PLOTS
# =============================================================================

def plot_confusion_matrix(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
) -> None:
    """Save an annotated seaborn confusion-matrix heatmap.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    plots_dir : Path
        Output directory.

    Returns
    -------
    None
    """
    from sklearn.metrics import confusion_matrix

    y_pred = fitted_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Predicted Fail", "Predicted Pass"],
        yticklabels=["Actual Fail", "Actual Pass"],
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    out = plots_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {out}")


def plot_roc_curves(
    models_dict: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
) -> None:
    """Train each model fresh and plot all ROC curves on one figure.

    Parameters
    ----------
    models_dict : dict
        Candidate models from build_models().
    preprocessor : ColumnTransformer
        Unfitted preprocessor.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    plots_dir : Path
        Output directory.

    Returns
    -------
    None
    """
    colours = ["#3498db", "#e67e22", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for (name, clf), colour in zip(models_dict.items(), colours):
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colour, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    out = plots_dir / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {out}")


def plot_feature_importances(
    fitted_pipeline: Pipeline,
    numerical_cols: list,
    categorical_cols: list,
    plots_dir: Path,
) -> dict:
    """Extract, plot, and return the top 10 feature importances.

    # Feature importance provides explainability — ethically required for AI in education.
    # These are global importances across all training students, not personalised explanations.
    # Preprocessor saved separately to allow independent inspection of transformation parameters.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted pipeline whose classifier supports feature_importances_ or coef_.
    numerical_cols : list[str]
        Numerical feature names (pre-encoding order).
    categorical_cols : list[str]
        Categorical feature names (pre-encoding order).
    plots_dir : Path
        Output directory.

    Returns
    -------
    dict[str, float]
        Top 10 features sorted descending by importance.
    """
    preprocessor_fitted = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        print("[plot]  Classifier has no importances — skipping feature importance plot.")
        return {}

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top10 = pairs[:10]
    names_top, vals_top = zip(*top10)

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.viridis(np.linspace(0.3, 0.85, len(names_top)))
    bars = ax.barh(list(reversed(names_top)), list(reversed(vals_top)), color=list(reversed(colours)))
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "feature_importances.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot]  Saved {out}")

    top_dict = {name: float(val) for name, val in top10}
    return top_dict


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_all_outputs(
    fitted_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
    feature_names,
    top_features: dict,
    metrics: dict,
    best_model_name: str,
    optimal_threshold: float,
    models_dir: Path,
) -> None:
    """Persist all model artefacts and metrics to disk.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Trained pipeline.
    preprocessor : ColumnTransformer
        Fitted preprocessor (extracted for independent inspection).
    feature_names : np.ndarray
        Full list of post-encoding feature names.
    top_features : dict
        Top 10 feature importances.
    metrics : dict
        Test-set evaluation metrics at the optimal threshold.
    best_model_name : str
        Name of the selected model.
    optimal_threshold : float
        Threshold found by find_optimal_threshold(), saved for app use.
    models_dir : Path
        Directory to write files into.

    Returns
    -------
    None
    """
    paths = {
        "best_model": models_dir / "best_model.joblib",
        "preprocessor": models_dir / "preprocessor.joblib",
        "feature_names": models_dir / "feature_names.joblib",
        "top_features": models_dir / "top_features.joblib",
        "metrics": models_dir / "metrics.json",
    }

    joblib.dump(fitted_pipeline, paths["best_model"])
    print(f"[save]  {paths['best_model']}")

    joblib.dump(preprocessor, paths["preprocessor"])
    print(f"[save]  {paths['preprocessor']}")

    joblib.dump(feature_names, paths["feature_names"])
    print(f"[save]  {paths['feature_names']}")

    joblib.dump(top_features, paths["top_features"])
    print(f"[save]  {paths['top_features']}")

    metrics_payload = {
        "best_model": best_model_name,
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "optimal_threshold": round(float(optimal_threshold), 4),
    }
    with open(paths["metrics"], "w") as fh:
        json.dump(metrics_payload, fh, indent=2)
    print(f"[save]  {paths['metrics']}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the complete training pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  Student Performance Risk Predictor — Training Pipeline")
    print("=" * 60 + "\n")

    # 1. Directories
    setup_directories()

    # 2. Load data
    df = load_data(DATA_PATH)

    # 3. Validate
    validate_data(df)

    # 4. Engineer target
    X, y = engineer_target(df)

    # 5. Train/test split — stratified to preserve 67/33 class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"[split] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples"
    )

    # 6. EDA
    run_eda(df, y, PLOTS_DIR)

    # 7. Feature lists
    numerical_cols, categorical_cols = define_features(X)

    # 8. Preprocessor
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # 9. Candidate models
    models_dict = build_models()

    # 10. Cross-validation
    cv_results = run_cross_validation(
        models_dict, preprocessor, X_train, y_train, N_FOLDS
    )
    print_cv_results(cv_results)

    # 11. Select best
    best_name, best_pipeline = select_best_model(cv_results, models_dict, preprocessor)

    # 12. Train final model
    fitted_pipeline = train_final_model(best_pipeline, X_train, y_train)

    # 13. Baseline evaluation at default threshold 0.5 — printed for comparison only
    print("\n--- Baseline evaluation (threshold = 0.50) ---")
    evaluate_final_model(fitted_pipeline, X_test, y_test, threshold=0.5)

    # 14. Find optimal threshold on training data only — test set never touched
    optimal_threshold = find_optimal_threshold(
        best_name, models_dict, preprocessor, X_train, y_train, N_FOLDS
    )

    # 15. Final evaluation at optimal threshold — these metrics are saved and reported
    print("\n--- Final evaluation (optimal threshold) ---")
    metrics = evaluate_final_model(
        fitted_pipeline, X_test, y_test, threshold=optimal_threshold
    )

    # 16. Evaluation plots
    plot_confusion_matrix(fitted_pipeline, X_test, y_test, PLOTS_DIR)
    plot_roc_curves(
        models_dict, preprocessor, X_train, y_train, X_test, y_test, PLOTS_DIR
    )
    feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    top_features = plot_feature_importances(
        fitted_pipeline, numerical_cols, categorical_cols, PLOTS_DIR
    )

    # 17. Save artefacts including optimal_threshold in metrics.json
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    save_all_outputs(
        fitted_pipeline,
        fitted_preprocessor,
        feature_names,
        top_features,
        metrics,
        best_name,
        optimal_threshold,
        MODELS_DIR,
    )

    print("\n" + "=" * 60)
    print("  Training complete. Saved files:")
    for f in sorted(MODELS_DIR.iterdir()):
        print(f"    {f}")
    for f in sorted(PLOTS_DIR.iterdir()):
        print(f"    {f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
