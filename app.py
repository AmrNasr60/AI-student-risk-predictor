# =============================================================================
# Student Performance Risk Predictor — Streamlit Application
# =============================================================================
# This app ONLY loads pre-trained model artefacts. It never trains models
# and never writes files. Run model.py first to generate the artefacts.
#
# Dataset: UCI Student Performance Dataset
# Cortez, P. & Silva, A. (2008). Using Data Mining to Predict Secondary School
# Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th
# Annual Future Business Technology Conference, Porto, Portugal. EUROSIS.
# =============================================================================

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# st.set_page_config MUST be the absolute first Streamlit call in the script.
st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

# Label mapping: internal sklearn ColumnTransformer names → human-readable labels.
# Any key not present in this mapping falls back to the raw feature name.
LABEL_MAPPING = {
    "num__G2": "Period 2 Grade",
    "num__G1": "Period 1 Grade",
    "num__absences": "Absences",
    "num__age": "Age",
    "num__failures": "Past Failures",
    "num__studytime": "Study Time",
    "num__Medu": "Mother's Education",
    "num__Fedu": "Father's Education",
    "num__traveltime": "Travel Time",
    "num__famrel": "Family Relations",
    "num__freetime": "Free Time",
    "num__goout": "Going Out",
    "num__Dalc": "Weekday Alcohol",
    "num__Walc": "Weekend Alcohol",
    "num__health": "Health Status",
    "cat__school_GP": "School: GP",
    "cat__school_MS": "School: MS",
    "cat__sex_F": "Sex: Female",
    "cat__sex_M": "Sex: Male",
    "cat__address_U": "Address: Urban",
    "cat__address_R": "Address: Rural",
    "cat__famsize_GT3": "Family Size: >3",
    "cat__famsize_LE3": "Family Size: ≤3",
    "cat__Pstatus_T": "Parents: Together",
    "cat__Pstatus_A": "Parents: Apart",
    "cat__Mjob_at_home": "Mother's Job: At Home",
    "cat__Mjob_health": "Mother's Job: Health",
    "cat__Mjob_other": "Mother's Job: Other",
    "cat__Mjob_services": "Mother's Job: Services",
    "cat__Mjob_teacher": "Mother's Job: Teacher",
    "cat__Fjob_at_home": "Father's Job: At Home",
    "cat__Fjob_health": "Father's Job: Health",
    "cat__Fjob_other": "Father's Job: Other",
    "cat__Fjob_services": "Father's Job: Services",
    "cat__Fjob_teacher": "Father's Job: Teacher",
    "cat__reason_course": "Reason: Course",
    "cat__reason_home": "Reason: Home",
    "cat__reason_reputation": "Reason: Reputation",
    "cat__reason_other": "Reason: Other",
    "cat__guardian_mother": "Guardian: Mother",
    "cat__guardian_father": "Guardian: Father",
    "cat__guardian_other": "Guardian: Other",
    "cat__schoolsup_yes": "School Support: Yes",
    "cat__schoolsup_no": "School Support: No",
    "cat__famsup_yes": "Family Support: Yes",
    "cat__famsup_no": "Family Support: No",
    "cat__paid_yes": "Paid Classes: Yes",
    "cat__paid_no": "Paid Classes: No",
    "cat__activities_yes": "Activities: Yes",
    "cat__activities_no": "Activities: No",
    "cat__nursery_yes": "Nursery: Yes",
    "cat__nursery_no": "Nursery: No",
    "cat__higher_yes": "Higher Education: Yes",
    "cat__higher_no": "Higher Education: No",
    "cat__internet_yes": "Internet: Yes",
    "cat__internet_no": "Internet: No",
    "cat__romantic_yes": "Romantic: Yes",
    "cat__romantic_no": "Romantic: No",
}

# =============================================================================
# ASSET LOADING
# =============================================================================


def load_model_assets(models_dir: Path):
    """Load model artefacts from disk and extract the optimal threshold.

    preprocessor and feature_names are not loaded here — feature labels are
    resolved via LABEL_MAPPING applied directly to the top_features keys.
    optimal_threshold defaults to 0.5 if the key is absent (backwards compat
    with any metrics.json written before threshold optimisation was added).

    Displays an error and halts the app if any required file is missing.

    Parameters
    ----------
    models_dir : Path
        Directory containing the joblib/json artefacts.

    Returns
    -------
    tuple
        (pipeline, top_features, metrics, optimal_threshold)
    """
    try:
        pipeline = joblib.load(models_dir / "best_model.joblib")
        top_features = joblib.load(models_dir / "top_features.joblib")
        with open(models_dir / "metrics.json") as fh:
            metrics = json.load(fh)
        optimal_threshold = float(metrics.get("optimal_threshold", 0.5))
    except Exception:
        st.error(
            "Model files not found. Please run model.py first.\n\n"
            "```\npython model.py\n```"
        )
        st.stop()
    return pipeline, top_features, metrics, optimal_threshold


# =============================================================================
# CUSTOM CSS
# =============================================================================


def inject_custom_css() -> None:
    """Inject minor CSS refinements that .streamlit/config.toml cannot express.

    Core colours (background, sidebar, text, primary accent) are set in
    config.toml and applied natively by Streamlit — no CSS needed for those.
    This function only adds layout/weight touches using stable, standard
    HTML selectors that do not depend on Streamlit's internal data-testid
    or data-baseweb attributes (which change across versions).

    Returns
    -------
    None
    """
    st.markdown(
        """
        <style>
        /* Heading weight — config.toml sets the colour */
        h2, h3 { font-weight: 600; }

        /* Button shape — config.toml sets the colour */
        .stButton > button { border-radius: 8px; font-weight: 600; }

        /* Alert boxes — consistent corner rounding */
        .stAlert { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR INPUTS
# =============================================================================


def render_sidebar_inputs() -> dict:
    """Render all 32 student-feature inputs in the sidebar.

    Returns
    -------
    dict
        Mapping of feature column names to user-supplied values.
        Keys match training column names exactly (including capitalisation).
    """
    st.sidebar.info(
        "ℹ️ Some fields such as family background and lifestyle indicators may be "
        "sourced from student enrollment forms, pastoral records, or annual student "
        "surveys. All predictions should be reviewed by a qualified educator."
    )

    inputs = {}

    # ── Academic Information ──────────────────────────────────────────────────
    st.sidebar.header("📚 Academic Information")

    inputs["G1"] = st.sidebar.slider("Period 1 Grade (G1)", 0, 20, 10)
    inputs["G2"] = st.sidebar.slider("Period 2 Grade (G2)", 0, 20, 10)
    inputs["studytime"] = st.sidebar.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        index=1,
        format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x],
    )
    inputs["failures"] = st.sidebar.slider("Past Class Failures", 0, 3, 0)
    inputs["absences"] = st.sidebar.slider("Number of Absences", 0, 75, 0)
    inputs["schoolsup"] = st.sidebar.selectbox(
        "Extra Educational Support", ["yes", "no"], index=1
    )
    inputs["famsup"] = st.sidebar.selectbox(
        "Family Educational Support", ["yes", "no"], index=1
    )
    inputs["paid"] = st.sidebar.selectbox(
        "Extra Paid Classes (Maths)", ["yes", "no"], index=1
    )
    inputs["higher"] = st.sidebar.selectbox(
        "Wants Higher Education", ["yes", "no"], index=0
    )

    # ── Personal Background ───────────────────────────────────────────────────
    st.sidebar.header("👤 Personal Background")

    inputs["school"] = st.sidebar.selectbox("School", ["GP", "MS"], index=0)
    inputs["sex"] = st.sidebar.selectbox("Sex", ["F", "M"], index=0)
    inputs["age"] = st.sidebar.slider("Age", 15, 22, 17)
    inputs["address"] = st.sidebar.selectbox(
        "Address (U=Urban, R=Rural)", ["U", "R"], index=0
    )
    inputs["famsize"] = st.sidebar.selectbox(
        "Family Size (GT3=>3, LE3=≤3)", ["GT3", "LE3"], index=0
    )
    inputs["Pstatus"] = st.sidebar.selectbox(
        "Parent Status (T=Together, A=Apart)", ["T", "A"], index=0
    )
    inputs["Medu"] = st.sidebar.selectbox(
        "Mother's Education",
        options=[0, 1, 2, 3, 4],
        index=2,
        format_func=lambda x: {
            0: "None", 1: "Primary 4th", 2: "Primary 9th",
            3: "Secondary", 4: "Higher",
        }[x],
    )
    inputs["Fedu"] = st.sidebar.selectbox(
        "Father's Education",
        options=[0, 1, 2, 3, 4],
        index=2,
        format_func=lambda x: {
            0: "None", 1: "Primary 4th", 2: "Primary 9th",
            3: "Secondary", 4: "Higher",
        }[x],
    )
    inputs["Mjob"] = st.sidebar.selectbox(
        "Mother's Job",
        ["at_home", "health", "other", "services", "teacher"],
        index=2,
    )
    inputs["Fjob"] = st.sidebar.selectbox(
        "Father's Job",
        ["at_home", "health", "other", "services", "teacher"],
        index=2,
    )
    inputs["reason"] = st.sidebar.selectbox(
        "Reason for Choosing School",
        ["course", "home", "reputation", "other"],
        index=0,
    )
    inputs["guardian"] = st.sidebar.selectbox(
        "Guardian", ["mother", "father", "other"], index=0
    )
    inputs["traveltime"] = st.sidebar.selectbox(
        "Home-to-School Travel Time",
        options=[1, 2, 3, 4],
        index=0,
        format_func=lambda x: {
            1: "<15 min", 2: "15-30 min", 3: "30-60 min", 4: ">1 hr"
        }[x],
    )
    inputs["nursery"] = st.sidebar.selectbox(
        "Attended Nursery School", ["yes", "no"], index=0
    )

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    st.sidebar.header("🌱 Lifestyle")

    inputs["internet"] = st.sidebar.selectbox(
        "Internet Access at Home", ["yes", "no"], index=0
    )
    inputs["romantic"] = st.sidebar.selectbox(
        "In a Romantic Relationship", ["yes", "no"], index=1
    )
    inputs["freetime"] = st.sidebar.slider(
        "Free Time (1=Very Low, 5=Very High)", 1, 5, 3
    )
    inputs["goout"] = st.sidebar.slider(
        "Going Out (1=Very Low, 5=Very High)", 1, 5, 3
    )
    inputs["Dalc"] = st.sidebar.slider(
        "Weekday Alcohol (1=Very Low, 5=Very High)", 1, 5, 1
    )
    inputs["Walc"] = st.sidebar.slider(
        "Weekend Alcohol (1=Very Low, 5=Very High)", 1, 5, 1
    )
    inputs["health"] = st.sidebar.slider(
        "Health (1=Very Bad, 5=Very Good)", 1, 5, 3
    )
    inputs["activities"] = st.sidebar.selectbox(
        "Extra-Curricular Activities", ["yes", "no"], index=1
    )
    inputs["famrel"] = st.sidebar.slider(
        "Family Relations (1=Very Bad, 5=Excellent)", 1, 5, 4
    )

    # ── About the Input Fields expander ───────────────────────────────────────
    with st.sidebar.expander("ℹ️ About the Input Fields"):
        st.markdown(
            """
**School records:** G1, G2, absences, failures, school, age, sex

**Enrollment forms:** address, family size, parent status, guardian,
mother/father education and job, reason, travel time

**Student surveys:** study time, internet, activities, higher education
aspiration, health, going out frequency

**Pastoral records:** school support, family support, paid classes

---
*Sensitive fields such as romantic relationship status and alcohol
consumption are included because they were present in the original
research dataset. In a real deployment these would require careful
data governance and student consent.*
            """
        )

    return inputs


# =============================================================================
# INPUT DATAFRAME
# =============================================================================


def build_input_dataframe(input_dict: dict) -> pd.DataFrame:
    """Convert sidebar inputs into a single-row DataFrame matching training column order.

    Parameters
    ----------
    input_dict : dict
        Feature values from render_sidebar_inputs().

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with column order identical to training data.
    """
    column_order = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "nursery", "higher", "internet", "romantic",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2",
    ]
    row = {col: input_dict[col] for col in column_order}

    # Ensure integer types for numeric fields
    int_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2",
    ]
    for col in int_cols:
        row[col] = int(row[col])

    return pd.DataFrame([row], columns=column_order)


# =============================================================================
# PREDICTION
# =============================================================================


def run_prediction(pipeline, input_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Run inference and derive the predicted class from the fail probability.

    pipeline.predict() is intentionally NOT used for final class assignment.
    Instead the fail-class probability is extracted via predict_proba, the
    fail class index is verified from classifier.classes_ (not assumed by
    position), and is_at_risk is set by comparing against the optimal
    threshold supplied at call time.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted pipeline from load_model_assets().
    input_df : pd.DataFrame
        One-row DataFrame from build_input_dataframe().
    threshold : float, optional
        Optimal fail-probability threshold loaded from metrics.json.
        Defaults to 0.5 for backwards compatibility.

    Returns
    -------
    dict
        Keys: predicted_class, probability_fail, probability_pass, is_at_risk.
    """
    try:
        proba = pipeline.predict_proba(input_df)
        classes = pipeline.named_steps["classifier"].classes_
        fail_class_index = list(classes).index(0)
        probability_fail = float(proba[0][fail_class_index])
        probability_pass = 1.0 - probability_fail
        is_at_risk = probability_fail >= threshold
        predicted_class = 0 if is_at_risk else 1
        return {
            "predicted_class": predicted_class,
            "probability_fail": probability_fail,
            "probability_pass": probability_pass,
            "is_at_risk": is_at_risk,
        }
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()


# =============================================================================
# PREDICTION OUTPUT
# =============================================================================


def render_prediction_output(prediction_result: dict, borderline_profile: bool) -> None:
    """Render the risk banner, optional borderline warning, and probability metric cards.

    No probability values or classification results are altered here.
    The borderline warning is purely contextual — it does not change the
    model's output, only flags that the profile warrants closer educator review.

    Parameters
    ----------
    prediction_result : dict
        Output of run_prediction().
    borderline_profile : bool
        True when G2 is in the borderline range (10–12) and at least one
        additional risk indicator is present. Triggers a contextual warning
        between the main banner and the metric cards.

    Returns
    -------
    None
    """
    is_at_risk = prediction_result["is_at_risk"]
    prob_fail = prediction_result["probability_fail"]

    if is_at_risk:
        st.error(
            "⚠️ **AT RISK** — This student shows indicators associated with "
            "potential failure. Educator follow-up recommended."
        )
    else:
        st.success(
            "✅ **NOT AT RISK** — This student is not currently showing "
            "significant risk indicators."
        )

    if borderline_profile:
        st.warning(
            "⚠️ Borderline Academic Profile Detected — "
            "This student's Period 2 grade falls within the borderline range (10–12) "
            "and additional risk indicators are present in this profile. "
            "The estimated fail probability reflects the model's learned patterns "
            "and should be interpreted alongside these contextual indicators. "
            "Educator review is strongly recommended regardless of the probability score."
        )

    col1, col2 = st.columns(2)
    col1.metric("Estimated Fail Probability", f"{prob_fail * 100:.1f}%")
    col2.metric("Fail Risk Score", f"{prob_fail:.3f}")


# =============================================================================
# STUDENT SUMMARY CARD
# =============================================================================


def render_student_summary_card(input_dict: dict) -> None:
    """Display a styled summary card of key student profile values.

    Parameters
    ----------
    input_dict : dict
        Feature values from render_sidebar_inputs().

    Returns
    -------
    None
    """
    studytime_labels = {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}

    with st.container():
        st.subheader("📋 Student Profile Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Period 1 Grade:** {input_dict['G1']}/20")
            st.write(f"**Period 2 Grade:** {input_dict['G2']}/20")
            st.write(f"**Past Failures:** {input_dict['failures']}")
            st.write(f"**Absences:** {input_dict['absences']} days")
        with col2:
            st.write(f"**Study Time:** {studytime_labels[input_dict['studytime']]}")
            st.write(f"**School Support:** {input_dict['schoolsup'].title()}")
            st.write(f"**Higher Education Goal:** {input_dict['higher'].title()}")
            st.write(f"**Internet Access:** {input_dict['internet'].title()}")


# =============================================================================
# RISK GAUGE
# =============================================================================


def render_risk_gauge(prob_fail: float) -> None:
    """Display a three-band risk gauge (Low / Medium / High) highlighting the active band.

    The active band shows the actual fail probability percentage.
    The inactive bands show a dash. Emoji indicators provide colour context.

    Parameters
    ----------
    prob_fail : float
        Estimated probability of failure, in range [0, 1].

    Returns
    -------
    None
    """
    pct = prob_fail * 100
    st.write("**Risk Band Assessment**")
    col_low, col_med, col_high = st.columns(3)

    low_val  = f"{pct:.1f}%" if pct <= 33.0 else "—"
    med_val  = f"{pct:.1f}%" if 33.0 < pct <= 66.0 else "—"
    high_val = f"{pct:.1f}%" if pct > 66.0 else "—"

    col_low.metric("🟢 Low Risk (0–33%)", low_val)
    col_med.metric("🟡 Medium Risk (34–66%)", med_val)
    col_high.metric("🔴 High Risk (67–100%)", high_val)


# =============================================================================
# SMART RECOMMENDATIONS
# =============================================================================


def render_recommendations(prediction_result: dict, input_dict: dict) -> None:
    """Generate and display context-sensitive educator recommendations.

    Recommendations are built dynamically from the entered input values.
    Academic flags are listed first, then attendance, then support/access flags.

    Parameters
    ----------
    prediction_result : dict
        Output of run_prediction().
    input_dict : dict
        Feature values from render_sidebar_inputs().

    Returns
    -------
    None
    """
    G1 = input_dict["G1"]
    G2 = input_dict["G2"]
    failures = input_dict["failures"]
    absences = input_dict["absences"]
    studytime = input_dict["studytime"]
    schoolsup = input_dict["schoolsup"]
    higher = input_dict["higher"]
    internet = input_dict["internet"]
    is_at_risk = prediction_result["is_at_risk"]

    # Academic flags
    academic_flags = []
    if G2 < 10:
        academic_flags.append(
            "Period 2 grade is below the passing threshold. "
            "Priority academic intervention is recommended."
        )
    if G1 < 10 and G2 < 10:
        academic_flags.append(
            "Both period grades are below passing. "
            "Consider immediate structured support plan."
        )
    if failures > 0:
        academic_flags.append(
            f"Student has {failures} past failure(s). "
            "Discuss underlying causes and review current academic plan."
        )
    if failures >= 2:
        academic_flags.append(
            "Multiple past failures indicate a persistent pattern. "
            "Escalate to academic support team."
        )

    # Attendance flags
    attendance_flags = []
    if absences > 15:
        attendance_flags.append(
            f"Absence count of {absences} is significantly elevated. "
            "Initiate attendance review and investigate causes."
        )
    elif absences > 6:
        attendance_flags.append(
            f"Absence count of {absences} is above average. "
            "Monitor attendance closely."
        )

    # Support and access flags
    support_flags = []
    if studytime == 1:
        support_flags.append(
            "Student reports studying less than 2 hours per week. "
            "Discuss study habits and provide structured guidance."
        )
    if schoolsup == "no" and G2 < 12:
        support_flags.append(
            "Student is not receiving school support despite borderline grades. "
            "Consider referral to academic support."
        )
    if higher == "no":
        support_flags.append(
            "Student has not indicated interest in higher education. "
            "Discuss aspirations and long-term goals."
        )
    if internet == "no":
        support_flags.append(
            "Student lacks home internet access which may limit independent "
            "study and resource access."
        )

    all_flags = academic_flags + attendance_flags + support_flags

    st.subheader("💡 Educator Recommendations")

    if not all_flags:
        if not is_at_risk:
            st.success(
                "No immediate concerns identified. Continue regular monitoring "
                "and check in at the next scheduled review point."
            )
        else:
            st.warning(
                "Risk indicators detected. Schedule a one-to-one meeting with "
                "the student to discuss academic progress and wellbeing."
            )
    else:
        bullet_list = "\n".join(f"- {flag}" for flag in all_flags)
        if is_at_risk:
            st.warning(bullet_list)
        else:
            st.info(bullet_list)

    st.caption(
        "These recommendations are generated based on entered data and should be "
        "used as a starting point for educator judgement, not as a definitive "
        "action plan."
    )


# =============================================================================
# FEATURE IMPORTANCE CHART
# =============================================================================


def render_feature_importance_chart(top_features: dict) -> None:
    """Display a horizontal bar chart of global model feature importances.

    Raw sklearn feature names are mapped to human-readable labels via LABEL_MAPPING.
    Keys absent from the mapping fall back to their raw name.

    Parameters
    ----------
    top_features : dict
        Top 10 feature name → importance score (sorted descending by model.py).

    Returns
    -------
    None
    """
    st.subheader("📊 Model Feature Importances")

    raw_names = list(top_features.keys())
    values = list(top_features.values())
    display_names = [LABEL_MAPPING.get(n, n) for n in raw_names]

    reversed_names = list(reversed(display_names))
    reversed_values = list(reversed(values))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#161b27")
    ax.set_facecolor("#161b27")

    bars = ax.barh(reversed_names, reversed_values, color="#3b82f6")

    # Value labels at the end of each bar
    for bar, val in zip(bars, reversed_values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            color="#9ca3af",
            fontsize=9,
        )

    ax.set_xlabel("Importance Score", color="#f9fafb")
    ax.set_title(
        "Top 10 Feature Importances (Global)", fontsize=12, fontweight="bold", color="#f9fafb"
    )
    ax.tick_params(colors="#f9fafb", labelsize=9)
    ax.xaxis.grid(True, color="#1f2937", alpha=0.6)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1f2937")
    ax.spines["bottom"].set_color("#1f2937")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "ℹ️ These are global model feature importances showing which factors "
        "this model relies on most across all training students. They are not "
        "personalised to this individual student and do not explain why this "
        "specific prediction was made. Personalised explanations would require "
        "SHAP value analysis."
    )


# =============================================================================
# TRAINING PLOTS
# =============================================================================


def render_training_plots(plots_dir: Path) -> None:
    """Display all five training diagnostic plots saved by model.py.

    Shows a single info message if the plots/ folder or any expected file
    is missing, prompting the user to run model.py first.

    Parameters
    ----------
    plots_dir : Path
        Directory containing the PNG plot files.

    Returns
    -------
    None
    """
    plot_files = [
        (
            "class_distribution.png",
            "Class distribution showing 67% pass and 33% fail in the training data",
        ),
        (
            "correlation_heatmap.png",
            "Pearson correlation heatmap of numeric features including the pass/fail target",
        ),
        (
            "roc_curves.png",
            "ROC curves for all three models showing discriminative performance",
        ),
        (
            "confusion_matrix.png",
            "Confusion matrix for the final model on the held-out test set",
        ),
        (
            "feature_importances.png",
            "Top 10 global feature importances from the final trained model",
        ),
    ]

    if not plots_dir.exists() or not all(
        (plots_dir / fname).exists() for fname, _ in plot_files
    ):
        st.info("Training plots not found. Please run model.py first.")
        return

    for fname, caption in plot_files:
        st.image(str(plots_dir / fname), caption=caption, use_container_width=True)


# =============================================================================
# PERFORMANCE SECTION
# =============================================================================


def render_performance_section(metrics: dict) -> None:
    """Display model evaluation metrics inside a collapsible expander.

    Parameters
    ----------
    metrics : dict
        Loaded metrics from metrics.json.

    Returns
    -------
    None
    """
    with st.expander("📈 Model Performance and Evaluation", expanded=False):
        st.write(
            "The model was selected and evaluated using Stratified 5-Fold "
            "Cross-Validation on the 395-student UCI Student Performance dataset "
            "(Cortez & Silva, 2008), with a held-out 20% test set used for "
            "final metric reporting."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", metrics.get("best_model", "N/A"))
        col2.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        col3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
        col4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

        st.caption(
            "Accuracy is reported for reference only. Due to 67/33 class imbalance, "
            "F1 and ROC-AUC are the primary quality indicators."
        )


# =============================================================================
# FOOTER
# =============================================================================


def render_footer() -> None:
    """Render a styled HTML attribution footer consistent with the dark theme.

    Returns
    -------
    None
    """
    st.markdown(
        """
<div style="margin-top:2rem; padding:1rem 1.5rem;
            border-top:1px solid #1f2937; text-align:center;">
    <p style="color:#6b7280; font-size:0.78rem; margin:0;">
        Student Performance Risk Predictor &nbsp;·&nbsp;
        UCI Student Performance Dataset (Cortez &amp; Silva, 2008) &nbsp;·&nbsp;
        Final Year Project &nbsp;·&nbsp;
        All predictions must be reviewed by a qualified educator
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Entry point for the Streamlit application."""

    # Load artefacts — halts with st.error if model.py has not been run
    pipeline, top_features, metrics, optimal_threshold = load_model_assets(MODELS_DIR)

    inject_custom_css()

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="padding:1.5rem 0 0.5rem 0;
            border-bottom:1px solid #1f2937;
            margin-bottom:1.5rem;">
    <h1 style="color:#f9fafb; font-size:1.9rem; font-weight:700;
               letter-spacing:-0.02em; margin:0;">
        🎓 Student Performance Risk Predictor
    </h1>
    <p style="color:#9ca3af; font-size:0.95rem; margin:0.3rem 0 0 0;">
        Mid-year monitoring tool for secondary school educators &nbsp;·&nbsp;
        UCI Student Performance Dataset (Cortez &amp; Silva, 2008)
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar inputs ─────────────────────────────────────────────────────────
    input_dict = render_sidebar_inputs()
    st.sidebar.divider()
    predict_clicked = st.sidebar.button(
        "🔍 Predict Risk", type="primary", use_container_width=True
    )

    # ── Run prediction once so all tabs share the same result ──────────────────
    prediction_result = None
    borderline_profile = False
    if predict_clicked:
        input_df = build_input_dataframe(input_dict)
        prediction_result = run_prediction(pipeline, input_df, optimal_threshold)
        borderline_profile = (
            10 <= input_dict["G2"] <= 12 and
            any([
                input_dict["failures"] >= 1,
                input_dict["absences"] > 8,
                input_dict["studytime"] == 1,
                input_dict["G1"] < 10,
            ])
        )

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["🎯 Risk Assessment", "📊 Model Analysis", "📈 Training Plots"]
    )

    with tab1:
        if prediction_result is not None:
            render_prediction_output(prediction_result, borderline_profile)
            render_student_summary_card(input_dict)
            render_risk_gauge(prediction_result["probability_fail"])
            render_recommendations(prediction_result, input_dict)
        else:
            st.info(
                "👈 Configure student details in the sidebar and click **Predict Risk**."
            )
        render_footer()

    with tab2:
        render_feature_importance_chart(top_features)
        render_performance_section(metrics)
        render_footer()

    with tab3:
        render_training_plots(PLOTS_DIR)
        render_footer()


if __name__ == "__main__":
    main()
