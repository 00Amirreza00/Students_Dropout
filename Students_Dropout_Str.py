"""
Students' Dropout and Academic Success — Streamlit Dashboard
=============================================================
Interactive dashboard that compares multiple resampling strategies for handling
class imbalance in a student dropout prediction task.

Dataset
-------
UCI ML Repository — "Predict Students' Dropout and Academic Success"
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Models evaluated
----------------
- Random Forest Classifier (RFC) — baseline, weighted, + 4 resampling strategies
- K-Nearest Neighbours (KNN)     — without normalisation, with normalisation + 4 strategies

Usage
-----
    streamlit run Students_Dropout_Str.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_FILE = "data.csv"
TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
KNN_N_NEIGHBORS = 3

CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

CONTINUOUS_COLUMNS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

BINARY_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ORDINAL_COLUMNS = ["Application order"]

# Resampling method keys used in the training loops
RFC_METHOD_KEYS = ["RFC", "RFC_weighted", "SMOTE", "ROS", "RUS", "ADASYN"]
KNN_METHOD_KEYS = ["KNN_no_scaling", "KNN", "SMOTE", "ROS", "RUS", "ADASYN"]


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_clean_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the student dropout CSV and fix known column-name issues.

    Parameters
    ----------
    filepath : str
        Path to the semicolon-delimited CSV file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with corrected column names and typed columns.
    """
    df = pd.read_csv(filepath, delimiter=";")

    # Fix known column-name typos / trailing whitespace in the source file
    df.rename(
        columns={
            "Nacionality": "Nationality",
            "Daytime/evening attendance\t": "Daytime/evening attendance",
        },
        inplace=True,
    )

    # Cast column types
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    for col in BINARY_COLUMNS:
        df[col] = df[col].astype(bool)

    return df


def encode_target(labels: pd.Series):
    """
    Encode string class labels to integers.

    Parameters
    ----------
    labels : pd.Series
        Raw target column (e.g. "Dropout", "Graduate", "Enrolled").

    Returns
    -------
    encoded_labels : np.ndarray
        Integer-encoded labels.
    label_encoder : LabelEncoder
        Fitted encoder (used to recover class names for display).
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder


def split_dataset(features: pd.DataFrame, encoded_labels: np.ndarray):
    """
    Perform a stratified train/test split.

    Parameters
    ----------
    features : pd.DataFrame
    encoded_labels : np.ndarray

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        features,
        encoded_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


# ── Preprocessing transformers ────────────────────────────────────────────────

def build_rfc_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for the RFC pipeline.
    One-hot encodes categorical columns; passes all others through unchanged.
    """
    return ColumnTransformer(
        transformers=[
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
        ],
        remainder="passthrough",
    )


def build_knn_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for the KNN pipeline.
    One-hot encodes categorical columns and standard-scales continuous/ordinal ones.
    """
    return ColumnTransformer(
        transformers=[
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("scaler", StandardScaler(), CONTINUOUS_COLUMNS + ORDINAL_COLUMNS),
        ],
        remainder="passthrough",
    )


# ── Resampler factory ─────────────────────────────────────────────────────────

def get_resampler(method_key: str):
    """
    Return the appropriate imbalanced-learn resampler for the given key.

    Parameters
    ----------
    method_key : str
        One of "SMOTE", "ROS", "RUS", "ADASYN".

    Returns
    -------
    resampler
        A fitted-ready imbalanced-learn resampler instance.

    Raises
    ------
    ValueError
        If `method_key` is not a recognised resampling method.
    """
    resamplers = {
        "SMOTE": SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE),
        "ROS":   RandomOverSampler(random_state=RANDOM_STATE),
        "RUS":   RandomUnderSampler(random_state=RANDOM_STATE),
        "ADASYN": ADASYN(random_state=RANDOM_STATE),
    }
    if method_key not in resamplers:
        raise ValueError(
            f"Unknown resampling method: {method_key!r}. "
            f"Choose from {list(resamplers)}."
        )
    return resamplers[method_key]


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_and_evaluate_rfc_models(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Train all RFC variants and return a results DataFrame sorted by Macro F1.

    The preprocessor is fit once on the training split and applied to both
    train and test to avoid data leakage.

    Parameters
    ----------
    X_train_raw, X_test_raw : pd.DataFrame
        Raw (un-preprocessed) feature splits.
    y_train, y_test : np.ndarray
        Integer-encoded label splits.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: type, macro_f1_score, confusion_matrix.
        Sorted by macro_f1_score descending.
    """
    rfc_preprocessor = build_rfc_preprocessor()
    X_train_preprocessed = rfc_preprocessor.fit_transform(X_train_raw)
    X_test_preprocessed  = rfc_preprocessor.transform(X_test_raw)

    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    evaluation_records = []

    for method_key in RFC_METHOD_KEYS:
        if method_key == "RFC":
            X_train_resampled = X_train_preprocessed
            y_train_resampled = y_train
            classifier.fit(X_train_resampled, y_train_resampled)

        elif method_key == "RFC_weighted":
            # Assign per-sample weights proportional to inverse class frequency
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            classifier.fit(X_train_preprocessed, y_train, sample_weight=sample_weights)
            X_train_resampled = X_train_preprocessed  # not used for predict, just for clarity

        else:
            resampler = get_resampler(method_key)
            X_train_resampled, y_train_resampled = resampler.fit_resample(
                X_train_preprocessed, y_train
            )
            classifier.fit(X_train_resampled, y_train_resampled)

        predictions = classifier.predict(X_test_preprocessed)

        evaluation_records.append({
            "type": method_key,
            "macro_f1_score": f1_score(y_test, predictions, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, predictions, normalize="true"),
        })

    results_df = (
        pd.DataFrame(evaluation_records)
        .sort_values("macro_f1_score", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def train_and_evaluate_knn_models(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Train all KNN variants and return a results DataFrame sorted by Macro F1.

    Two preprocessor configurations are compared:
    - KNN_no_scaling : only one-hot encoding (no feature scaling)
    - All others     : one-hot encoding + StandardScaler

    Parameters
    ----------
    X_train_raw, X_test_raw : pd.DataFrame
    y_train, y_test : np.ndarray

    Returns
    -------
    results_df : pd.DataFrame
        Columns: type, macro_f1_score, confusion_matrix.
        Sorted by macro_f1_score descending.
    """
    evaluation_records = []

    # ── Variant 1: KNN without feature scaling ────────────────────────────────
    knn_pipeline_no_scaling = Pipeline([
        ("preprocessor", build_rfc_preprocessor()),   # one-hot only
        ("classifier",   KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)),
    ])
    knn_pipeline_no_scaling.fit(X_train_raw, y_train)
    predictions_no_scaling = knn_pipeline_no_scaling.predict(X_test_raw)

    evaluation_records.append({
        "type": "KNN_no_scaling",
        "macro_f1_score": f1_score(y_test, predictions_no_scaling, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, predictions_no_scaling, normalize="true"),
    })

    # ── Variant 2+: KNN with feature scaling (± resampling) ──────────────────
    knn_pipeline_scaled = Pipeline([
        ("preprocessor", build_knn_preprocessor()),   # one-hot + StandardScaler
        ("classifier",   KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)),
    ])

    for method_key in KNN_METHOD_KEYS:
        if method_key == "KNN_no_scaling":
            continue  # already handled above

        if method_key == "KNN":
            X_train_to_fit = X_train_raw
            y_train_to_fit = y_train
        else:
            # Resampling must happen on raw (un-transformed) data when using a Pipeline,
            # because the pipeline's preprocessor will transform during .fit()
            resampler = get_resampler(method_key)
            X_train_to_fit, y_train_to_fit = resampler.fit_resample(X_train_raw, y_train)

        knn_pipeline_scaled.fit(X_train_to_fit, y_train_to_fit)
        predictions = knn_pipeline_scaled.predict(X_test_raw)

        evaluation_records.append({
            "type": method_key,
            "macro_f1_score": f1_score(y_test, predictions, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, predictions, normalize="true"),
        })

    results_df = (
        pd.DataFrame(evaluation_records)
        .sort_values("macro_f1_score", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix_grid(
    results_df: pd.DataFrame,
    class_display_labels,
    figure_size: tuple = (15, 12),
) -> plt.Figure:
    """
    Plot a grid of normalised confusion matrices, one per model in results_df.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain "type", "macro_f1_score", "confusion_matrix" columns.
    class_display_labels : array-like
        Class names shown on the confusion matrix axes.
    figure_size : tuple
        Overall figure size in inches (width, height).

    Returns
    -------
    fig : plt.Figure
    """
    n_models = len(results_df)
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size)
    axes = axes.flatten()

    for plot_idx, row in results_df.iterrows():
        display = ConfusionMatrixDisplay(
            confusion_matrix=row["confusion_matrix"],
            display_labels=class_display_labels,
        )
        display.plot(ax=axes[plot_idx], colorbar=False, cmap="Blues")
        axes[plot_idx].set_title(
            f"{row['type']}\nMacro F1-score: {row['macro_f1_score']:.3f}"
        )

    # Hide unused subplot cells
    for empty_idx in range(n_models, len(axes)):
        axes[empty_idx].axis("off")

    plt.tight_layout()
    return fig


# ── Streamlit page layout ─────────────────────────────────────────────────────

def render_page_header():
    """Render the page title, dataset description, and class distribution chart."""
    st.image("picture1.jpg")
    st.title("Students' Dropout and Academic Success")

    st.write(
        "The dataset was collected from a higher education institution and covers students "
        "enrolled in various undergraduate programmes (agronomy, design, education, nursing, "
        "journalism, management, social service, and technology)."
    )
    st.write(
        "It includes information available at enrolment time — academic path, demographics, "
        "and socio-economic factors — as well as academic performance at the end of the first "
        "and second semesters."
    )
    st.write(
        "The classification task predicts one of three outcomes: **Dropout**, **Enrolled**, or "
        "**Graduate**. The dataset exhibits a strong class imbalance towards the Graduate class."
    )
    st.write(
        "**Source:** UC Irvine Machine Learning Repository — "
        "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
    )


def render_data_exploration_section(df: pd.DataFrame, imbalance_ratio: float):
    """Render the Data Exploration expander with a pie chart and statistics."""
    st.subheader("Data Exploration")

    with st.expander("See details"):
        st.write(f"The dataset contains **{df.shape[0]:,}** students and **{df.shape[1]}** features.")
        st.write(
            f"Feature breakdown: {len(CATEGORICAL_COLUMNS)} categorical, "
            f"{len(BINARY_COLUMNS)} binary, "
            f"{len(CONTINUOUS_COLUMNS) + len(ORDINAL_COLUMNS)} numerical."
        )
        st.write("There are no missing values in the feature columns.")

        class_counts = df[TARGET_COLUMN].value_counts()
        plt.figure(figsize=(4, 3))
        plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
        plt.title("Class Distribution in Target Variable")
        st.pyplot(plt)

        st.write(
            f"The imbalance ratio (largest / smallest class) is **{imbalance_ratio:.2f}**, "
            "indicating a significant skew in the target variable."
        )


def render_methodology_section():
    """Render the Data Modelling expander with method descriptions and illustrations."""
    st.subheader("Data Modelling and Evaluation")

    with st.expander("Evaluation metrics"):
        st.write(
            "Conventional accuracy is misleading on imbalanced data. This dashboard uses:"
        )
        st.markdown(
            "- **Macro-averaged F1-score**: computes the F1-score (harmonic mean of precision "
            "and recall) per class and averages them with equal weight, regardless of class size. "
            "This penalises poor performance on any class equally.\n"
            "- **Normalised confusion matrix**: shows the proportion of true instances per class "
            "that were assigned to each predicted class, making per-class error patterns visible."
        )

    with st.expander("Modelling approach"):
        st.write(
            "The dataset is split 80/20 (train/test). Two classifiers are evaluated:"
        )
        st.markdown(
            "- **Random Forest Classifier (RFC)**: baseline, class-weighted, "
            "and four resampling variants.\n"
            "- **K-Nearest Neighbours (KNN, k=3)**: without normalisation, "
            "with StandardScaler normalisation, and four resampling variants."
        )
        st.write("Resampling strategies:")

        smote_tab, ros_tab, rus_tab, adasyn_tab = st.tabs(["SMOTE", "ROS", "RUS", "ADASYN"])

        with smote_tab:
            st.write(
                "**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic "
                "minority-class samples by interpolating between existing instances in feature space, "
                "rather than simply duplicating them."
            )
            st.image("smote.jpg", caption="SMOTE illustration", use_container_width=True)

        with ros_tab:
            st.write(
                "**Random Over Sampling (ROS)** randomly duplicates existing minority-class "
                "samples until all classes reach the same size as the majority class."
            )
            st.image("ros.jpg", caption="ROS illustration", use_container_width=True)

        with rus_tab:
            st.write(
                "**Random Under Sampling (RUS)** randomly removes majority-class samples until "
                "all classes are the same size as the smallest minority class."
            )
            st.image("rus.jpg", caption="RUS illustration", use_container_width=True)

        with adasyn_tab:
            st.write(
                "**ADASYN** (Adaptive Synthetic Sampling) generates synthetic minority samples "
                "adaptively, focusing on regions of the feature space where the classifier "
                "finds it most difficult to learn the minority class."
            )
            st.image("adasyn.jpg", caption="ADASYN illustration", use_container_width=True)


def render_results_section(
    rfc_results_df: pd.DataFrame,
    knn_results_df: pd.DataFrame,
    class_display_labels,
):
    """
    Render the Results expander with confusion matrix grids and summary tables.

    Parameters
    ----------
    rfc_results_df : pd.DataFrame
        Output of train_and_evaluate_rfc_models.
    knn_results_df : pd.DataFrame
        Output of train_and_evaluate_knn_models.
    class_display_labels : array-like
        Class names for confusion matrix axis labels.
    """
    with st.expander("Results"):
        knn_tab, rfc_tab, table_tab = st.tabs(["KNN", "RFC", "Results table"])

        with rfc_tab:
            st.write(
                "Confusion matrices for the Random Forest Classifier (RFC) with different "
                "sampling strategies. The **RFC_weighted** variant assigns per-sample weights "
                "inversely proportional to class frequency, without modifying the training data."
            )
            st.write("Models are ordered by Macro F1-score (best first).")
            rfc_figure = plot_confusion_matrix_grid(rfc_results_df, class_display_labels)
            st.pyplot(rfc_figure)

        with knn_tab:
            st.write(
                "Confusion matrices for the KNN classifier. **KNN_no_scaling** uses only "
                "one-hot encoding; all other KNN variants additionally apply StandardScaler "
                "to continuous and ordinal features."
            )
            st.write("Models are ordered by Macro F1-score (best first).")
            knn_figure = plot_confusion_matrix_grid(knn_results_df, class_display_labels)
            st.pyplot(knn_figure)

        with table_tab:
            st.write("Summary of Macro F1-scores across all models (best first).")
            st.caption("KNN Classifier Results")
            st.table(knn_results_df.drop(columns=["confusion_matrix"]))
            st.caption("Random Forest Classifier Results")
            st.table(rfc_results_df.drop(columns=["confusion_matrix"]))


def render_conclusion_section(rfc_results_df: pd.DataFrame, knn_results_df: pd.DataFrame):
    """Render the Conclusion expander with dynamic best-model summaries."""
    st.subheader("Conclusion")

    best_rfc_f1   = rfc_results_df.iloc[0]["macro_f1_score"]
    best_rfc_name = rfc_results_df.iloc[0]["type"]
    best_knn_f1   = knn_results_df.iloc[0]["macro_f1_score"]
    best_knn_name = knn_results_df.iloc[0]["type"]

    with st.expander("See conclusion"):
        st.write(
            f"The best RFC variant is **{best_rfc_name}** with a Macro F1-score of "
            f"**{best_rfc_f1:.3f}**."
        )
        st.write(
            f"The best KNN variant is **{best_knn_name}** with a Macro F1-score of "
            f"**{best_knn_f1:.3f}**."
        )
        st.write(
            "Overall, the RFC with Random Oversampling performs best on this dataset. "
            "However, the most appropriate method depends on the specific dataset and "
            "the relative cost of different error types in the application context."
        )
        st.write(
            "The confusion matrices reveal that models generally classify the majority "
            "(Graduate) class well, while minority classes (Dropout, Enrolled) remain "
            "harder to predict correctly — highlighting the ongoing challenge of class imbalance."
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    """Build and render the full Streamlit dashboard."""

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_and_clean_dataset(DATA_FILE)

    class_counts    = df[TARGET_COLUMN].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()

    features = df.drop(columns=[TARGET_COLUMN])
    encoded_labels, label_encoder = encode_target(df[TARGET_COLUMN])
    class_display_labels = label_encoder.classes_

    X_train, X_test, y_train, y_test = split_dataset(features, encoded_labels)

    # ── Train models ──────────────────────────────────────────────────────────
    rfc_results_df = train_and_evaluate_rfc_models(X_train, X_test, y_train, y_test)
    knn_results_df = train_and_evaluate_knn_models(X_train, X_test, y_train, y_test)

    # ── Render page ───────────────────────────────────────────────────────────
    render_page_header()
    render_data_exploration_section(df, imbalance_ratio)
    render_methodology_section()
    render_results_section(rfc_results_df, knn_results_df, class_display_labels)
    render_conclusion_section(rfc_results_df, knn_results_df)


if __name__ == "__main__":
    main()
