"""
Dry Bean Variety Classifier — Streamlit app for ML Assignment 2.

Required features:
  a. Upload a TEST CSV (16 features + "Class" column)
  b. Choose a model from a dropdown
  c. Show the 6 evaluation metrics
  d. Show the confusion matrix and the classification report

The app NEVER retrains. It only loads the saved pipelines from model/
and calls predict / predict_proba on the uploaded rows.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model_evaluation import (
    METRIC_DISPLAY_ORDER,
    MODEL_DIRECTORY,
    MODEL_FILE_NAME_BY_DISPLAY_NAME,
    TEST_DATA_FILE,
    TRAINING_METADATA_FILE,
    build_classification_report,
    build_confusion_matrix,
    encode_class_labels,
    evaluate_trained_pipeline,
    load_training_metadata,
)

st.set_page_config(page_title="Dry Bean Variety Classifier", page_icon="🫘", layout="wide")

# Short, general descriptions of each variety, shown in the "About" tab.
BEAN_VARIETY_NOTES = {
    "BARBUNYA": "Large, oval, speckled beans",
    "BOMBAY": "Very large, round beans (rarest class)",
    "CALI": "Large, kidney-shaped white beans",
    "DERMASON": "Small, flat white beans (most common class)",
    "HOROZ": "Long, slim beans",
    "SEKER": "Round, medium-sized white beans",
    "SIRA": "Medium-sized, slightly oval beans",
}


# ---------------------------------------------------------------------------
# Loading (cached: files are read once per app process, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the five trained models...")
def load_trained_pipelines() -> dict:
    trained_pipelines = {}
    for model_name, model_file_name in MODEL_FILE_NAME_BY_DISPLAY_NAME.items():
        model_file = MODEL_DIRECTORY / model_file_name
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file: model/{model_file_name}. Run train_models.py first.")
        trained_pipelines[model_name] = joblib.load(model_file)
    return trained_pipelines


@st.cache_data
def load_metadata() -> dict:
    return load_training_metadata(TRAINING_METADATA_FILE)


@st.cache_data
def read_uploaded_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    # sep=None lets pandas detect ',' or ';' automatically.
    uploaded_table = pd.read_csv(io.BytesIO(uploaded_bytes), sep=None, engine="python", encoding="utf-8-sig")
    uploaded_table.columns = [column_name.strip() for column_name in uploaded_table.columns]
    return uploaded_table


def find_upload_problems(uploaded_table: pd.DataFrame, training_metadata: dict) -> list[str]:
    """Return readable problems with the uploaded file; an empty list means it is usable."""
    problems = []
    missing_feature_columns = [
        column_name for column_name in training_metadata["feature_columns"] if column_name not in uploaded_table.columns
    ]
    if missing_feature_columns:
        problems.append(f"Missing feature columns: {missing_feature_columns}")
    if training_metadata["target_column"] not in uploaded_table.columns:
        problems.append(f"Missing target column: '{training_metadata['target_column']}'")
    if uploaded_table.empty:
        problems.append("The file has no rows.")
    return problems


def draw_confusion_matrix(matrix_counts: np.ndarray, class_names: list[str], chart_title: str):
    figure, axes = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(
        matrix_counts, annot=True, fmt="d", cmap="YlOrBr", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes, linewidths=0.5, linecolor="white",
    )
    axes.set_xlabel("Predicted variety")
    axes.set_ylabel("Actual variety")
    axes.set_title(chart_title)
    plt.setp(axes.get_xticklabels(), rotation=35, ha="right")
    figure.tight_layout()
    return figure


def draw_metric_ranking(scores_by_model: pd.Series):
    """
    Horizontal bars sorted from best to worst. The x-axis starts just below
    the lowest score (not at 0), because all models score close to each
    other and bars starting at 0 would look identical.
    """
    ranked_scores = scores_by_model.dropna().sort_values()
    figure, axes = plt.subplots(figsize=(8, 3.2))
    bar_colours = ["#b5651d" if model_name == ranked_scores.idxmax() else "#e3c08d" for model_name in ranked_scores.index]
    axes.barh(ranked_scores.index, ranked_scores.values, color=bar_colours)
    lowest_score = ranked_scores.min() if not ranked_scores.empty else 0.0
    axes.set_xlim(max(0.0, lowest_score - 0.03), 1.0)
    for row_position, score in enumerate(ranked_scores.values):
        axes.text(score + 0.002, row_position, f"{score:.4f}", va="center", fontsize=9)
    axes.set_xlabel(f"{scores_by_model.name} (axis zoomed to show differences)")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def format_score(score: float) -> str:
    return "n/a" if np.isnan(score) else f"{score:.4f}"


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------
try:
    training_metadata = load_metadata()
    trained_pipelines = load_trained_pipelines()
except FileNotFoundError as missing_file_error:
    st.error(f"The app is not ready: {missing_file_error}")
    st.stop()

class_names: list[str] = training_metadata["class_names"]
class_count = len(class_names)
target_column = training_metadata["target_column"]

# ---------------------------------------------------------------------------
# Sidebar: data source and model choice
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("① Test data")
    uploaded_file = st.file_uploader(
        "Upload a labelled test CSV",
        type=["csv"],
        help=f"Needs the {len(training_metadata['feature_columns'])} measurement columns and the "
             f"'{target_column}' column. Upload test data only (free tier has limited memory).",
    )
    use_bundled_test_data = st.toggle("Use the bundled test_data.csv", value=uploaded_file is None)

    if TEST_DATA_FILE.exists():
        st.download_button(
            "⬇ Download sample test CSV",
            data=TEST_DATA_FILE.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            help="Download this file, then upload it above to try the app.",
        )

    st.header("② Classifier")
    selected_model_name = st.selectbox("Choose a model", list(MODEL_FILE_NAME_BY_DISPLAY_NAME.keys()))

    st.divider()
    st.caption(
        f"Trained on {training_metadata['training_row_count']:,} beans · "
        f"{len(training_metadata['feature_columns'])} features · {class_count} varieties · "
        f"scikit-learn {training_metadata['sklearn_version']}"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🫘 Dry Bean Variety Classifier")
st.markdown(
    "Which of **7 dry bean varieties** is this bean? Each row is one bean photographed by a "
    "computer-vision system and described by 16 shape measurements. "
    "Upload labelled test data to see how five **pre-trained** classifiers perform. "
    "The models are loaded from disk and are **never retrained** on your file."
)

if uploaded_file is not None and not use_bundled_test_data:
    evaluation_table = read_uploaded_csv(uploaded_file.getvalue())
    data_source_label = uploaded_file.name
elif use_bundled_test_data and TEST_DATA_FILE.exists():
    evaluation_table = pd.read_csv(TEST_DATA_FILE)
    data_source_label = "bundled test_data.csv"
else:
    st.info("👈 Upload a test CSV in the sidebar, or switch on the bundled test data.")
    st.stop()

upload_problems = find_upload_problems(evaluation_table, training_metadata)
if upload_problems:
    for problem_message in upload_problems:
        st.error(problem_message)
    st.stop()

evaluation_features = evaluation_table[training_metadata["feature_columns"]]
try:
    true_class_codes = encode_class_labels(evaluation_table[target_column], class_names)
except ValueError as label_error:
    st.error(str(label_error))
    st.stop()

with st.expander(f"Preview: {data_source_label} — {len(evaluation_table):,} rows"):
    st.dataframe(evaluation_table.head(20), width="stretch")

selected_model_tab, comparison_tab, about_tab = st.tabs(
    ["🔍 Selected model", "📊 Compare all models", "ℹ️ About the data & models"]
)

# --- Tab 1: one model in detail -------------------------------------------
with selected_model_tab:
    selected_evaluation = evaluate_trained_pipeline(
        trained_pipelines[selected_model_name], evaluation_features, true_class_codes, class_count
    )

    st.subheader(f"{selected_model_name} — evaluation metrics")
    metric_columns = st.columns(len(METRIC_DISPLAY_ORDER))
    for metric_column, metric_name in zip(metric_columns, METRIC_DISPLAY_ORDER):
        metric_column.metric(metric_name, format_score(selected_evaluation.metric_scores[metric_name]))
    st.caption("Precision, Recall and F1 are weighted averages over the 7 classes; AUC is one-vs-rest weighted.")
    if np.isnan(selected_evaluation.metric_scores["AUC"]):
        st.caption("AUC shows n/a because the uploaded data does not contain all 7 varieties.")

    confusion_column, report_column = st.columns([1.1, 1])
    with confusion_column:
        st.markdown("**Confusion matrix**")
        confusion_counts = build_confusion_matrix(true_class_codes, selected_evaluation.predicted_class_codes, class_count)
        confusion_figure = draw_confusion_matrix(confusion_counts, class_names, selected_model_name)
        st.pyplot(confusion_figure)
        plt.close(confusion_figure)  # free memory on the small free tier
    with report_column:
        st.markdown("**Classification report**")
        report_table = build_classification_report(true_class_codes, selected_evaluation.predicted_class_codes, class_names)
        st.dataframe(report_table.style.format("{:.3f}"), width="stretch")

    predictions_table = evaluation_table.copy()
    predictions_table["predicted_variety"] = [class_names[code] for code in selected_evaluation.predicted_class_codes]
    predictions_table["is_correct"] = predictions_table["predicted_variety"] == predictions_table[target_column].astype(str)
    st.download_button(
        "⬇ Download predictions as CSV",
        data=predictions_table.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# --- Tab 2: all models side by side ---------------------------------------
with comparison_tab:
    comparison_rows = []
    for model_name, trained_pipeline in trained_pipelines.items():
        model_scores = evaluate_trained_pipeline(trained_pipeline, evaluation_features, true_class_codes, class_count).metric_scores
        comparison_rows.append({"ML Model Name": model_name, **model_scores})
    comparison_table = pd.DataFrame(comparison_rows).set_index("ML Model Name")

    st.subheader(f"All five models on {data_source_label}")
    st.dataframe(
        comparison_table.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#f6d7a7"),
        width="stretch",
    )
    best_model_name = comparison_table["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best_model_name}** "
               f"(MCC {comparison_table.loc[best_model_name, 'MCC']:.4f})")

    metric_to_plot = st.radio("Metric to plot", METRIC_DISPLAY_ORDER, horizontal=True, index=5)
    st.pyplot(draw_metric_ranking(comparison_table[metric_to_plot]))
    plt.close("all")

# --- Tab 3: background -------------------------------------------------------
with about_tab:
    variety_column, training_column = st.columns(2)
    with variety_column:
        st.markdown("**Bean varieties (classes) in the training data**")
        variety_table = pd.DataFrame({
            "Variety": class_names,
            "Rows": [training_metadata["class_counts"][name] for name in class_names],
            "Description": [BEAN_VARIETY_NOTES.get(name, "") for name in class_names],
        })
        st.dataframe(variety_table, hide_index=True, width="stretch")
    with training_column:
        st.markdown("**Training accuracy vs test accuracy**")
        st.caption("A large gap means the model memorised the training data (overfitting).")
        fit_table = pd.DataFrame({
            "Training accuracy": training_metadata["training_accuracy_by_model"],
            "Test accuracy": {name: scores["Accuracy"] for name, scores in training_metadata["test_scores_by_model"].items()},
            "Fit time (s)": training_metadata["training_seconds_by_model"],
        })
        fit_table["Gap"] = fit_table["Training accuracy"] - fit_table["Test accuracy"]
        st.dataframe(fit_table.style.format({"Training accuracy": "{:.4f}", "Test accuracy": "{:.4f}",
                                             "Gap": "{:+.4f}", "Fit time (s)": "{:.2f}"}), width="stretch")
    st.markdown(
        "Source: Koklu, M. & Ozkan, I. A. (2020), *Multiclass classification of dry beans using computer "
        "vision and machine learning techniques*, UCI Machine Learning Repository, dataset 602."
    )
