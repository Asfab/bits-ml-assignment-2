"""
Shared evaluation code used by BOTH the training script (train_models.py)
and the Streamlit app (app.py).

Why one shared module?
    The numbers in README.md come from the training script, and the numbers
    on the web page come from the app. If each file computed metrics in its
    own way (for example with a different averaging method), the two sets of
    numbers could disagree. Keeping one implementation guarantees they match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# File locations. Paths are relative to this file, so they work the same way
# on a laptop, on BITS Virtual Lab and on Streamlit Community Cloud.
# ---------------------------------------------------------------------------
PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = PROJECT_ROOT_DIRECTORY / "model"
TRAINING_METADATA_FILE = MODEL_DIRECTORY / "training_metadata.json"
METRICS_TABLE_FILE = MODEL_DIRECTORY / "metrics_table.md"
TEST_DATA_FILE = PROJECT_ROOT_DIRECTORY / "test_data.csv"

# Display name of each classifier -> file name of its saved pipeline.
# This order is used everywhere: README table, app dropdown, comparison tab.
MODEL_FILE_NAME_BY_DISPLAY_NAME: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "k_nearest_neighbors.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

METRIC_DISPLAY_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@dataclass(frozen=True)
class ModelEvaluation:
    """Result of running one trained model on one labelled data set."""

    metric_scores: dict[str, float]
    predicted_class_codes: np.ndarray


# ---------------------------------------------------------------------------
# Metadata (written by training, read by the app)
# ---------------------------------------------------------------------------
def load_training_metadata(metadata_file: Path = TRAINING_METADATA_FILE) -> dict:
    with open(metadata_file, encoding="utf-8") as metadata_stream:
        return json.load(metadata_stream)


def save_training_metadata(training_metadata: dict, metadata_file: Path = TRAINING_METADATA_FILE) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as metadata_stream:
        json.dump(training_metadata, metadata_stream, indent=2)


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------
def normalise_label_text(raw_labels: pd.Series) -> pd.Series:
    """
    Turn every label into clean text.

    A CSV round trip can change the label 1 into "1" or "1.0", and labels can
    have stray spaces. Comparing clean text avoids false "unknown label" errors.
    """
    label_text = raw_labels.astype(str).str.strip()
    return label_text.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


def encode_class_labels(raw_labels: pd.Series, class_names: list[str]) -> np.ndarray:
    """
    Convert class names (for example "SEKER") into integer codes 0..K-1,
    using the class order saved at training time.

    An unknown label raises a clear error, instead of silently producing
    wrong metrics.
    """
    # Matching ignores upper/lower case, so "seker" and "Seker" both mean "SEKER".
    original_label_text = normalise_label_text(raw_labels)
    label_text = original_label_text.str.casefold()
    class_code_by_name = {class_name.casefold(): class_code for class_code, class_name in enumerate(class_names)}

    # Report unknown labels exactly as they are written in the file.
    unknown_labels = sorted(set(original_label_text[~label_text.isin(class_code_by_name)]))
    if unknown_labels:
        raise ValueError(
            f"Target column has labels not seen during training: {unknown_labels[:10]}. "
            f"Expected labels: {class_names}"
        )
    return label_text.map(class_code_by_name).to_numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_evaluation_metrics(
    true_class_codes: np.ndarray,
    predicted_class_codes: np.ndarray,
    predicted_probabilities: np.ndarray | None,
    class_count: int,
) -> dict[str, float]:
    """
    Compute the 6 metrics required by the assignment.

    Averaging rule:
      - Binary problem      -> Precision / Recall / F1 of the positive class (code 1).
      - Multi-class problem -> "weighted" average: each class's score is
                               weighted by the number of rows of that class.

    AUC (Area Under the ROC Curve) needs predicted PROBABILITIES, not labels.
      - Binary      -> probability of class 1.
      - Multi-class -> "one-vs-rest": one AUC per class (that class against
                       all others), then a weighted average.
    If AUC cannot be computed (for example the data contains only one class),
    the value is NaN instead of an error.
    """
    is_binary_problem = class_count == 2
    averaging_method = "binary" if is_binary_problem else "weighted"

    metric_scores = {
        "Accuracy": accuracy_score(true_class_codes, predicted_class_codes),
        "Precision": precision_score(true_class_codes, predicted_class_codes, average=averaging_method, zero_division=0),
        "Recall": recall_score(true_class_codes, predicted_class_codes, average=averaging_method, zero_division=0),
        "F1": f1_score(true_class_codes, predicted_class_codes, average=averaging_method, zero_division=0),
        "MCC": matthews_corrcoef(true_class_codes, predicted_class_codes),
        "AUC": float("nan"),
    }

    # MCC needs at least two different classes among the true and predicted
    # labels. With only one class, scikit-learn returns 0, which would wrongly
    # suggest a bad model, so we report "not defined" (NaN) instead.
    if len(set(np.asarray(true_class_codes).tolist()) | set(np.asarray(predicted_class_codes).tolist())) < 2:
        metric_scores["MCC"] = float("nan")

    if predicted_probabilities is not None:
        try:
            if is_binary_problem:
                metric_scores["AUC"] = roc_auc_score(true_class_codes, predicted_probabilities[:, 1])
            else:
                metric_scores["AUC"] = roc_auc_score(
                    true_class_codes,
                    predicted_probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=list(range(class_count)),
                )
        except ValueError:
            # Raised when the data does not contain every class. Keep NaN.
            pass

    return {metric_name: float(metric_scores[metric_name]) for metric_name in METRIC_DISPLAY_ORDER}


def evaluate_trained_pipeline(
    trained_pipeline,
    feature_table: pd.DataFrame,
    true_class_codes: np.ndarray,
    class_count: int,
) -> ModelEvaluation:
    """Predict with an already trained pipeline (no fitting here) and score it."""
    predicted_class_codes = trained_pipeline.predict(feature_table)
    predicted_probabilities = (
        trained_pipeline.predict_proba(feature_table) if hasattr(trained_pipeline, "predict_proba") else None
    )
    metric_scores = compute_evaluation_metrics(
        true_class_codes, predicted_class_codes, predicted_probabilities, class_count
    )
    return ModelEvaluation(metric_scores=metric_scores, predicted_class_codes=predicted_class_codes)


def build_confusion_matrix(true_class_codes, predicted_class_codes, class_count: int) -> np.ndarray:
    # Passing all labels keeps the full K x K size even if a class is
    # missing from the uploaded file.
    return confusion_matrix(true_class_codes, predicted_class_codes, labels=list(range(class_count)))


def find_relevant_class_codes(true_class_codes, predicted_class_codes) -> list[int]:
    """Classes that appear in the data OR in the predictions (sorted)."""
    return sorted(set(np.asarray(true_class_codes).tolist()) | set(np.asarray(predicted_class_codes).tolist()))


def build_classification_report(true_class_codes, predicted_class_codes, class_names: list[str]) -> pd.DataFrame:
    """
    Per-class precision / recall / F1.

    Only classes that appear in the data or in the predictions are listed.
    A class that is in neither would be an all-zero row, and those rows
    would wrongly pull the "macro avg" (plain average over rows) down.
    """
    relevant_class_codes = find_relevant_class_codes(true_class_codes, predicted_class_codes)
    report_by_class = classification_report(
        true_class_codes,
        predicted_class_codes,
        labels=relevant_class_codes,
        target_names=[class_names[class_code] for class_code in relevant_class_codes],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report_by_class).transpose()
