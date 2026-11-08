"""
Train the 5 classifiers required by the assignment on the UCI Dry Bean
dataset, evaluate them on a held-out test split, and save everything the
Streamlit app needs.

Run on BITS Virtual Lab (take the screenshot while it runs):

    pip install -r requirements.txt
    python train_models.py

Optional arguments (defaults are set for the Dry Bean dataset):
    --dataset-file data/dry_bean_dataset.csv
    --target-column Class
    --columns-to-drop ""        comma-separated (for example an ID column)
    --test-fraction 0.2

Outputs:
    model/*.joblib                 one saved pipeline per classifier
    model/training_metadata.json   features, class names, metrics, versions
    model/metrics_table.md         comparison table for README.md
    test_data.csv                  held-out test rows (features + target)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_evaluation import (
    METRIC_DISPLAY_ORDER,
    METRICS_TABLE_FILE,
    MODEL_DIRECTORY,
    MODEL_FILE_NAME_BY_DISPLAY_NAME,
    TEST_DATA_FILE,
    encode_class_labels,
    evaluate_trained_pipeline,
    normalise_label_text,
    save_training_metadata,
)

RANDOM_SEED = 42  # fixed seed -> same split and same results on every run
MINIMUM_FEATURE_COUNT = 12  # assignment rule
MINIMUM_ROW_COUNT = 500  # assignment rule


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5 classifiers for ML Assignment 2")
    parser.add_argument("--dataset-file", default="data/dry_bean_dataset.csv")
    parser.add_argument("--target-column", default="Class")
    parser.add_argument("--columns-to-drop", default="")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def load_dataset(arguments: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series]:
    # utf-8-sig removes an invisible "byte order mark" that some CSV exports
    # put before the first column name.
    dataset = pd.read_csv(arguments.dataset_file, encoding="utf-8-sig")
    dataset.columns = [column_name.strip() for column_name in dataset.columns]

    if arguments.target_column not in dataset.columns:
        sys.exit(f"ERROR: target column '{arguments.target_column}' not found. Columns: {list(dataset.columns)}")

    columns_to_drop = [name.strip() for name in arguments.columns_to_drop.split(",") if name.strip()]
    unknown_columns = [name for name in columns_to_drop if name not in dataset.columns]
    if unknown_columns:
        sys.exit(f"ERROR: columns to drop not found: {unknown_columns}")
    dataset = dataset.drop(columns=columns_to_drop)

    # Rows without a label cannot be used for supervised learning.
    # Exact duplicate rows are removed so the same bean cannot appear in
    # both the training split and the test split (that would inflate scores).
    original_row_count = len(dataset)
    dataset = dataset.dropna(subset=[arguments.target_column]).drop_duplicates()
    print(f"Rows: {original_row_count} -> {len(dataset)} after removing missing labels and duplicates")

    feature_table = dataset.drop(columns=[arguments.target_column])
    raw_target_labels = dataset[arguments.target_column]

    print(f"Features: {feature_table.shape[1]} | Instances: {feature_table.shape[0]}")
    if feature_table.shape[1] < MINIMUM_FEATURE_COUNT or feature_table.shape[0] < MINIMUM_ROW_COUNT:
        print("WARNING: dataset is below the assignment minimum (12 features, 500 rows).")
    return feature_table, raw_target_labels


def build_preprocessing_step(feature_table: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessing that is saved INSIDE every pipeline, so the app never has
    to fit anything.

    - Numeric columns: fill missing values with the median, then scale to
      mean 0 and standard deviation 1. Scaling matters for Logistic
      Regression, KNN and SVM, which compare distances or weights across
      features. Without it, "Area" (values around 50,000) would dominate
      "ShapeFactor2" (values around 0.001).
    - Text columns (none in Dry Bean, kept for safety): fill missing values
      with the most frequent value, then one-hot encode.
    """
    numeric_column_names = feature_table.select_dtypes(include="number").columns.tolist()
    text_column_names = [name for name in feature_table.columns if name not in numeric_column_names]

    numeric_preprocessing = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("standardise", StandardScaler()),
    ])
    text_preprocessing = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric_features", numeric_preprocessing, numeric_column_names),
        ("text_features", text_preprocessing, text_column_names),
    ])


def build_classifiers() -> dict:
    """The 5 required classifiers with the settings used in this project."""
    return {
        # multinomial logistic regression (softmax over the 7 classes).
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
        # max_depth=10 and min_samples_leaf=5 stop the tree from memorising
        # single training rows (overfitting).
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_SEED),
        # 9 neighbours, nearer neighbours get a stronger vote.
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        # Gaussian NB, because all features are continuous measurements.
        "Naive Bayes": GaussianNB(),
        # probability=True is REQUIRED so predict_proba exists for AUC.
        "Support Vector Machine": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=RANDOM_SEED),
    }


def format_markdown_table(test_scores_by_model: dict[str, dict[str, float]]) -> str:
    header_row = "| ML Model Name | " + " | ".join(METRIC_DISPLAY_ORDER) + " |"
    divider_row = "|---|" + "---:|" * len(METRIC_DISPLAY_ORDER)
    body_rows = [
        f"| {model_name} | " + " | ".join(f"{scores[metric]:.4f}" for metric in METRIC_DISPLAY_ORDER) + " |"
        for model_name, scores in test_scores_by_model.items()
    ]
    return "\n".join([header_row, divider_row, *body_rows])


def main() -> None:
    arguments = parse_command_line()
    feature_table, raw_target_labels = load_dataset(arguments)

    # Class names are stored as sorted text so the app can map an uploaded
    # target column back to exactly the same integer codes.
    class_names = sorted(normalise_label_text(raw_target_labels).unique().tolist())
    target_class_codes = encode_class_labels(raw_target_labels, class_names)
    class_counts = dict(zip(class_names, np.bincount(target_class_codes).tolist()))
    print(f"Classes ({len(class_names)}): {class_counts}")

    # stratify keeps the class proportions equal in the train and test parts.
    (
        training_features, test_features,
        training_class_codes, test_class_codes,
        _, test_raw_labels,
    ) = train_test_split(
        feature_table, target_class_codes, raw_target_labels,
        test_size=arguments.test_fraction, stratify=target_class_codes, random_state=RANDOM_SEED,
    )
    print(f"Training rows: {len(training_features)} | Test rows: {len(test_features)}\n")

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    test_scores_by_model: dict[str, dict[str, float]] = {}
    training_accuracy_by_model: dict[str, float] = {}
    training_seconds_by_model: dict[str, float] = {}

    for model_name, classifier in build_classifiers().items():
        model_pipeline = Pipeline([
            ("preprocessing", build_preprocessing_step(training_features)),
            ("classifier", classifier),
        ])
        start_time = time.perf_counter()
        model_pipeline.fit(training_features, training_class_codes)
        training_seconds_by_model[model_name] = round(time.perf_counter() - start_time, 2)

        test_evaluation = evaluate_trained_pipeline(model_pipeline, test_features, test_class_codes, len(class_names))
        test_scores_by_model[model_name] = test_evaluation.metric_scores
        # Training accuracy is kept to compare with test accuracy:
        # a large gap means the model memorised the training data.
        training_accuracy_by_model[model_name] = float(
            accuracy_score(training_class_codes, model_pipeline.predict(training_features))
        )

        model_file = MODEL_DIRECTORY / MODEL_FILE_NAME_BY_DISPLAY_NAME[model_name]
        joblib.dump(model_pipeline, model_file)
        score_summary = ", ".join(f"{metric}={value:.4f}" for metric, value in test_evaluation.metric_scores.items())
        print(f"[{model_name}] {score_summary} | train acc={training_accuracy_by_model[model_name]:.4f} "
              f"| fit {training_seconds_by_model[model_name]}s -> {model_file.name}")

    # Save the test split with its ORIGINAL label text, exactly as a user
    # would upload it in the app.
    test_data = test_features.copy()
    test_data[arguments.target_column] = test_raw_labels.values
    test_data.to_csv(TEST_DATA_FILE, index=False)
    print(f"\nSaved {TEST_DATA_FILE.name} ({len(test_data)} rows)")

    save_training_metadata({
        "dataset_file": arguments.dataset_file,
        "target_column": arguments.target_column,
        "feature_columns": feature_table.columns.tolist(),
        "class_names": class_names,
        "class_counts": class_counts,
        "training_row_count": int(len(training_features)),
        "test_row_count": int(len(test_features)),
        "random_seed": RANDOM_SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_scores_by_model": test_scores_by_model,
        "training_accuracy_by_model": training_accuracy_by_model,
        "training_seconds_by_model": training_seconds_by_model,
    })

    comparison_table = format_markdown_table(test_scores_by_model)
    METRICS_TABLE_FILE.write_text(comparison_table + "\n", encoding="utf-8")
    print(f"\nComparison table (also saved to {METRICS_TABLE_FILE.relative_to(MODEL_DIRECTORY.parent)}):\n")
    print(comparison_table)


if __name__ == "__main__":
    main()
