"""Train and save the TF-IDF plus extra-features spam classifier."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from spam_utils import (
    NUMERIC_FEATURES,
    build_feature_frame,
    ensure_nltk_resources,
)

DATA_PATH = Path("data/mail_data.csv")
MODEL_PATH = Path("spam_pipeline.pkl")
RANDOM_STATE = 42
C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]


FALLBACK_ROWS = [
    ("ham", "Hi, are we still meeting for lunch today?"),
    ("ham", "Please remember to bring the project notes tomorrow."),
    ("ham", "Your appointment is confirmed for Monday at ten."),
    ("ham", "Thanks for sending the photos. They look great."),
    ("ham", "Can you call me when you arrive at the station?"),
    ("ham", "The report is attached for your review."),
    ("ham", "Dinner is ready whenever you get home."),
    ("ham", "I will pick up groceries on my way back."),
    ("ham", "Happy birthday! Hope you have a wonderful day."),
    ("ham", "The class starts at nine in room three."),
    ("ham", "Let me know if you need any help with that."),
    ("ham", "Meeting moved to Thursday afternoon."),
    ("spam", "Congratulations! You have won a free cash prize. Click now!"),
    ("spam", "URGENT: claim your reward at www.example.com before it expires!"),
    ("spam", "You have been selected for an exclusive free vacation."),
    ("spam", "Win guaranteed money today by replying YES to this message."),
    ("spam", "Limited offer! Get cheap loans with no credit check."),
    ("spam", "You are a lucky winner. Call now to collect your prize."),
    ("spam", "Earn thousands from home with this secret opportunity."),
    ("spam", "Free entry in our weekly contest. Text WIN to 80000."),
    ("spam", "Act now to receive your bonus gift card."),
    ("spam", "Your account has won a cash bonus. Verify immediately."),
    ("spam", "Special promotion: buy one and get five free today."),
    ("spam", "Exclusive deal ends tonight. Click the link to claim."),
]


def _normalise_label(value: object) -> str | None:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"ham", "0"}:
            return "ham"
        if normalised in {"spam", "1"}:
            return "spam"
    elif isinstance(value, (bool, int, float, np.integer, np.floating)) and not pd.isna(
        value
    ):
        if float(value) == 0:
            return "ham"
        if float(value) == 1:
            return "spam"
    return None


def load_dataset() -> tuple[pd.DataFrame, bool]:
    """Load the user's CSV, or return an in-memory demo dataset when absent."""

    if DATA_PATH.exists():
        dataset = pd.read_csv(DATA_PATH)
        used_fallback = False
    else:
        warnings.warn(
            f"{DATA_PATH} was not found. Training a tiny in-memory demo model. "
            "Replace it with your own data/mail_data.csv and retrain for useful results.",
            UserWarning,
        )
        dataset = pd.DataFrame(FALLBACK_ROWS, columns=["Category", "Message"])
        used_fallback = True

    required_columns = {"Category", "Message"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    dataset = dataset[["Category", "Message"]].dropna(subset=["Category", "Message"])
    dataset["label"] = dataset["Category"].map(_normalise_label)
    if dataset["label"].isna().any():
        invalid = dataset.loc[dataset["label"].isna(), "Category"].unique().tolist()
        raise ValueError(
            "Category values must be ham/spam or 0/1. "
            f"Invalid value(s): {invalid}"
        )

    dataset["Message"] = dataset["Message"].astype(str)
    counts = dataset["label"].value_counts()
    if set(counts.index) != {"ham", "spam"}:
        raise ValueError("The dataset must contain both ham and spam examples.")
    if counts.min() < 5:
        raise ValueError(
            "At least 5 examples of each class are required for the requested "
            "5-fold cross-validation."
        )

    return dataset, used_fallback


def create_pipeline() -> Pipeline:
    """Build the preprocessing and LinearSVC pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
                "clean_text",
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    dual=False,
                    max_iter=5000,
                ),
            ),
        ]
    )


def main() -> None:
    ensure_nltk_resources(verbose=True)
    dataset, used_fallback = load_dataset()
    features = build_feature_frame(dataset["Message"])
    labels = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    grid_search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid={"model__C": C_VALUES},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(x_train, y_train)

    best_pipeline = grid_search.best_estimator_
    test_predictions = best_pipeline.predict(x_test)

    print(f"Best C: {grid_search.best_params_['model__C']}")
    print(f"Cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(best_pipeline, model_file)

    print(f"Saved pipeline to {MODEL_PATH}")
    if used_fallback:
        print(
            "WARNING: This is a demo model trained on an in-memory fallback dataset. "
            "Place your own data/mail_data.csv in the data/ folder and rerun training."
        )


if __name__ == "__main__":
    main()