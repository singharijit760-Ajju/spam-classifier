"""Streamlit interface for single-message and batch spam prediction."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from spam_utils import build_feature_frame, ensure_nltk_resources

MODEL_PATH = Path(__file__).with_name("spam_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    """Load the trained pipeline once per Streamlit process."""

    ensure_nltk_resources(verbose=False)
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def prediction_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the absolute LinearSVC decision scores as confidence values."""

    scores = pipeline.decision_function(features)
    return np.abs(np.asarray(scores).reshape(-1))


def prediction_label(value: object) -> str:
    return "SPAM" if str(value).strip().lower() == "spam" else "HAM"


def main() -> None:
    st.set_page_config(page_title="Spam Classifier", page_icon="✉️")
    st.title("Spam Classifier")
    st.write(
        "Classify messages with a TF-IDF and message-pattern model. "
        "Confidence is the absolute LinearSVC decision score, not a probability."
    )

    pipeline = load_pipeline()
    if pipeline is None:
        st.error("No trained model found. Run `python model_training.py` first.")
        st.info(
            "If data/mail_data.csv is not available yet, the training script can "
            "create a temporary demo model in memory so you can verify the app."
        )
        st.stop()

    st.subheader("Single message")
    message = st.text_area(
        "Message to classify",
        placeholder="Paste an email or text message here...",
        height=160,
    )
    if st.button("Classify message", type="primary"):
        if not message.strip():
            st.warning("Enter a message before classifying it.")
        else:
            features = build_feature_frame([message])
            result = prediction_label(pipeline.predict(features)[0])
            confidence = float(prediction_scores(pipeline, features)[0])
            if result == "SPAM":
                st.error(f"Prediction: {result}")
            else:
                st.success(f"Prediction: {result}")
            st.metric("Confidence score", f"{confidence:.4f}")

    st.divider()
    st.subheader("Batch prediction")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a Message column",
        type=["csv"],
    )
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
        else:
            if "Message" not in batch_data.columns:
                st.error("The uploaded CSV must contain a 'Message' column.")
            elif batch_data.empty:
                st.warning("The uploaded CSV does not contain any rows.")
            else:
                features = build_feature_frame(batch_data["Message"])
                predictions = pipeline.predict(features)
                scores = prediction_scores(pipeline, features)

                results = batch_data.copy()
                results["Prediction"] = [prediction_label(value) for value in predictions]
                results["Confidence"] = scores

                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()