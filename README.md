# Spam Classifier

A Python spam/ham classifier built with scikit-learn. The model combines
TF-IDF word and bigram features with message-level signals such as message
length, punctuation, uppercase usage, and exclamation marks. A Streamlit app
supports both individual messages and CSV batch predictions.

## Project files

- `model_training.py` — loads the dataset, preprocesses messages, tunes a
  `LinearSVC`, prints evaluation metrics, and saves `spam_pipeline.pkl`.
- `spam_utils.py` — shared NLTK preprocessing and feature extraction used by
  both training and the app.
- `app.py` — Streamlit user interface for single and batch predictions.
- `requirements.txt` — Python dependencies.
- `spam_pipeline.pkl` — generated trained pipeline required by the app.
- `data/mail_data.csv` — your dataset; it is intentionally not included.

## Dataset format

Place your own file at `data/mail_data.csv`. It must contain exactly these
important columns:

```csv
Category,Message
ham,"Are we still meeting at 4?"
spam,"Congratulations, you won a prize! Click now."
```

`Category` may contain `ham`/`spam` or `0`/`1`. The training set must contain
at least five examples of each class because the requested hyperparameter
tuning uses five-fold cross-validation.

## Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The training and app scripts attempt to download the required NLTK resources
(`punkt`, `stopwords`, `wordnet`, and `omw-1.4`) automatically. If a download
is unavailable, the scripts use safe preprocessing fallbacks where possible.

## Train the model

After placing your dataset in `data/mail_data.csv`, run:

```bash
python model_training.py
```

The script uses:

- lowercasing
- URL, email, digit, and punctuation removal
- tokenization, lemmatization, and English stopword removal
- TF-IDF with unigrams and bigrams, capped at 10,000 features
- five numeric message features
- a balanced `LinearSVC` with `dual=False`
- grid search over `C = [0.1, 0.5, 1.0, 2.0, 5.0]`

It prints the best `C`, cross-validation accuracy, test accuracy, and a
classification report, then saves the complete fitted pipeline to
`spam_pipeline.pkl`.

If the dataset is missing, the script uses a tiny in-memory demo dataset only
to validate that the project runs. It does not write that data to the
repository. Replace it with your own `data/mail_data.csv` and retrain before
using the model for real predictions.

## Run the app

```bash
streamlit run app.py
```

The app loads `spam_pipeline.pkl`, classifies a single message, and accepts a
CSV upload with a `Message` column. Batch results include `Prediction` and the
absolute `decision_function` value as `Confidence`, and can be downloaded as
a new CSV.

## Deploy to Streamlit Cloud

1. Push this project to a GitHub repository.
2. Make sure `requirements.txt`, `app.py`, and the trained
   `spam_pipeline.pkl` are committed.
3. Train with your real dataset before committing the pickle file. The
   original training CSV is ignored and is not required by the app at runtime.
4. In Streamlit Cloud, create a new app, choose the repository and branch, and
   set the main file to `app.py`.
5. Deploy. NLTK resources are downloaded by the app when it starts.

Do not commit private or sensitive training data. The generated pickle file
contains the fitted model and vocabulary, so treat it as part of the model
artifact.