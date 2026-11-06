"""Dry bean variety classifier.

The five models are loaded from model/*.joblib. Uploaded rows are scored
only. This file never calls fit().
"""

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluate import METRIC_KEYS
from inference import (
    DEFAULT_TEST_PATH,
    largest_confusion,
    load_encoder_and_pipelines,
    load_schema,
    score_all,
)

st.set_page_config(
    page_title="Dry Bean Variety Classifier",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f3f6ef;
        border: 1px solid #d5dfcc;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def artifacts():
    schema = load_schema()
    encoder, pipelines = load_encoder_and_pipelines(schema)
    return schema, encoder, pipelines


@st.cache_data(show_spinner="Scoring the five saved models on this test file...")
def cached_score(csv_bytes, schema_token):
    del schema_token
    schema, encoder, pipelines = artifacts()
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    return score_all(frame, schema, encoder, pipelines)


def report_frame(report):
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "Class": label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1-score"],
                "Support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows)


def draw_confusion(matrix, labels, title):
    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
        cbar=False,
    )
    axis.set_xlabel("Predicted variety")
    axis.set_ylabel("Actual variety")
    axis.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    figure.tight_layout()
    return figure


def winner_name(results, model_order):
    def sort_key(name):
        metrics = results[name]["metrics"]
        return (metrics["F1"], metrics["MCC"], metrics["AUC"], metrics["Accuracy"])

    return max(model_order, key=sort_key)


schema, encoder, pipelines = artifacts()
model_order = schema["model_order"]

st.title("Dry Bean Variety Classifier")
st.write(
    "Seven registered dry-bean varieties are predicted from 16 geometric measurements. "
    "Every number on this page comes from a saved classifier. The test file is not used to train or update the models."
)

with st.sidebar:
    st.header("Test data")
    upload = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help="The file must contain the 16 feature columns and the Class column. Use the held-out test split, not the training rows.",
    )
    st.header("Model selection")
    selected = st.selectbox("Choose a saved classifier", model_order)
    st.caption("Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine.")
    st.caption("Precision, Recall, and F1 are weighted averages. AUC is weighted one-vs-rest.")

if upload is None:
    csv_bytes = DEFAULT_TEST_PATH.read_bytes()
    source = "Bundled held-out test file (test_data.csv)"
else:
    csv_bytes = upload.getvalue()
    source = "Uploaded file: " + upload.name

try:
    scored = cached_score(csv_bytes, json.dumps(schema["model_files"]))
except Exception as exc:
    st.error(str(exc))
    st.stop()

selected_result = scored["results"][selected]
metrics = selected_result["metrics"]
labels = scored["labels"]

st.subheader("Evaluation data")
left, right = st.columns(2)
left.write(source)
right.write(f"{scored['n_rows']} rows, {len(schema['feature_columns'])} features, target column `{schema['target_column']}`")

st.subheader("Evaluation metrics")
cards = st.columns(6)
for card, key in zip(cards, METRIC_KEYS):
    card.metric(key, f"{metrics[key]:.4f}")

st.caption(
    "Weighted recall matches accuracy because each bean has one true variety. "
    "That match is a property of the weighted-recall definition, and the per-class recall below is not the same number."
)

matrix_col, report_col = st.columns([1.15, 0.85])
with matrix_col:
    st.subheader("Confusion matrix")
    figure = draw_confusion(
        selected_result["confusion_matrix"],
        labels,
        selected,
    )
    st.pyplot(figure)
    plt.close(figure)
    mixup = largest_confusion(selected_result["confusion_matrix"], labels)
    if mixup:
        st.caption(
            f"Most common mix-up: actual {mixup['actual']} predicted as {mixup['predicted']} "
            f"({mixup['count']} test rows)."
        )
with report_col:
    st.subheader("Classification report")
    table = report_frame(selected_result["report"])
    st.dataframe(
        table.style.format(
            {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"},
            na_rep="",
        ),
        hide_index=True,
        width="stretch",
    )

actual = scored["actual"]
predicted = selected_result["predicted"]
mistakes = [
    {"Actual variety": truth, "Predicted variety": guess}
    for truth, guess in zip(actual, predicted)
    if truth != guess
]
st.subheader("Misclassified test rows")
st.write(f"{len(mistakes)} of {len(actual)} rows were misclassified by {selected}.")
if mistakes:
    st.dataframe(pd.DataFrame(mistakes).head(12), hide_index=True, width="stretch")

st.subheader("All five models on this test file")
comparison_rows = []
for name in model_order:
    row = {"ML Model Name": name}
    row.update(scored["results"][name]["metrics"])
    comparison_rows.append(row)
comparison = pd.DataFrame(comparison_rows)
st.dataframe(
    comparison.style.format({key: "{:.4f}" for key in METRIC_KEYS}),
    hide_index=True,
    width="stretch",
)
leading = winner_name(scored["results"], model_order)
st.success(
    f"On this file, {leading} leads by weighted F1, with MCC, AUC, and accuracy as tie-breaks."
)

with st.expander("Metric definitions and model files"):
    st.write(
        "Accuracy is the share of test beans assigned the correct variety. "
        "AUC uses the saved model's class probabilities. "
        "Precision, Recall, and F1 are weighted by the number of test rows in each variety. "
        "MCC is the multiclass Matthews correlation, from -1 to +1."
    )
    st.write("Saved files: " + ", ".join(f"`model/{filename}`" for filename in schema["model_files"].values()))
    st.write("Label mapping file: `model/label_encoder.joblib`. Each pipeline starts with a StandardScaler that was fit on the training split only.")
