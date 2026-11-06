"""Load the saved pipelines and score a test CSV. No fitting happens here."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from evaluate import classification_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SCHEMA_PATH = MODEL_DIR / "schema.json"
DEFAULT_TEST_PATH = ROOT / "test_data.csv"


def load_schema():
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_encoder_and_pipelines(schema):
    encoder = joblib.load(MODEL_DIR / schema["label_encoder"])
    pipelines = {}
    for name, filename in schema["model_files"].items():
        pipelines[name] = joblib.load(MODEL_DIR / filename)
    return encoder, pipelines


def prepare_test_frame(frame, feature_columns, target_column):
    renamed = {column: column.strip() for column in frame.columns}
    frame = frame.rename(columns=renamed)
    required = list(feature_columns) + [target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "The test CSV is missing required columns: " + ", ".join(missing)
        )
    features = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Non-numeric values found in feature columns: " + ", ".join(bad)
        )
    target = frame[target_column].astype(str).str.strip()
    if (target == "").any() or target.str.lower().isin(["nan", "none"]).any():
        raise ValueError("The target column has empty values.")
    return features, target


def score_pipeline(pipeline, encoder, features, target):
    known = set(encoder.classes_)
    unknown = sorted(set(target.unique()) - known)
    if unknown:
        allowed = ", ".join(encoder.classes_)
        raise ValueError(
            "Unknown class labels "
            + ", ".join(unknown)
            + ". Expected: "
            + allowed
        )
    y_true = encoder.transform(target)
    y_pred = pipeline.predict(features)
    y_proba = pipeline.predict_proba(features)
    if y_proba.shape[1] != len(encoder.classes_):
        raise ValueError("Saved model probability columns do not match the label encoder.")
    metrics = classification_metrics(y_true, y_pred, y_proba)
    labels = list(encoder.classes_)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    predicted_names = encoder.inverse_transform(y_pred)
    return metrics, matrix, report, predicted_names


def score_all(frame, schema, encoder, pipelines):
    features, target = prepare_test_frame(
        frame, schema["feature_columns"], schema["target_column"]
    )
    labels = list(encoder.classes_)
    actual = target.tolist()
    results = {}
    for name, pipeline in pipelines.items():
        metrics, matrix, report, predicted = score_pipeline(
            pipeline, encoder, features, target
        )
        results[name] = {
            "metrics": metrics,
            "confusion_matrix": matrix.astype(int).tolist(),
            "report": report,
            "predicted": predicted.tolist(),
        }
    return {
        "labels": labels,
        "actual": actual,
        "n_rows": int(len(actual)),
        "results": results,
    }


def metrics_frame(results, model_order):
    rows = []
    for name in model_order:
        row = {"ML Model Name": name}
        row.update(results[name]["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def largest_confusion(matrix, labels):
    grid = np.array(matrix, dtype=int)
    off = grid.copy()
    np.fill_diagonal(off, 0)
    if off.sum() == 0:
        return None
    flat = int(off.argmax())
    row, col = divmod(flat, off.shape[1])
    return {
        "actual": labels[row],
        "predicted": labels[col],
        "count": int(off[row, col]),
    }
