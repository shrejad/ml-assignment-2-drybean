"""Train the five required classifiers and save a held-out test CSV.

The saved pipelines are what app.py loads. Nothing in the app fits a model.
Run this file on BITS Virtual Lab for the execution screenshot:

    python train_models.py
"""

import json
import warnings
import urllib.request
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import METRIC_KEYS, classification_metrics
from inference import load_encoder_and_pipelines, score_all

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dry_bean.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
TARGET_COLUMN = "Class"
RANDOM_STATE = 2026
TEST_SIZE = 0.2

MODEL_SPECS = [
    (
        "Logistic Regression",
        "logistic_regression.joblib",
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    ),
    (
        "Decision Tree",
        "decision_tree.joblib",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
    ),
    (
        "KNN",
        "knn.joblib",
        KNeighborsClassifier(n_neighbors=5),
    ),
    (
        "Naive Bayes",
        "naive_bayes.joblib",
        GaussianNB(),
    ),
    (
        "Support Vector Machine",
        "svm.joblib",
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE, cache_size=1000),
    ),
]


def download_dry_bean_csv(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(UCI_ZIP, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    archive = zipfile.ZipFile(BytesIO(payload))
    text = archive.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8")
    lines = text.splitlines()
    data_start = 0
    for index, line in enumerate(lines):
        if line.strip().upper().startswith("@DATA"):
            data_start = index + 1
            break
    frame = pd.read_csv(
        StringIO("\n".join(lines[data_start:])),
        header=None,
        names=FEATURE_COLUMNS + [TARGET_COLUMN],
    )
    frame.to_csv(destination, index=False)
    return frame


def load_dataset():
    if DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH)
    else:
        print("Local CSV not found. Downloading the UCI Dry Bean dataset...")
        frame = download_dry_bean_csv(DATA_PATH)
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise SystemExit("Dataset is missing columns: " + ", ".join(missing))
    if frame[FEATURE_COLUMNS].isna().any().any() or frame[TARGET_COLUMN].isna().any():
        raise SystemExit("Dataset has missing values. This assignment file expects a complete table.")
    if len(FEATURE_COLUMNS) < 12 or len(frame) < 500:
        raise SystemExit("Dataset is below the assignment minimum of 12 features and 500 rows.")
    return frame


def high_correlations(features, threshold=0.9):
    corr = features.corr().abs()
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if value >= threshold:
                pairs.append((left, right, value))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs


def build_pipeline(estimator):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def top_confusions(y_true, y_pred, class_names, limit=3):
    labels = list(range(len(class_names)))
    matrix = pd.crosstab(
        pd.Series(y_true, name="actual"),
        pd.Series(y_pred, name="predicted"),
    )
    found = []
    for actual in labels:
        for predicted in labels:
            if actual == predicted or actual not in matrix.index or predicted not in matrix.columns:
                continue
            count = int(matrix.loc[actual, predicted])
            if count:
                found.append((str(class_names[actual]), str(class_names[predicted]), count))
    found.sort(key=lambda item: item[2], reverse=True)
    return found[:limit]


def print_table(metric_rows):
    header = ["ML Model Name", *METRIC_KEYS]
    widths = [24] + [10] * len(METRIC_KEYS)
    line = " ".join(name.ljust(width) for name, width in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in metric_rows:
        cells = [row["model"].ljust(widths[0])]
        for key, width in zip(METRIC_KEYS, widths[1:]):
            cells.append(f"{row['test'][key]:.4f}".ljust(width))
        print(" ".join(cells))


def main():
    import numpy
    import pandas
    import sklearn

    warnings.filterwarnings(
        "ignore",
        message="The `probability` parameter was deprecated",
        category=FutureWarning,
    )

    print("===== ASSIGNMENT 2 MODEL EVALUATION =====")
    print(f"numpy {numpy.__version__} | pandas {pandas.__version__} | scikit-learn {sklearn.__version__}")
    frame = load_dataset()
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN].astype(str).str.strip()
    print(f"Rows: {len(frame)} | Features: {len(FEATURE_COLUMNS)} | Classes: {target.nunique()}")
    print("Class counts:")
    print(target.value_counts().to_string())
    print("Strongest feature pairs (absolute correlation):")
    for left, right, value in high_correlations(features)[:5]:
        print(f"  {left} ~ {right}: {value:.4f}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    majority = y_test.value_counts().idxmax()
    baseline = float((y_test == majority).mean())
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(f"Majority-class baseline on the test set ({majority}): {baseline:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    model_files = {}
    for name, filename, estimator in MODEL_SPECS:
        pipeline = build_pipeline(estimator)
        print(f"Fitting {name}...")
        started = perf_counter()
        pipeline.fit(x_train, y_train_enc)
        elapsed = perf_counter() - started
        train_pred = pipeline.predict(x_train)
        train_proba = pipeline.predict_proba(x_train)
        test_pred = pipeline.predict(x_test)
        test_proba = pipeline.predict_proba(x_test)
        train_metrics = classification_metrics(y_train_enc, train_pred, train_proba)
        test_metrics = classification_metrics(y_test_enc, test_pred, test_proba)
        joblib.dump(pipeline, MODEL_DIR / filename)
        model_files[name] = filename
        extra = {}
        classifier = pipeline.named_steps["classifier"]
        if name == "Decision Tree":
            order = np.argsort(classifier.feature_importances_)[::-1][:5]
            extra["top_features"] = [
                [FEATURE_COLUMNS[index], float(classifier.feature_importances_[index])]
                for index in order
            ]
            extra["depth"] = int(classifier.get_depth())
            extra["leaves"] = int(classifier.get_n_leaves())
        if name == "Support Vector Machine":
            extra["support_vectors"] = int(classifier.n_support_.sum())
            extra["train_rows"] = int(len(x_train))
        metric_rows.append(
            {
                "model": name,
                "file": filename,
                "fit_seconds": round(elapsed, 2),
                "train": train_metrics,
                "test": test_metrics,
                "confusions": top_confusions(y_test_enc, test_pred, list(encoder.classes_)),
                "extra": extra,
            }
        )
        print(f"  done in {elapsed:.1f}s | test F1 {test_metrics['F1']:.4f} | test accuracy {test_metrics['Accuracy']:.4f}")

    raw_knn = KNeighborsClassifier(n_neighbors=5)
    raw_knn.fit(x_train, y_train_enc)
    raw_accuracy = float(accuracy_score(y_test_enc, raw_knn.predict(x_test)))
    print(
        "Diagnostic only, not a submitted model: "
        f"KNN without scaling has test accuracy {raw_accuracy:.4f}."
    )

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_frame.to_csv(TEST_PATH, index=False)

    winner = max(
        metric_rows,
        key=lambda row: (row["test"]["F1"], row["test"]["MCC"], row["test"]["AUC"], row["test"]["Accuracy"]),
    )
    schema = {
        "dataset": "UCI Dry Bean Dataset",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_encoder": "label_encoder.joblib",
        "model_files": model_files,
        "model_order": [name for name, _, _ in MODEL_SPECS],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
        },
        "metric_average": {
            "precision_recall_f1": "weighted",
            "auc": "weighted one-vs-rest",
        },
        "class_names": [str(name) for name in encoder.classes_],
        "versions": {"scikit-learn": sklearn_version},
    }
    report = {
        "n_rows": int(len(frame)),
        "n_features": len(FEATURE_COLUMNS),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "class_counts": {key: int(value) for key, value in target.value_counts().items()},
        "baseline_accuracy": baseline,
        "baseline_class": str(majority),
        "correlations_above_0_90": [
            {"left": left, "right": right, "abs_corr": value}
            for left, right, value in high_correlations(features)
        ],
        "models": metric_rows,
        "winner_rule": "Highest test weighted F1, then MCC, then AUC, then accuracy.",
        "winner": winner["model"],
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn_version,
        },
    }
    with (MODEL_DIR / "schema.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n===== COMPARISON TABLE (held-out test set) =====")
    print_table(metric_rows)
    print(f"\nWinner by the stated rule: {winner['model']}")
    print("\nTrain accuracy vs test accuracy (overfit check):")
    for row in metric_rows:
        gap = row["train"]["Accuracy"] - row["test"]["Accuracy"]
        print(
            f"  {row['model']}: train {row['train']['Accuracy']:.4f} | "
            f"test {row['test']['Accuracy']:.4f} | gap {gap:.4f}"
        )

    print("\nReloading saved models and rescoring test_data.csv...")
    encoder_loaded, pipelines = load_encoder_and_pipelines(schema)
    rescored = score_all(pd.read_csv(TEST_PATH), schema, encoder_loaded, pipelines)
    for row in metric_rows:
        fresh = rescored["results"][row["model"]]["metrics"]
        for key in METRIC_KEYS:
            delta = abs(fresh[key] - row["test"][key])
            if delta > 1e-9:
                raise SystemExit(f"Saved {row['model']} {key} drifted by {delta}")
    print("Saved models match the table above on test_data.csv.")
    print(f"Wrote {TEST_PATH}")
    print("===== END =====")


if __name__ == "__main__":
    main()
