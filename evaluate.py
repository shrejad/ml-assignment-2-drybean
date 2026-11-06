"""Shared metric definitions for training and the Streamlit app.

Precision, Recall, and F1 use a support-weighted average so larger bean
varieties influence the score in proportion to how often they appear.
AUC is weighted one-vs-rest, which is defined for this 7-class problem.
Weighted recall equals accuracy in single-label multiclass problems;
that equality is expected and is not a second copy of the accuracy code.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def classification_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
