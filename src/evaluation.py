import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)


def cross_validate_model(model, X, y, scoring="roc_auc", cv=5):
    """
    Perform k-fold cross validation.

    Args:
        model: Trained sklearn model
        X: Features
        y: Labels
        scoring: Evaluation metric
        cv: Number of folds

    Returns:
        mean_score, std_score, scores
    """

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )

    return scores.mean(), scores.std(), scores


def evaluate_model(model, X, y):
    """
    Evaluate a trained model.

    Args:
        model: Trained classifier
        X: Features
        y: True labels

    Returns:
        Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "ROC AUC": roc_auc_score(y, y_prob),
    }

    return metrics


def print_classification_report(model, X, y):
    """
    Print classification report.
    """

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))


def plot_confusion_matrix(model, X, y, save_path=None):
    """
    Plot confusion matrix.
    """

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_roc_curve(model, X, y, save_path=None):
    """
    Plot ROC Curve.
    """

    RocCurveDisplay.from_estimator(
        model,
        X,
        y,
    )

    plt.title("ROC Curve")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()