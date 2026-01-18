import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def add_cm_to_results_dict(results_dict, y_true, y_pred, prefix="test"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results_dict[f"{prefix}_tn"] = int(tn)
    results_dict[f"{prefix}_fp"] = int(fp)
    results_dict[f"{prefix}_fn"] = int(fn)
    results_dict[f"{prefix}_tp"] = int(tp)
    return results_dict



def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    y_pred_train = model.predict(X_train.values)
    y_pred_test = model.predict(X_test.values)

    y_prob_train = model.foward(X_train.values)
    y_prob_test = model.foward(X_test.values)

    results = {}
    results["model"] = model_name

    results["train_accuracy"] = accuracy_score(y_train, y_pred_train)
    results["train_precision"] = precision_score(y_train, y_pred_train, zero_division=0)
    results["train_recall"] = recall_score(y_train, y_pred_train, zero_division=0)
    results["train_f1"] = f1_score(y_train, y_pred_train, zero_division=0)

    if len(np.unique(y_train)) == 2:
        results["train_roc_auc"] = roc_auc_score(y_train, y_prob_train)
    else:
        results["train_roc_auc"] = np.nan

    results["test_accuracy"] = accuracy_score(y_test, y_pred_test)
    results["test_precision"] = precision_score(y_test, y_pred_test, zero_division=0)
    results["test_recall"] = recall_score(y_test, y_pred_test, zero_division=0)
    results["test_f1"] = f1_score(y_test, y_pred_test, zero_division=0)

    if len(np.unique(y_test)) == 2:
        results["test_roc_auc"] = roc_auc_score(y_test, y_prob_test)
    else:
        results["test_roc_auc"] = np.nan

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    return results, cm_train, cm_test


def majority_baseline(y_train, y_test):
    majority_class = int(y_train.value_counts().idxmax())
    y_pred_base = np.full_like(y_test, fill_value=majority_class)

    baseline = {
        "model": f"Baseline majority={majority_class}",
        "test_accuracy": accuracy_score(y_test, y_pred_base),
        "test_precision": precision_score(y_test, y_pred_base, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_base, zero_division=0),
        "test_f1": f1_score(y_test, y_pred_base, zero_division=0),
        "test_roc_auc": np.nan,
    }

    print("\nMajority class:", majority_class)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
    print(classification_report(y_test, y_pred_base, zero_division=0))

    return baseline
