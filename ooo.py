# ...existing code...

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# Standard split (train/test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Define models (pipelines with scaling where appropriate)
models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, random_state=42)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVC": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
}

# Train, predict, and evaluate on the hold-out test set
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # get probabilities for ROC AUC (use predict_proba if available, else decision_function)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback for estimators without predict_proba
        y_prob = model.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    })
    print(f"\n=== {name} ===")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

# Summarize results
results_df = pd.DataFrame(results).set_index("model")
print("\nSummary (test set):")
print(results_df.sort_values(by="f1", ascending=False))

# Optional: 5-fold cross-validated metrics on the full dataset for more robust comparison
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_summary = []
for name, model in models.items():
    cv = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1, return_train_score=False)
    cv_summary.append({
        "model": name,
        "accuracy_mean": np.mean(cv["test_accuracy"]),
        "precision_mean": np.mean(cv["test_precision"]),
        "recall_mean": np.mean(cv["test_recall"]),
        "f1_mean": np.mean(cv["test_f1"]),
        "roc_auc_mean": np.mean(cv["test_roc_auc"])
    })
cv_df = pd.DataFrame(cv_summary).set_index("model")
print("\n5-fold CV summary:")
print(cv_df.sort_values(by="f1_mean", ascending=False))

# ...existing code...