"""
logreg_model.py
LOGISTIC REGRESSION BASELINE MODEL - MILESTONE 1

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 1 (First Run - Baseline Models)
Date: November 2025

This script trains a Logistic Regression model using the One-vs-Rest
strategy as an interpretable linear baseline for the Historical Military
Battles prediction project.

In Milestone 1, Logistic Regression serves as our starting point because
it is easy to understand and provides clear coefficient-based insights
into which features most strongly influence battle outcomes.

Purpose:
    • Provide a simple, highly interpretable linear model
    • Demonstrate proper feature scaling (StandardScaler)
    • Show coefficient-based feature importance
    • Generate diagnostic visualizations for class performance
    • Establish a benchmark before introducing non-linear models

Key Features:
    • Uses One-vs-Rest (OvR) strategy for multi-class classification
    • Balanced class weights to handle potential imbalance
    • Automatic timestamped results folder
    • Saves model, scaler, and all visualizations

Input:
    - models/data_split_enhanced.pkl (from preprocess.py)

Outputs (saved in results/YYYY-MM-DD_HH-MM-SS_logreg_model/):
    - 01_predicted_hist.png          → Predicted class distribution
    - 01_per_class_recall.png        → Recall per class
    - coefficients.png               → Top 10 features by coefficient magnitude
    - model.pkl                      → Trained model + scaler + metadata
    - summary.txt                    → Run summary
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# ====================== PATH CONFIGURATION ======================
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR    = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"

DATA_SPLIT_PATH = MODEL_DIR / "data_split_enhanced.pkl"
if not DATA_SPLIT_PATH.exists():
    raise FileNotFoundError(f"data_split_enhanced.pkl not found at {DATA_SPLIT_PATH}")

# Create a unique folder for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = RESULTS_ROOT / f"{RUN_TIMESTAMP}_logreg_model"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"Outputs saved to: {RUN_DIR}")
print(f"Using data: {DATA_SPLIT_PATH.name}")


def load_data():
    """
        Loads the preprocessed data and applies feature scaling.

        Logistic Regression benefits significantly from scaled features,
        so StandardScaler is applied here. The function also handles any
        remaining non-numeric columns for robustness.
        """
    data = joblib.load(DATA_SPLIT_PATH)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le      = data["label_encoder"]

    # Safety cleaning for any lingering objects/strings
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col]  = pd.to_numeric(X_test[col],  errors='coerce')

    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Loaded & scaled: X_train {X_train_scaled.shape}, X_test {X_test_scaled.shape}")
    print(f"Classes: {le.classes_}")
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


def train_logreg(X_train, y_train):
    """
        Trains a multi-class Logistic Regression model using One-vs-Rest strategy.

        This linear model is useful for establishing a simple, interpretable
        baseline. We use balanced class weights to better handle any imbalance
        in the distribution of battle outcomes.
        """

    logreg = LogisticRegression(
        multi_class='ovr',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    logreg.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return logreg


def evaluate_model(clf, X_test, y_test, le):
    """
        Trains a multi-class Logistic Regression model using One-vs-Rest strategy.

        This linear model is useful for establishing a simple, interpretable
        baseline. We use balanced class weights to better handle any imbalance
        in the distribution of battle outcomes.
        """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[MLP] Accuracy: {acc:.3f}")

    # 1) Predicted class distribution
    pred_counts = pd.Series(y_pred).value_counts().reindex(range(len(le.classes_)), fill_value=0)
    plt.figure()
    sns.barplot(x=le.classes_, y=pred_counts.values, palette="Blues_d")
    plt.title("MLP – Predicted Class Distribution")
    plt.xlabel("Predicted Class"); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "01_predicted_hist.png", dpi=300, bbox_inches="tight"); plt.close()

    # 2) Per-class recall
    cm = confusion_matrix(y_test, y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_recall = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0.0)

    plt.figure()
    sns.barplot(x=le.classes_, y=per_class_recall, palette="Greens_d")
    plt.title("MLP – Per-class Recall (TPR)")
    plt.xlabel("True Class"); plt.ylabel("Recall")
    plt.ylim(0, 1)
    for i, v in enumerate(per_class_recall):
        plt.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "01_per_class_recall.png", dpi=300, bbox_inches="tight"); plt.close()


def plot_coefficients(logreg, feature_names, le):
    """
        Creates a visualization of the top features based on coefficient magnitude.

        Because Logistic Regression is linear, the coefficients tell us the
        direction and strength of each feature's relationship with each battle
        outcome class. This makes the model valuable for interpretation.
        """
    coef_df = pd.DataFrame(
        logreg.coef_.T,
        index=feature_names,
        columns=[f"vs_{cls}" for cls in le.classes_]
    )

    # Top 10 features by average absolute coefficient
    top_features = coef_df.abs().sum(axis=1).nlargest(10).index
    coef_plot = coef_df.loc[top_features]

    plt.figure(figsize=(10, 6))
    coef_plot.plot(kind='barh', ax=plt.gca())
    plt.title("Top 10 Features by Coefficient Magnitude")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "coefficients.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: coefficients.png")


def main():
    """
        Runs the complete Logistic Regression modeling pipeline.
        This function coordinates data loading, model training, evaluation,
        and visualization to produce consistent, reproducible results.
        """
    X_train, X_test, y_train, y_test, le, scaler = load_data()
    logreg = train_logreg(X_train, y_train)
    evaluate_model(logreg, X_test, y_test, le)

    with open(MODEL_DIR / "feature_list.csv") as f:
        feature_names = [line.strip() for line in f.readlines()]
    plot_coefficients(logreg, feature_names, le)

    # Save model + metadata
    joblib.dump({
        "model": logreg,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names
    }, RUN_DIR / "model.pkl")
    print("Model + scaler saved: model.pkl")

    # Summary file
    with open(RUN_DIR / "summary.txt", "w") as f:
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, logreg.predict(X_test)):.3f}\n")
    print("Summary saved.")

if __name__ == "__main__":
    main()