"""
rf_model.py
RANDOM FOREST CLASSIFIER – MILESTONE 1 (Pre-Tactical Features)

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 1 (First Run - Baseline Models)
Date: November 2025

As part of Milestone 1, this baseline helps us understand how well an ensemble
of decision trees performs on the historical data before we introduce more
specialized tactical features. This script trains a Random Forest baseline using ALL features from the initial preprocessing
pipeline (preprocess.py → data_split_enhanced.pkl).

This version represents the state of the project in Milestone 1 — before the
introduction of the two high-impact theory-driven features derived from battle names:
    • supply_cut   → sieges, blockades, encirclements, pockets, cauldrons
    • naval_power  → naval battles, amphibious landings, fleet engagements

Purpose in final report:
    • Establishes a robust tree-based baseline (RF typically leads at this stage)
    • Demonstrates clean implementation with class weighting and permutation importance
    • Provides interpretable feature rankings (Gini + permutation)
    • Sets up clear "before vs. after" comparison when tactical factors are added

Outputs (saved to timestamped folder):
    1. Confusion matrix
    2. Top 10 features: Gini importance (left) + Permutation importance (right)
    3. Trained model (model.pkl)
    4. Summary text file with accuracy and notes
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

# Set visualization style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# ====================== PATH CONFIGURATION ======================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"

#
DATA_SPLIT_PATH = MODEL_DIR / "data_split_enhanced.pkl"
if not DATA_SPLIT_PATH.exists():
    raise FileNotFoundError(f"data_split_enhanced.pkl not found at {DATA_SPLIT_PATH}")

# Unique run folder to ensure outputs aren't overwritten
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = RESULTS_ROOT / f"{RUN_TIMESTAMP}_rf_model"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"Outputs saved to: {RUN_DIR}")
print(f"Using data: {DATA_SPLIT_PATH.name}")


# Data is loaded in dictionary format
def load_data():
    """
        Loads and prepares the preprocessed battle dataset for modeling.

        This function retrieves the train/test split created by preprocess.py,
        performs light safety cleaning on the data (converting any remaining
        object columns to numeric and filling NaNs), and provides a clear
        summary of the dataset being used.

        The cleaning steps ensure the model receives only numeric features,
        which is required by Random Forest.
        """
    data = joblib.load(DATA_SPLIT_PATH)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    le = data["label_encoder"]

    # Convert object columns to numeric
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Classes: {le.classes_}")
    return X_train, X_test, y_train, y_test, le


def train_rf(X_train, y_train):
    """
        Trains a Random Forest classifier on the battle features.

        Random Forest is chosen here because it handles mixed feature types well,
        is resistant to overfitting through ensemble averaging, and naturally
        provides feature importance scores. We use balanced class weights to
        account for any imbalance in battle outcome classes.
        """
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Random Forest trained.")
    return rf


def evaluate_model(rf, X_test, y_test, le):
    """
        Evaluates the trained model on the test set and creates a confusion matrix.

        This gives us both quantitative metrics (accuracy + per-class precision/recall)
        and a visual summary of where the model is making mistakes. The confusion
        matrix is particularly useful for understanding performance across different
        battle outcome directions.
        """
    y_pred = rf.predict(X_test)

    # Accuracy score for data visualizations
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")

    # Generate and save the classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))

    # Generate and save confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Random Forest: Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()

    # SAVE TO RUN_DIR
    plt.savefig(RUN_DIR / "confusion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion.png")

    return y_pred


def plot_feature_importance(rf, X_train, X_test, y_test):
    """
        Creates side-by-side visualizations of feature importance using two methods.

        Gini importance shows how much each feature contributes to reducing impurity
        across all trees. Permutation importance measures the drop in model performance
        when a feature's values are randomly shuffled — this is often more reliable
        when features are correlated.

        These plots help us understand which historical factors (terrain, surprise,
        commanders, etc.) are most predictive of battle direction.
        """
    importances = rf.feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]

    # Permutation importance (correlated features)
    perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances = perm_result.importances_mean
    perm_indices = np.argsort(perm_importances)[::-1]

    plt.figure(figsize=(10, 6))

    # Gini - side-by-side comparison plot
    plt.subplot(1, 2, 1)
    sns.barplot(x=importances[indices[:10]], y=feature_names[indices[:10]], palette="viridis")
    plt.title("Top 10 Features (Gini)")
    plt.xlabel("Importance")

    # Permutation
    plt.subplot(1, 2, 2)
    sns.barplot(x=perm_importances[perm_indices[:10]], y=feature_names[perm_indices[:10]], palette="magma")
    plt.title("Top 10 Features (Permutation)")
    plt.xlabel("Importance")

    plt.tight_layout()

    # Save to run directory
    plt.savefig(RUN_DIR / "importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: importance.png")

    # Print top 3
    print("\nTop 3 Features (Gini):")
    for i in range(min(3, len(indices))):
        print(f"  {i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")


def main():
    """
        Runs the complete Random Forest modeling workflow from data loading through
        evaluation and visualization. This central function ties everything together
        and ensures the pipeline runs consistently each time.
        """
    X_train, X_test, y_train, y_test, le = load_data()
    rf = train_rf(X_train, y_train)
    evaluate_model(rf, X_test, y_test, le)
    plot_feature_importance(rf, X_train, X_test, y_test)

    # Save model to directory
    joblib.dump(rf, RUN_DIR / "model.pkl")
    print("Model saved: model.pkl")

    # Summary
    with open(RUN_DIR / "summary.txt", "w") as f:
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}\n")
    print("Summary saved: summary.txt")


if __name__ == "__main__":
    main()