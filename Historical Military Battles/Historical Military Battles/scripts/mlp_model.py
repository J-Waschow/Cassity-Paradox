"""
mlp_model.py
MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK BASELINE - MILESTONE 1

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 1 (First Run - Baseline Models)
Date: November 2025

This script trains a Multi-Layer Perceptron (MLP) neural network as a
strong non-linear baseline model for the Historical Military Battles
prediction project.

As part of Milestone 1, the MLP serves as our primary deep learning
approach, allowing us to capture complex patterns in the historical
battle data that simpler linear models might miss.

Purpose:
    • Establish a powerful non-linear model to compare against linear baselines
    • Demonstrate modern neural network implementation using scikit-learn
    • Generate rich diagnostic visualizations (confusion matrix, ROC curves, probability calibration)
    • Provide a strong benchmark before adding tactical features (supply_cut, naval_power)

Key Features:
    • Deep architecture: 128 → 64 → 32 hidden neurons
    • Early stopping to prevent overfitting
    • One-vs-Rest ROC curves with AUC scores
    • Prediction probability analysis
    • Automatic timestamped results folder

Input:
    - models/data_split_enhanced.pkl (produced by preprocess.py)

Outputs (saved in results/YYYY-MM-DD_HH-MM-SS_mlp_model/):
    - confusion.png          → Confusion matrix heatmap
    - roc_curves.png         → One-vs-Rest ROC curves with AUC
    - probabilities.png      → Boxplot of prediction probabilities
    - model.pkl              → Trained MLP model
    - summary.txt            → Performance summary
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

# Consistent visualization across all plots
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

# Creates a unique results folder for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = RESULTS_ROOT / f"{RUN_TIMESTAMP}_mlp_model"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"Outputs saved to: {RUN_DIR}")
print(f"Using data: {DATA_SPLIT_PATH.name}")


def load_data():
    """
        Loads the preprocessed battle dataset prepared by preprocess.py.

        This function handles the train/test split and performs basic cleaning
        to ensure all features are numeric, which is necessary for neural
        network training. It also prints a summary of the dataset dimensions
        and classes for transparency.
        """
    data = joblib.load(DATA_SPLIT_PATH)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le      = data["label_encoder"]

    # Convert and fix any non-numeric columns
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col]  = pd.to_numeric(X_test[col],  errors='coerce')
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Classes: {le.classes_}")
    return X_train, X_test, y_train, y_test, le


def train_mlp(X_train, y_train):
    """
        Trains a Multi-Layer Perceptron neural network classifier.

        The MLP is a fully connected feed-forward neural network. We use a
        three-hidden-layer architecture (128 → 64 → 32 neurons) with ReLU
        activation and the Adam optimizer. Early stopping is enabled to
        prevent overfitting on the historical battle data.
        """
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    mlp.fit(X_train, y_train)
    print("MLP trained.")
    return mlp


def evaluate_model(mlp, X_test, y_test, le):
    """
        Evaluates the trained MLP on the test set and generates a confusion matrix.

        This provides both overall accuracy and a detailed classification report.
        The confusion matrix visualization helps identify which battle outcome
        classes the model struggles with most.
        """
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)

    # Accuracy score for data visualizations
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")

    # Generate and save the classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))

    # Generate and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("MLP: Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "confusion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion.png")

    return y_pred, y_prob


def plot_roc_curves(mlp, X_test, y_test, le):
    """
        Plots One-vs-Rest ROC curves for all classes with their AUC scores.

        This visualization helps us understand how well the model can distinguish
        each battle outcome class from the others, which is especially useful
        in multi-class problems.
        """
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], mlp.predict_proba(X_test)[:, i])
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                        estimator_name=f'Class {le.classes_[i]}').plot(ax=plt.gca())

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MLP: ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: roc_curves.png")


def plot_prediction_probs(mlp, X_test, y_test, le):
    """
        Creates boxplots showing the distribution of predicted probabilities
        for each true class.

        This helps us evaluate model calibration and confidence — whether the
        model is appropriately certain or overly confident when making predictions
        about battle outcomes.
        """
    y_prob = mlp.predict_proba(X_test)
    df_prob = pd.DataFrame(y_prob, columns=le.classes_)
    df_prob['true'] = le.inverse_transform(y_test)
    df_prob = df_prob.melt(id_vars='true', var_name='predicted_class', value_name='probability')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_prob, x='true', y='probability', hue='predicted_class')
    plt.title("MLP: Prediction Probabilities by True Class")
    plt.xlabel("True Class")
    plt.ylabel("Predicted Probability")
    plt.legend(title="Predicted")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "probabilities.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: probabilities.png")


def main():
    """
        Runs the complete MLP modeling pipeline from data loading through
        training, evaluation, and visualization. This central function ensures
        all steps are executed in the correct order and produces consistent results.
        """
    X_train, X_test, y_train, y_test, le = load_data()
    mlp = train_mlp(X_train, y_train)
    y_pred, y_prob = evaluate_model(mlp, X_test, y_test, le)
    plot_roc_curves(mlp, X_test, y_test, le)
    plot_prediction_probs(mlp, X_test, y_test, le)

    # Save the trained model with meta data
    joblib.dump(mlp, RUN_DIR / "model.pkl")
    print("Model saved: model.pkl")

    # Summary
    with open(RUN_DIR / "summary.txt", "w") as f:
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Summary saved.")

if __name__ == "__main__":
    main()