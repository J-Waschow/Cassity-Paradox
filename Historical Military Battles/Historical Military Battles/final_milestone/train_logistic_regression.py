"""
train_logistic_regression.py
OPTIMIZED LOGISTIC REGRESSION - MILESTONE 3

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 3
Date: December 2025

This script trains a refined and optimized multinomial Logistic Regression
model as the primary **interpretable baseline** for Milestone 3.

It builds directly on the theory-aligned features and tactical variables
developed in Milestone 2, with important improvements in regularization,
interaction terms, and model selection methodology.

Key Changes from Milestone 2:
    • Replaced standard LogisticRegression + SMOTE with LogisticRegressionCV
      (automatic regularization tuning via cross-validation).
    • Simplified tactical keyword patterns for better maintainability.
    • Added "surprise_sum" as a new interaction term to better capture
      combined effects of surprise with supply and naval factors.
    • Removed manual SMOTE pipeline in favor of built-in class_weight + CV.
    • Output structure changed to final_milestone/results/ for project culmination.
    • Stronger emphasis on regularization and interpretability for final report.

Purpose:
    Demonstrate that a well-tuned linear model grounded in classical war
    theory (logistics, sea power, surprise, etc.) remains highly competitive
    even against more complex models in Milestone 3.
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 7)

# Output directory for final milestone
fig_path = Path("final_milestone/results/figures")
fig_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function for the Milestone 3 Logistic Regression pipeline."""
    print("Loading data...")
    data = joblib.load("../models/data_split_enhanced.pkl")
    X_train = data["X_train"].copy()
    X_test  = data["X_test"].copy()
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le = data["label_encoder"]

    """
    ====================== TACTICAL FEATURE ENGINEERING ======================
    Hard-coded tactical features carried over from Milestone 2
    These are intentionally hard-coded using keyword patterns on battle names
    to directly inject military theory (logistics disruption and sea power)
    """
    battles_df = pd.read_excel("../data/battles.xlsx")
    all_names = battles_df["name"].fillna("").astype(str).tolist()
    train_names = all_names[:len(X_train)]
    test_names  = all_names[len(X_train):len(X_train) + len(X_test)]
    def add_feature(series, pattern):
        return series.str.contains(pattern, case=False, regex=True).fillna(False).astype(int)

    # Supply Cut Feature - Logistics disruption (sieges, encirclements, etc.)
    X_train["supply_cut"] = add_feature(pd.Series(train_names),
        r"siege|blockade|leningrad|stalingrad|singapore|malta|tobruk|bastogne|dien bien phu|pocket|kessel|surrounded|cut off|cauldron|isolated|encircled")

    X_test["supply_cut"] = add_feature(pd.Series(test_names),
        r"siege|blockade|leningrad|stalingrad|singapore|malta|tobruk|bastogne|dien bien phu|pocket|kessel|surrounded|cut off|cauldron|isolated|encircled")

    # Naval Power Feature - Maritime and amphibious operations
    X_train["naval_power"] = add_feature(pd.Series(train_names),
        r"naval|midway|leyte|guadalcanal|trafalgar|normandy|inchon|okinawa|tarawa|amphibious|landing|beach|dardanelles|gallipoli|java sea|coral sea")

    X_test["naval_power"] = add_feature(pd.Series(test_names),
        r"naval|midway|leyte|guadalcanal|trafalgar|normandy|inchon|okinawa|tarawa|amphibious|landing|beach|dardanelles|gallipoli|java sea|coral sea")

    print(f"Tactical features added — supply_cut: {X_train['supply_cut'].sum() + X_test['supply_cut'].sum()}, "
          f"naval_power: {X_train['naval_power'].sum() + X_test['naval_power'].sum()}")

    # Same core feature set used in Milestone 2 for consistency and interpretability
    base_features = [
        "duration_days", "multi_phase", "force_ratio",
        "supply_cut", "naval_power",
        "surprise_level_Complete surprise achieved by attacker",
        "surprise_level_Substantial surprise achieved by attacker",
        "surprise_level_Minor surprise achieved by attacker",
        "surprise_level_Neither side achieved surprise, or it did not affect the outcome",
        "surprise_level_None",
        "air_superiority_Attacker had air superiority in the theater",
        "air_superiority_Neither side had theater air superiority",
        "air_superiority_None",
        "tactical_posture_Unknown",
        "terra_B", "terra_D", "terra_F", "terra_G", "terra_M", "terra_R", "terra_U", "terra_W"
    ]

    X_train = X_train[base_features]
    X_test  = X_test[base_features]

    """ 
    ====================== INTERACTION TERMS (Milestone 3 Enhancement) ======================
    New in Milestone 3: Create interaction terms to capture combined effects
    This allows the linear model to better represent theory (e.g., surprise is more 
    powerful when combined with supply disruption or naval superiority)
    """
    surprise_cols = [c for c in base_features if "surprise_level" in c]
    X_train["surprise_sum"] = X_train[surprise_cols].sum(axis=1)
    X_test["surprise_sum"]  = X_test[surprise_cols].sum(axis=1)
    X_train["supply_surprise"] = X_train["supply_cut"] * X_train["surprise_sum"]
    X_test["supply_surprise"]  = X_test["supply_cut"]  * X_test["surprise_sum"]
    X_train["naval_surprise"]  = X_train["naval_power"] * X_train["surprise_sum"]
    X_test["naval_surprise"]   = X_test["naval_power"]  * X_test["surprise_sum"]

    final_features = base_features + ["supply_surprise", "naval_surprise"]

    # Data Cleaning: Fill missing values using median then zero as fallback
    X_train = X_train[final_features].fillna(X_train[final_features].median(numeric_only=True)).fillna(0)
    X_test  = X_test[final_features].fillna(X_train[final_features].median(numeric_only=True)).fillna(0)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Automatically tunes regularization strength using cross-validation
    print("\nTraining optimized Logistic Regression with automatic regularization tuning...")
    model = LogisticRegressionCV(
        Cs=50,  # Number of regularization strengths to try
        cv=5,   # 5-fold cross-validation
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        max_iter=5000,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        refit=True  # Refit final model with best C on full training data
    )
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)

    print(f"\nBEST REGULARIZATION (C): {model.C_[0]:.6f}")
    print(f"LOGISTIC REGRESSION FINAL ACCURACY: {acc:.4f} 🚀")

    # ====================== VISUALIZATIONS ======================
    # 1) Plot top coefficients for interpretability (core value of linear models)
    coefs = np.abs(model.coef_).mean(axis=0)
    top_idx = np.argsort(coefs)[-15:]
    top_features = np.array(final_features)[top_idx]
    top_coefs = coefs[top_idx]

    plt.figure()
    sns.barplot(x=top_coefs, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f"Optimized Logistic Regression — Top 15 Features\nTest Accuracy: {acc:.3f}")
    plt.xlabel("Average Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(fig_path / "logreg_optimized_top_features.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Classification report saved as CSV for easy inclusion in final report
    report_df = pd.DataFrame(classification_report(y_test, pred, target_names=le.classes_, output_dict=True)).T
    report_df.round(3).to_csv(fig_path / "logreg_optimized_report.csv")
    print("\nClassification Report:")
    print(report_df.round(3))

    # Save model
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "best_C": model.C_[0],
        "features": final_features
    }, "final_milestone/results/logistic_regression_optimized.pkl")

    print("\nOptimized Logistic Regression saved — ready for comparison!")

if __name__ == "__main__":
    main()