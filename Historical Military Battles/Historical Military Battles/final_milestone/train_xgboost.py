"""
train_xgboost.py
XGBoost GRADIENT BOOSTING MODEL - MILESTONE 3 (FINAL HIGH-PERFORMANCE MODEL)

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 3
Date: December 2025

This script trains an XGBoost classifier as the **state-of-the-art, highest-accuracy model**
for the Historical Military Battles prediction project in Milestone 3.

Switch from MLP to XGBoost in the final run:
    • Milestone 2 focused on interpretability (Logistic Regression) and strong baselines (Random Forest).
    • In Milestone 3, we needed the best possible predictive performance to validate whether
      theory-driven features (especially supply_cut and supply×surprise) could achieve top-tier results.
    • XGBoost was chosen because it excels at capturing complex non-linear interactions, handles
      imbalanced classes well, provides robust feature importance (via 'gain'), and is widely
      regarded as one of the strongest tree-based algorithms for structured tabular data.

Purpose:
    Deliver the highest accuracy achievable on this dataset while confirming that
    logistics disruption (supply_cut) and its interaction with surprise remain the
    most dominant predictors — providing strong empirical support for classical
    military theory in a modern machine learning framework.
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Visualization configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 7)

# Output directory for final milestone deliverables
fig_path = Path("final_milestone/results/figures")
fig_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function for the Milestone 3 XGBoost pipeline."""
    print("Loading clean data...")

    # Using the fully prepared data from the final preprocessing step
    data = joblib.load("final_milestone/results/data_ready.pkl")
    X_train = data["X_train"].copy()
    X_test  = data["X_test"].copy()
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le = data["le"]

    """
    ====================== INTERACTION TERMS ======================
    Add the same interaction terms used in Logistic Regression and Random Forest
    for fair model comparison across all final models
    """
    surprise_cols = [c for c in X_train.columns if "surprise_level" in c]
    X_train["surprise_sum"] = X_train[surprise_cols].sum(axis=1)
    X_test["surprise_sum"]  = X_test[surprise_cols].sum(axis=1)
    X_train["supply_surprise"] = X_train["supply_cut"] * X_train["surprise_sum"]
    X_test["supply_surprise"]  = X_test["supply_cut"]  * X_test["surprise_sum"]
    X_train["naval_surprise"]  = X_train["naval_power"] * X_train["surprise_sum"]
    X_test["naval_surprise"]   = X_test["naval_power"]  * X_test["surprise_sum"]

    final_features = X_train.columns.tolist()
    print(f"XGBoost using {len(final_features)} features")

    # ====================== TRAIN XGBOOST MODEL ======================
    # XGBoost configuration optimized for multiclass classification
    # This represents the highest-performing model in the final milestone
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='multi:softmax',   
        num_class=3,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # Use predict() to get class labels
    pred = model.predict(X_test)  # ← Returns class labels: 0, 1, 2
    acc = accuracy_score(y_test, pred)
    print(f"\nXGBoost — Final Test Accuracy: {acc:.4f}")

    # ====================== FEATURE IMPORTANCE ======================
    # XGBoost provides powerful 'gain' importance, helping validate military theory
    plt.figure()
    xgb.plot_importance(model, max_num_features=15, importance_type='gain', show_values=False)
    plt.title(f"XGBoost — Top 15 Features\nAccuracy: {acc:.3f}")
    plt.tight_layout()
    plt.savefig(fig_path / "xgboost_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === CLASSIFICATION REPORT ===
    # Save classification report as CSV for detailed performance analysis
    report = classification_report(y_test, pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    report_df.to_csv(fig_path / "xgboost_report.csv")
    print("\nClassification Report:")
    print(report_df)

    # Save model
    joblib.dump({"model": model, "accuracy": acc},
                "final_milestone/results/xgboost.pkl")
    print("\nXGBoost completed and saved!")

if __name__ == "__main__":
    main()
