"""
train_random_forest.py
RANDOM FOREST CLASSIFIER - MILESTONE 3

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 3
Date: December 2025

This script trains a high-performance Random Forest model using the
theory-aligned feature set plus tactical variables and interaction terms.
It serves as the **non-linear benchmark** for Milestone 3, complementing
the interpretable Logistic Regression baseline.

Key Changes from Milestone 2:
    • Now loads from the consolidated `data_ready.pkl` (post all feature engineering)
    • Added the same interaction terms (supply_surprise, naval_surprise) used in Logistic Regression
    • Increased number of trees to 1200 for greater stability
    • Simplified workflow by using pre-engineered data instead of re-adding tactical features
    • Focus on Gini importance rankings to validate military theory (especially logistics)

Purpose:
    Demonstrate that a powerful ensemble model continues to highlight
    supply_cut and supply×surprise as top predictors, strongly supporting
    classical war theory (Clausewitz, van Creveld) in a non-linear context.
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Visualization Configuration
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 8)

# Output directory for final milestone deliberables
fig_path = Path("final_milestone/results/figures")
fig_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function for the Milestone 3 Random Forest pipeline."""
    print("Loading clean data (from master_fix.py)...")
    # Note: This script uses the final prepared dataset (data_ready.pkl)
    # which already contains all base features + tactical variables
    data = joblib.load("final_milestone/results/data_ready.pkl")

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le = data["le"]

    """
    ====================== INTERACTION TERMS (Consistent with LogReg) ======================
    Create interaction terms to capture combined theoretical effects
    This mirrors the approach used in the Logistic Regression model for fair comparison
    """
    surprise_cols = [c for c in X_train.columns if "surprise_level" in c]

    X_train = X_train.copy()    # Work on copies to avoid modifying original data
    X_test  = X_test.copy()

    X_train["surprise_sum"] = X_train[surprise_cols].sum(axis=1)
    X_test["surprise_sum"]  = X_test[surprise_cols].sum(axis=1)

    # Key interactions: tactical advantage x surprise
    X_train["supply_surprise"] = X_train["supply_cut"] * X_train["surprise_sum"]
    X_test["supply_surprise"]  = X_test["supply_cut"]  * X_test["surprise_sum"]
    X_train["naval_surprise"]  = X_train["naval_power"] * X_train["surprise_sum"]
    X_test["naval_surprise"]   = X_test["naval_power"]  * X_test["surprise_sum"]

    final_features = X_train.columns.tolist()
    print(f"Using {len(final_features)} features (including supply_surprise, naval_surprise)")

    # ====================== TRAIN RANDOM FOREST MODEL ======================
    # Optimized Random Forest with increased number of trees for stability
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"RANDOM FOREST ACCURACY: {acc:.4f}")

    # ====================== FEATURE IMPORTANCE ANALYSIS ======================
    # Gini importance is a key strength of Random Forest — helps validate theory
    importance = pd.Series(model.feature_importances_, index=final_features).sort_values(ascending=False)
    top20 = importance.head(20)

    plt.figure()
    sns.barplot(
        x=top20.values,
        y=top20.index,
        hue=top20.index,
        palette="Greens_d",
        legend=False
    )
    plt.title(f"Random Forest — Top 20 Most Important Features\nTest Accuracy: {acc:.3f}")
    plt.xlabel("Feature Importance (Gini)")
    plt.tight_layout()
    plt.savefig(fig_path / "random_forest_top20.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved: random_forest_top20.png")

    # === CLASSIFICATION REPORT ===
    # Save classification report as CSV for detailed performance analysis
    report = classification_report(y_test, pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    report_df.to_csv(fig_path / "random_forest_report.csv")
    print("\nClassification Report:")
    print(report_df)

    # Save model
    joblib.dump({
        "model": model,
        "accuracy": acc,
        "feature_importance": dict(importance)
    }, "final_milestone/results/random_forest.pkl")
    print("Model saved: random_forest.pkl")

if __name__ == "__main__":
    main()