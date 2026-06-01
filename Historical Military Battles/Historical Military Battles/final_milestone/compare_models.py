"""
compare_models.py
FINAL MODEL COMPARISON - MILESTONE 3

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 3
Date: December 2025

This script provides the culminating analysis for Milestone 3 by comparing
the performance of the three main models developed throughout the project:

• Logistic Regression (interpretable baseline)
• Random Forest (strong non-linear ensemble)
• XGBoost (highest-accuracy gradient boosting model)

It generates a summary table, a performance bar chart, and export files
(CSV + Markdown) for easy inclusion in the final project report.

Purpose:
    To objectively evaluate which modeling approach performed best and
    to visually demonstrate the value of theory-driven features (especially
    supply_cut and supply_surprise) across different algorithms. This
    comparison serves as the capstone of the entire project.
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Visualization configuration
sns.set_style("whitegrid")

# Output directory for final milestone deliverables
fig_path = Path("final_milestone/results/figures")
fig_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function that loads all trained models and generates comparison outputs."""
    # Load the saved models from their respective training scripts
    logreg = joblib.load("final_milestone/results/logistic_regression.pkl")
    rf     = joblib.load("final_milestone/results/random_forest.pkl")
    xgb    = joblib.load("final_milestone/results/xgboost.pkl")

    # Compile accuracy results from all three models into a comparison table
    results = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": round(logreg["accuracy"], 4)},
        {"Model": "Random Forest",       "Accuracy": round(rf["accuracy"], 4)},
        {"Model": "XGBoost",             "Accuracy": round(xgb["accuracy"], 4)}
    ]).sort_values("Accuracy", ascending=False)

    print("\nFINAL MODEL COMPARISON")
    print(results.to_string(index=False))

    # ====================== VISUALIZATION ======================
    # Bar chart comparing test accuracy across models
    plt.figure(figsize=(8,6))
    sns.barplot(data=results, x="Model", y="Accuracy", palette="viridis")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.ylabel("Test Accuracy")

    # Add accuracy labels on top of each bar
    for i, row in results.iterrows():
        plt.text(i, row.Accuracy + 0.01, f"{row.Accuracy:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_path / "model_comparison_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ====================== EXPORT RESULTS ======================
    # Save results in multiple formats for easy use in reports
    results.to_csv("final_milestone/results/model_comparison.csv", index=False)
    results.to_markdown("final_milestone/results/model_comparison.md", index=False)
    print("Comparison table + plot saved")

if __name__ == "__main__":
    main()