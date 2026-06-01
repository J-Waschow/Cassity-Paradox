"""
02_train_rf_v2.py
RANDOM FOREST CLASSIFIER WITH TACTICAL THEORY FEATURES - MILESTONE 2
Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025
Author: Jordan Waschow

This script trains a Random Forest model using the narrowed, theory-aligned
feature set developed in Milestone 2. By focusing on high-priority variables —
especially the new tactical features `supply_cut` and `naval_power` — we evaluate
how well an ensemble tree-based approach can leverage classical military theory
concepts (logistics disruption and sea power) to predict battle outcomes.

The model produces rich diagnostic visualizations and dual feature importance
analyses (Gini and Permutation), helping us understand which factors are most
influential in this enhanced Milestone 2 framework.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
from pathlib import Path
from datetime import datetime


# Consistent visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ====================== PATH CONFIGURATION ======================
DATA_PATH = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\models\data_split_enhanced.pkl")
RESULTS_DIR = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\results")

# Create a unique timestamped results folder for this run
RUN_DIR = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_rf_v2"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"[RF] Starting Random Forest training → {RUN_DIR.name}")

# Load battle names for tactical factors
battles_df = pd.read_excel(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data\battles.xlsx")
full_names = battles_df["name"].fillna("Unknown").astype(str).tolist()


def add_tactical_factors(X_train, X_test):
    """
        Adds the two key tactical features from Milestone 2 using battle names.

        `supply_cut` identifies battles involving sieges, encirclements, and
        logistics isolation. `naval_power` captures maritime, fleet, and
        amphibious operations. These features are derived from keyword patterns
        grounded in military theory.
        """
    start = len(X_train)    # Align battle names with the current train/test split
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["name"] = full_names[:start]
    X_test["name"] = full_names[start:start + len(X_test)]

    # Supply disruption
    supply_pat = r"siege|blockade|fall of|investment|relief of|leningrad|stalingrad|port arthur|singapore|malta|tobruk|corregidor|bastogne|dien bien phu|encircled|cut off|pocket|cauldron|kessel|starvation|besieged|surrounded|fortress"

    # Naval and amphibious operations pattern
    naval_pat = r"naval|leyte|midway|java sea|coral sea|guadalcanal|matapan|jutland|trafalgar|river plate|denmark strait|bismarck|yamato|submarine|u-boat|convoy|gallipoli|dardanelles|tarawa|peleliu|okinawa|amphibious|landing|beach|normandy|inchon"

    for df in [X_train, X_test]:
        df["supply_cut"] = df["name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(int)
        df["naval_power"] = df["name"].str.contains(naval_pat, case=False, regex=True, na=False).astype(int)
        df.drop(columns=["name"], inplace=True, errors="ignore")
    return X_train, X_test


def main():
    """
        Runs the complete Random Forest modeling pipeline for Milestone 2.

        This function loads the baseline data, incorporates the new tactical
        features, selects a focused set of theory-aligned variables, trains
        a powerful ensemble model, evaluates performance, and generates
        multiple visualizations to support analysis and presentation.
        """
    data = joblib.load(DATA_PATH)   # Load Milestone 1 preprocessed data
    X_train = data["X_train"].copy()
    X_test = data["X_test"].copy()
    y_train = data["y_train"]
    y_test = data["y_test"]
    le = data["label_encoder"]

    # Add theory-driven tactical factors
    X_train, X_test = add_tactical_factors(X_train, X_test)

    # Narrow to 20 theory-aligned features
    features = [
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
    available = [f for f in features if f in X_train.columns]
    X_train = X_train[available]
    X_test = X_test[available]
    print(f"[RF] Using {len(available)} theory-aligned features")

    # ====================== TRAIN RANDOM FOREST ======================
    rf = RandomForestClassifier(
        n_estimators=1000,  # Large number of trees for stability
        class_weight="balanced",    # Handle class imbalance
        random_state=42,
        n_jobs=-1,             # Use all CPU cores
        max_depth=None,
        min_samples_split=2
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RF] Accuracy: {acc:.3f}")

    # ====================== VISUALIZATIONS ======================
    # 1. Predicted Class Distribution
    pred_counts = pd.Series(y_pred).value_counts().reindex(range(len(le.classes_)), fill_value=0)
    plt.figure()
    sns.barplot(x=le.classes_, y=pred_counts.values, palette="Greens_d")
    plt.title("Random Forest – Predicted Class Distribution")
    plt.xlabel("Predicted Class"); plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "01_predicted_hist.png", dpi=300, bbox_inches="tight"); plt.close()

    # 2) Per-class recall
    cm = confusion_matrix(y_test, y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_recall = np.where(cm.sum(axis=1) > 0, np.diag(cm) / cm.sum(axis=1), 0.0)

    plt.figure()
    sns.barplot(x=le.classes_, y=per_class_recall, palette="Blues_d")
    plt.title("Random Forest – Per-class Recall (True Positive Rate)")
    plt.xlabel("True Class"); plt.ylabel("Recall")
    plt.ylim(0, 1)
    for i, v in enumerate(per_class_recall):
        plt.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "01_per_class_recall.png", dpi=300, bbox_inches="tight"); plt.close()

    # 3) Gini Importance
    gini = pd.DataFrame({"feature": available, "importance": rf.feature_importances_}).sort_values("importance", ascending=False).head(15)
    plt.figure()
    sns.barplot(data=gini, x="importance", y="feature", palette="Greens_d")
    plt.title("Top 15 Features – Gini Importance")
    plt.savefig(RUN_DIR / "02_gini_importance.png", dpi=300, bbox_inches="tight"); plt.close()

    # 4) Permutation Importance
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": available, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(15)
    plt.figure()
    sns.barplot(data=perm_df, x="importance", y="feature", palette="Blues_d")
    plt.title("Top 15 Features – Permutation Importance")
    plt.savefig(RUN_DIR / "03_permutation_importance.png", dpi=300, bbox_inches="tight"); plt.close()

    # 5) ROC Curves
    y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    plt.figure()
    for i in range(len(le.classes_)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.title("Random Forest – ROC Curves"); plt.legend()
    plt.savefig(RUN_DIR / "04_roc.png", dpi=300, bbox_inches="tight"); plt.close()

    # Save model
    joblib.dump({"model": rf, "features": available, "le": le, "accuracy": acc}, RUN_DIR / "model.pkl")
    print(f"[RF] All outputs saved → {RUN_DIR}")



if __name__ == "__main__":
    main()