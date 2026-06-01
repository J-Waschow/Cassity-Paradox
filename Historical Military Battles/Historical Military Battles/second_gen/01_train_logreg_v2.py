"""
01_train_logreg_theory.py
LOGISTIC REGRESSION WITH TACTICAL THEORY FEATURES - MILESTONE 2
Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025

This script trains an enhanced Logistic Regression model that incorporates
the new tactical factors introduced in Milestone 2. By focusing on a focused
set of theory-aligned features — particularly supply chain disruption and
naval power — we test whether classical military theory concepts can improve
predictive performance over the broader Milestone 1 baseline.

The model uses a carefully designed pipeline that handles missing values,
scales numeric features, applies SMOTE to address class imbalance, and
maintains strong interpretability through coefficient analysis.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipe
from pathlib import Path
from datetime import datetime

# Consistent visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ====================== PATH CONFIGURATION ======================
DATA_PATH = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\models\data_split_enhanced.pkl")
RESULTS_DIR = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\results")

# Unique timestamped folder for this run
RUN_DIR = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_logreg_theory_final"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOGREG] Starting – outputs → {RUN_DIR.name}")

# Load battle names for tactical factors
battles_df = pd.read_excel(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data\battles.xlsx")
full_names = battles_df["name"].fillna("Unknown").astype(str).tolist()


def add_tactical_factors(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
        Adds the two key tactical features from Milestone 2: supply_cut and naval_power.

        These features are derived from battle names using keyword patterns informed
        by military theory. Supply_cut captures logistics disruption and sieges,
        while naval_power identifies maritime and amphibious engagements.

        This function aligns the battle names from the original data with our
        train/test split and creates the new binary indicators.
        """
    start = len(X_train)    # Align battle names with the current train/test split
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["name"] = full_names[:start]
    X_test["name"]  = full_names[start:start + len(X_test)]

    # Supply disruption keywords (focused on sieges, encirclements, logistics)
    supply_pat = r"siege|blockade|fall of|investment|relief of|leningrad|stalingrad|port arthur|singapore|malta|tobruk|corregidor|bastogne|dien bien phu|encircled|cut off|pocket|cauldron|kessel|starvation|besieged|surrounded|fortress"

    # Naval and amphibious power keywords
    naval_pat  = r"naval|leyte|midway|java sea|coral sea|guadalcanal|matapan|jutland|trafalgar|river plate|denmark strait|bismarck|yamato|submarine|u-boat|convoy|gallipoli|dardanelles|tarawa|peleliu|okinawa|amphibious|landing|beach|normandy|inchon"

    for df in [X_train, X_test]:
        df["supply_cut"]  = df["name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(int)
        df["naval_power"] = df["name"].str.contains(naval_pat,  case=False, regex=True, na=False).astype(int)
        df.drop(columns=["name"], inplace=True, errors="ignore")
    return X_train, X_test


def main():
    """
        Runs the complete Logistic Regression modeling pipeline for Milestone 2.

        This function loads the baseline data, incorporates the new tactical
        features, selects a focused set of theory-aligned variables, builds
        a robust preprocessing and modeling pipeline, trains the model,
        evaluates performance, and generates rich diagnostic visualizations.
        """
    data = joblib.load(DATA_PATH)   # Load Milestone 1 preprocessed daata
    X_train = data["X_train"].copy()
    X_test  = data["X_test"].copy()
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le     = data["label_encoder"]

    # Load battle names for tactical feature creation
    X_train, X_test = add_tactical_factors(X_train, X_test)

    # 20 theory-aligned features
    theory_features = [
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

    # Keep available features
    cols = [c for c in theory_features if c in X_train.columns]
    X_train = X_train[cols]
    X_test  = X_test[cols]
    print(f"[LOGREG] Using {len(cols)} features (NaNs will be imputed)")

    # Identify numeric columns that need scaling + imputation
    numeric_cols = ["duration_days", "force_ratio"]

    # Build preprocessing pipeline (impute → scale numeric → pass-through others)
    preprocessor = ColumnTransformer([
        ("impute_scale", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ]), [c for c in numeric_cols if c in cols]),
        ("impute_binary", SimpleImputer(strategy="most_frequent"),
         [c for c in cols if c not in numeric_cols])
    ], remainder="passthrough")

    # Full modeling pipeline with SMOTE for class imbalance
    pipe = ImbPipe([
        ("prep",   preprocessor),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    LogisticRegression(multi_class="ovr", class_weight="balanced", max_iter=1000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[LOGREG] Accuracy: {acc:.3f}")

    # ====================== VISUALIZATIONS ======================
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                              xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.savefig(RUN_DIR / "01_confusion.png", dpi=300, bbox_inches="tight"); plt.close()

    # 2. Coefficient Importance (absolute mean across classes)
    coef = np.abs(pipe.named_steps["clf"].coef_).mean(axis=0)
    coef_df = pd.DataFrame({"feature": pipe.named_steps["prep"].get_feature_names_out(), "coef": coef})
    top_coef = coef_df.sort_values("coef", ascending=False).head(15)
    plt.figure(); sns.barplot(data=top_coef, x="coef", y="feature", palette="viridis")
    plt.title("Top 15 Features – Coefficient Magnitude")
    plt.savefig(RUN_DIR / "02_coefficients.png", dpi=300, bbox_inches="tight"); plt.close()

    # 3. ROC Curves (One-vs-Rest)
    y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    plt.figure()
    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.savefig(RUN_DIR / "03_roc.png", dpi=300, bbox_inches="tight"); plt.close()

    # 4. Probability Distribution
    prob_df = pd.DataFrame(y_prob, columns=le.classes_)
    prob_df["true"] = le.inverse_transform(y_test)
    melted = prob_df.melt(id_vars="true", var_name="predicted", value_name="probability")
    plt.figure(); sns.boxplot(data=melted, x="true", y="probability", hue="predicted")
    plt.title("Prediction Probabilities by True Class")
    plt.savefig(RUN_DIR / "04_prob_dist.png", dpi=300, bbox_inches="tight"); plt.close()

    # Save model
    joblib.dump({
        "model": pipe, "features": cols, "label_encoder": le,
        "accuracy": acc, "y_test": y_test, "y_pred": y_pred
    }, RUN_DIR / "model.pkl")

    print(f"[LOGREG] All done! → {RUN_DIR}")


if __name__ == "__main__":
    main()