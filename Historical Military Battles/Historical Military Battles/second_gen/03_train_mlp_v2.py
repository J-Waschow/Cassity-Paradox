"""
03_train_mlp_v2.py
MULTI-LAYER PERCEPTRON WITH TACTICAL THEORY FEATURES - MILESTONE 2
Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025

This script trains an optimized Multi-Layer Perceptron (MLP) neural network
using the enhanced feature set from Milestone 2. Building on the tactical
factors (supply_cut and naval_power), it also introduces interaction terms
that capture combined effects such as the synergy between logistics
disruption and surprise, or naval power and air superiority.

The deeper architecture and careful preprocessing allow the model to learn
complex non-linear patterns in the historical battle data, aiming for
improved performance over the Milestone 1 baselines.
"""
import joblib
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path
from datetime import datetime

# Consistent visualization style
sns.set_style("whitegrid")

# ====================== PATH CONFIGURATION ======================
DATA_PATH = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\models\data_split_enhanced.pkl")
RESULTS_DIR = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\results")

# Create a unique timestamped results folder for this run
RUN_DIR = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_mlp_high_accuracy"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Load battle names & tactical factors
battles_df = pd.read_excel(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data\battles.xlsx")
full_names = battles_df["name"].fillna("Unknown").astype(str).tolist()

def add_tactical_factors(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
        Adds the two key tactical features (supply_cut and naval_power)
        while carefully preserving the original data structure.

        This function handles potential string columns (e.g., from one-hot
        encoding) and safely adds the new binary indicators derived from
        battle names using comprehensive keyword patterns grounded in
        military theory.
        """
    train_names = full_names[:len(X_train)]
    test_names = full_names[len(X_train):]

    X_train = X_train.copy()    # Work on copies to avoid modifying original data
    X_test = X_test.copy()
    X_train["battle_name"] = train_names    # Align battle names with train/test split
    X_test["battle_name"] = test_names

    # Identify and preserve string columns
    string_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    if "battle_name" in string_cols:
        string_cols.remove("battle_name")

    train_strings = X_train[string_cols].copy() if string_cols else None
    test_strings = X_test[string_cols].copy() if string_cols else None

    # Drop string columns temporarily for tactical feature creation
    X_train_no_strings = X_train.drop(columns=(string_cols + ["battle_name"]) if string_cols else ["battle_name"])
    X_test_no_strings = X_test.drop(columns=(string_cols + ["battle_name"]) if string_cols else ["battle_name"])

    # Supply disruption pattern (sieges, encirclements, logistics isolation)
    supply_pat = (
        r"siege|blockade|fall of|investment of|relief of|leningrad|stalingrad|port arthur|singapore|malta|tob|tobruk|"
        r"corregidor|bastogne|dien bien phu|encircled|cut off|isolated|pocket|cauldron|kessel|starvation|besieged|"
        r"invested|surrounded|fortress"
    )
    # Naval and amphibious power pattern
    naval_pat = (
        r"naval|battle of the|leyte|midway|atlantic|java sea|coral sea|guadalcanal|solomons|savo island|matapan|"
        r"jutland|trafalgar|river plate|denmark strait|bismarck|scharnhorst|yamato|musashi|submarine|u-boat|uboat|"
        r"convoy|merchant|wolfpack|gallipoli|dardanelles|tarawa|peleliu|okinawa|amphibious|landing|beach|normandy|inchon"
    )

    # Create tactical features
    X_train["supply_cut"] = (
        X_train["battle_name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(int)
    )
    X_test["supply_cut"] = (
        X_test["battle_name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(int)
    )
    X_train["naval_power"] = (
        X_train["battle_name"].str.contains(naval_pat, case=False, regex=True, na=False).astype(int)
    )
    X_test["naval_power"] = (
        X_test["battle_name"].str.contains(naval_pat, case=False, regex=True, na=False).astype(int)
    )

    # Reassemble dataframes, preserving original string columns if present
    if train_strings is not None:
        X_train = pd.concat([X_train_no_strings, train_strings, X_train[["supply_cut", "naval_power"]]], axis=1)
        X_test = pd.concat([X_test_no_strings, test_strings, X_test[["supply_cut", "naval_power"]]], axis=1)
    else:
        X_train = pd.concat([X_train_no_strings, X_train[["supply_cut", "naval_power"]]], axis=1)
        X_test = pd.concat([X_test_no_strings, X_test[["supply_cut", "naval_power"]]], axis=1)

    total_supply = int(X_train["supply_cut"].sum() + X_test["supply_cut"].sum())
    total_naval = int(X_train["naval_power"].sum() + X_test["naval_power"].sum())

    print("[tactical_factors] ADDED")
    print(f"   supply_cut    = {total_supply} battles")
    print(f"   naval_power   = {total_naval} battles")

    return X_train, X_test

def main():
    """
        Runs the complete optimized MLP modeling pipeline for Milestone 2.

        This function loads the baseline data, adds tactical features, creates
        meaningful interaction terms, scales the data, trains a deep neural
        network with early stopping, and saves the final high-performing model.
        """
    data = joblib.load(DATA_PATH)   # Load Milestone 1 preprocessed data
    X_train = data["X_train"].copy()
    X_test = data["X_test"].copy()
    y_train = data["y_train"]
    y_test = data["y_test"]
    le = data["label_encoder"]

    # Add theory-driven tactical factors
    X_train, X_test = add_tactical_factors(X_train, X_test)

    # Narrow to 20 theory features
    features = [
        "duration_days", "multi_phase", "force_ratio", "supply_cut", "naval_power",
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
    cols = [c for c in features if c in X_train.columns]
    X_train = X_train[cols].copy()
    X_test = X_test[cols].copy()

    # Convert any remaining object columns to numeric
    for df in (X_train, X_test):
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = pd.to_numeric(df[c], errors="ignore")

    # ====================== INTERACTION TERMS ======================
    # These capture combined effects predicted by military theory
    train_surprise_sum = X_train.filter(like="surprise_level").sum(axis=1)
    test_surprise_sum = X_test.filter(like="surprise_level").sum(axis=1)

    # Safe getter for potential missing air superiority column
    train_air_sup_att = X_train.get("air_superiority_Attacker had air superiority in the theater", 0)
    test_air_sup_att = X_test.get("air_superiority_Attacker had air superiority in the theater", 0)

    # Safe getter for supply/naval columns (should exist, but guard anyway)
    train_supply = X_train.get("supply_cut", 0)
    test_supply = X_test.get("supply_cut", 0)
    train_naval = X_train.get("naval_power", 0)
    test_naval = X_test.get("naval_power", 0)

    X_train["supply_surprise"] = train_supply * train_surprise_sum
    X_train["naval_surprise"] = train_naval * train_surprise_sum
    X_train["supply_air"] = train_supply * train_air_sup_att

    X_test["supply_surprise"] = test_supply * test_surprise_sum
    X_test["naval_surprise"] = test_naval * test_surprise_sum
    X_test["supply_air"] = test_supply * test_air_sup_att

    # Scale everything
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train optimized MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),  # Deeper architecture
        activation='relu',
        solver='adam',
        alpha=0.001,    # Regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,    # Prevent overfitting
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"[MLP OPTIMIZED] Accuracy: {acc:.3f} ← Target 70–74%")

    # Save Model
    joblib.dump({
        "model": mlp, "scaler": scaler, "features": X_train.columns.tolist(),
        "le": le, "accuracy": acc
    }, RUN_DIR / "model.pkl")

    print(f"High-accuracy MLP saved → {RUN_DIR}")

if __name__ == "__main__":
    main()