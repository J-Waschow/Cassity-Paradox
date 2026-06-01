"""
second_gen/00_load_split_v2.py
PREPROCESSING V2 - TACTICAL FACTORS & WAR THEORY INTEGRATION
MILESTONE 2
Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025

This script builds upon the original preprocessing pipeline by introducing
two high-impact tactical features derived from military theory: supply chain
disruption and naval/amphibious power projection.

These features are extracted from battle names using comprehensive keyword
lists informed by historical war theory (Clausewitzian logistics emphasis
and Mahanian sea power concepts). This represents the core advancement
in Milestone 2: "Logistics & Sea Power in Historical Warfare."

Purpose of the script:
    • Loads the baseline preprocessed data from Milestone 1
    • Reconstructs a unified dataset (train + test)
    • Engineers new theory-driven features:
        - supply_cut: Identifies battles involving sieges, encirclements,
          and logistics disruption
        - naval_power: Identifies naval, amphibious, and maritime battles
    • Adds supporting engineered features (surprise_binary, log_duration)
    • Saves the enhanced dataset for use in Milestone 2 modeling scripts

This version allows us to test whether incorporating classical military
theory improves predictive performance over the Milestone 1 baselines.
"""

import joblib, pandas as pd, numpy as np
from pathlib import Path


def main():
    """
        Orchestrates the creation of the enhanced v2 dataset with tactical features.

        This central function loads the Milestone 1 data split, combines train
        and test sets for feature engineering, adds theory-informed tactical
        variables, and exports the result as a CSV for easier inspection and
        use in subsequent modeling scripts.
        """
    pkl_path = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\models\data_split_enhanced.pkl")
    out_csv = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data\preprocessed_v2.csv")

    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} not found!")
        return

    data = joblib.load(pkl_path)    # Reconstruct full dataset from train/test split
    X_train = data["X_train"].copy()
    X_test = data["X_test"].copy()
    y_train = data["y_train"]
    y_test = data["y_test"]

    df = pd.concat([X_train, X_test])
    df["target"] = pd.concat([y_train, y_test])

    # ====================== BASIC ENGINEERED FEATURES ======================
    # Binary surprise indicator (strong surprise vs none/minor)
    df["surprise_binary"] = (df["surpa"].abs() > 1).astype(int)
    df["log_duration"] = np.log1p(df["duration_days"])

    # ====================== TACTICAL FEATURES FROM WAR THEORY ======================
    # === SUPPLY CHAIN DISRUPTION ===
    # Keywords drawn from historical patterns of sieges, encirclements,
    # and logistical isolation — core to modern understanding of sustainment
    # as a decisive factor in warfare.
    supply_keywords = (
        "siege|blockade|cut off|rail|bridge|encircled|starved|isolated|pocket|investment|"
        "surrounded|cauldron|trapped|strangled|choked|starvation|supply|logistics|"
        "besieged|invested|cordoned|ring|kessel|fortress|pocketed|bottled"
    )
    df["supply_cut"] = df["name"].str.contains(supply_keywords, case=False, na=False).astype(int)

    # === NAVAL & AMPHIBIOUS POWER PROJECTION ===
    # Keywords capturing sea power, fleet actions, and amphibious operations —
    # reflecting Mahanian principles of maritime dominance and control of
    # lines of communication.
    naval_keywords = (
        "navy|ship|submarine|blockade|landing|amphibious|fleet|convoy|port|harbor|"
        "black sea|mediterranean|baltic|atlantic|pacific|naval|marine|gunboat|"
        "torpedo|destroyer|cruiser|battleship|carrier|uboat|u-boat|dreadnought|"
        "gallipoli|tarawa|normandy|leyte|midway|trafalgar|jutland|river|lake"
    )
    df["naval_power"] = df["name"].str.contains(naval_keywords, case=False, na=False).astype(int)
    df.to_csv(out_csv, index=False)

    # Validation and Summary
    print(f"✓ Preprocessed v2 + TACTICAL FACTORS → {out_csv.name}")
    print(f"   supply_cut    = {df['supply_cut'].sum()} battles ({df['supply_cut'].mean():.1%})")
    print(f"   naval_power   = {df['naval_power'].sum()} battles ({df['naval_power'].mean():.1%})")
    print(f"   Win rate when supply_cut  = 1: {df[df['supply_cut'] == 1]['target'].mean():.1%}")
    print(f"   Win rate when naval_power = 1: {df[df['naval_power'] == 1]['target'].mean():.1%}")


if __name__ == "__main__":
    main()