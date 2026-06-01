"""
tactical_factors.py
TACTICAL FACTORS MODULE - MILESTONE 2

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025

This reusable module adds two high-impact, theory-driven tactical features
to the modeling pipeline:

• **supply_cut**: Identifies battles involving significant logistics disruption,
  sieges, encirclements, blockades, or supply starvation.
• **naval_power**: Identifies battles with naval, submarine, fleet, or
  amphibious operations.

These features are **hard-coded** using comprehensive keyword patterns derived
from historical battle names. This approach allows us to directly inject
classical military theory (Clausewitzian emphasis on sustainment and Mahanian
principles of sea power) into the dataset without relying on external models.

Purpose:
    To provide a clean, reusable way to enhance any train/test split with
    tactically meaningful features that reflect real military dynamics.
    This module is used across all Milestone 2 modeling scripts.

Usage Example:
    ```python
    from tactical_factors import add_tactical_factors
    X_train, X_test = add_tactical_factors(X_train, X_test)
"""

import pandas as pd
from pathlib import Path

# ====================== PATH CONFIGURATION ======================
BATTLES_PATH = Path(r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data\battles.xlsx")

if not BATTLES_PATH.exists():
    raise FileNotFoundError(f"battles.xlsx not found: {BATTLES_PATH}")

print(f"[tactical_factors] Loading battle names from {BATTLES_PATH}")
# Load battle names once at module level (pandas DataFrame)
battles_df = pd.read_excel(BATTLES_PATH)
# full_names to use Python list of battle name strings
full_names = battles_df["name"].fillna("Unknown Battle").astype(str).tolist()


def add_tactical_factors(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
        Adds hard-coded tactical features (supply_cut and naval_power) to
        the training and test datasets.

        This is the main function of the module. It performs careful data
        handling to preserve original structure while injecting theory-based
        features derived from battle names.

        Parameters:
            X_train (pd.DataFrame): Training feature set from Milestone 1
            X_test (pd.DataFrame): Test feature set from Milestone 1

        Returns:
            tuple: (X_train enhanced, X_test enhanced)
        """
    train_names = full_names[:len(X_train)]
    test_names = full_names[len(X_train):]

    X_train = X_train.copy()    # Copies to avoid modifying original DataFrames in place
    X_test = X_test.copy()
    X_train["battle_name"] = train_names
    X_test["battle_name"] = test_names

    # Identify and preserve any existing string/object columns
    # (e.g., one-hot encoded categories that might still be strings)
    string_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    if "battle_name" in string_cols:
        string_cols.remove("battle_name")

    # Align original battle names with the current train/test split
    train_strings = X_train[string_cols].copy() if string_cols else None
    test_strings = X_test[string_cols].copy() if string_cols else None

    # Drop string columns temporarily for tactical features creations
    X_train_no_strings = X_train.drop(columns=string_cols + ["battle_name"])
    X_test_no_strings = X_test.drop(columns=string_cols + ["battle_name"])

    """
    ====================== HARD-CODED TACTICAL PATTERNS ======================
    These regex patterns are intentionally hard-coded to ensure 
    consistency and direct linkage to military theory concepts.
    """
    # Supply disruption pattern - reflects logistical isolation and sieges
    supply_pat = r"siege|blockade|fall of|investment of|relief of|leningrad|stalingrad|port arthur|singapore|malta|tob|tobruk|corregidor|bastogne|dien bien phu|encircled|cut off|isolated|pocket|cauldron|kessel|starvation|besieged|invested|surrounded|fortress"

    # Naval/amphibious power pattern - reflects sea power and maritime operations
    naval_pat = r"naval|battle of the|leyte|midway|atlantic|java sea|coral sea|guadalcanal|solomons|savo island|matapan|jutland|trafalgar|river plate|denmark strait|bismarck|scharnhorst|yamato|musashi|submarine|u-boat|uboat|convoy|merchant|wolfpack|gallipoli|dardanelles|tarawa|peleliu|okinawa|amphibious|landing|beach|normandy|inchon"

    X_train["supply_cut"] = X_train["battle_name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(
        int)
    X_test["supply_cut"] = X_test["battle_name"].str.contains(supply_pat, case=False, regex=True, na=False).astype(int)
    X_train["naval_power"] = X_train["battle_name"].str.contains(naval_pat, case=False, regex=True, na=False).astype(
        int)
    X_test["naval_power"] = X_test["battle_name"].str.contains(naval_pat, case=False, regex=True, na=False).astype(int)

    # Reassemble the DataFrames, restoring any original string columns
    if train_strings is not None:
        X_train = pd.concat([X_train_no_strings, train_strings, X_train[["supply_cut", "naval_power"]]], axis=1)
        X_test = pd.concat([X_test_no_strings, test_strings, X_test[["supply_cut", "naval_power"]]], axis=1)
    else:
        X_train = pd.concat([X_train_no_strings, X_train[["supply_cut", "naval_power"]]], axis=1)
        X_test = pd.concat([X_test_no_strings, X_test[["supply_cut", "naval_power"]]], axis=1)

    # Summary output
    total_supply = X_train["supply_cut"].sum() + X_test["supply_cut"].sum()
    total_naval = X_train["naval_power"].sum() + X_test["naval_power"].sum()

    print(f"[tactical_factors] ADDED")
    print(f"   supply_cut    = {total_supply} battles")
    print(f"   naval_power   = {total_naval} battles")

    return X_train, X_test