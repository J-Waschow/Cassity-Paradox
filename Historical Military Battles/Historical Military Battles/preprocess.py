"""
preprocess.py
ROBUST DATA PREPROCESSING PIPELINE - MILESTONE 1

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 1 (First Run - Baseline Models)
Date: November 2025

This script loads, merges, cleans, and engineers features from multiple Excel files
containing historical military battle data. It prepares a clean feature matrix and
target variable for modeling.
Milestone: 1 (Baseline - Pre-tactical features)

Key Steps:
    1. Load only existing Excel files
    2. Merge core battle information + target (direction)
    3. Engineer terrain (one-hot), multi-phase flag, etc.
    4. Handle categorical mappings from enum tables
    5. One-hot encode belligerents and top commanders
    6. Clean missing values and create train/test split
    7. Save data for modeling scripts

Output:
    - models/data_split_enhanced.pkl
    - models/feature_list.csv
"""

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------- PATHS -------------------------------
DATA_DIR = r"C:\Users\jorda\PycharmProjects\Historical Military Battles\data"
MODEL_DIR = r"C:\Users\jorda\PycharmProjects\Historical Military Battles\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------- FILE MAP -------------------------------
FILES = {
    # Core tables
    "battles": "battles.xlsx",
    "dyads": "battle_dyads.xlsx",
    "durations": "battle_durations.xlsx",
    "terrain": "terrain.xlsx",
    "active_periods": "active_periods.xlsx",

    # Enum lookup tables
    "strength_engaged": "strength_engaged.xlsx",
    "element_of_surprise": "element of surprise.xlsx",
    "battle_result": "battle_result.xlsx",
    "air_superiority": "air superiority.xlsx",
    "action_initiated": "action_initiated.xlsx",

    # Additional small tables
    "belligerents": "belligerents.xlsx",
    "clear_cut": "clearCut.xlsx",
    "commanders": "commanders.xlsx",
    "favors_side": "favors_side.xlsx"
}

# ------------------------------- LOAD DATA -------------------------------
print("Loading files from:", DATA_DIR)
dfs = {}
for key, fname in FILES.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"WARNING: {fname} NOT FOUND – skipping")
        continue
    dfs[key] = pd.read_excel(path, sheet_name=0)
    print(f"  {fname}: {dfs[key].shape}")

# ------------------------------- CORE MERGE -------------------------------
if "battles" not in dfs:
    raise FileNotFoundError("battles.xlsx is required – aborting.")
df = dfs["battles"][['isqno', 'terra', 'surpa']].copy()

# ---- TARGET: direction ----
if "dyads" in dfs:
    primary_mask = dfs["dyads"].get("primary", pd.Series([1] * len(dfs["dyads"]))) == 1
    dyads_primary = dfs["dyads"][primary_mask]
    df = df.merge(dyads_primary[["isqno", "direction"]], on="isqno", how="left")
    print("Merged target (direction)")
else:
    print("WARNING: battle_dyads.xlsx missing – dummy target")
    df["direction"] = 0

# Duration feature
if "durations" in dfs:
    df = df.merge(dfs["durations"][["isqno", "duration1"]], on="isqno", how="left")
    df.rename(columns={"duration1": "duration_days"}, inplace=True)
    print("Merged duration")
else:
    df["duration_days"] = 1.0

# ------------------------------- FEATURE ENGINEERING -------------------------------
# Terrain: Multi-label one-hot encoding
if "terrain" in dfs:
    terrain = dfs["terrain"].copy()
    terra_cols = [c for c in ["terra1", "terra2", "terra3"] if c in terrain.columns]
    terrain_long = terrain.melt(
        id_vars=["isqno"],
        value_vars=terra_cols,
        value_name="terra_code"
    ).dropna(subset=["terra_code"])
    terrain_long["terra_code"] = terrain_long["terra_code"].str.strip()

    terrain_onehot = pd.get_dummies(terrain_long, columns=["terra_code"], prefix="terra")
    terrain_onehot = terrain_onehot.groupby("isqno").max().reset_index()  # max = presence
    df = df.merge(terrain_onehot, on="isqno", how="left")
    print(f"Added {len(terrain_onehot.columns) - 1} terrain features")
else:
    print("WARNING: terrain.xlsx missing")

# Multi-phase battles flag
if "active_periods" in dfs:
    phase_counts = dfs["active_periods"].groupby("isqno")["atp_number"].nunique().reset_index()
    phase_counts["multi_phase"] = (phase_counts["atp_number"] > 1).astype(int)
    df = df.merge(phase_counts[["isqno", "multi_phase"]], on="isqno", how="left")
    print("Added multi_phase flag")
else:
    df["multi_phase"] = 0


def safe_map(col, enum_key, default="Unknown"):
    """
    Safely map coded values to human-readable descriptions using enum lookup tables.

    Args:
        df: Main dataframe to modify in place
        col: Column name containing codes
        enum_key: Key in the dfs dictionary for the enum table
        default: Value to use when mapping fails
    """
    if enum_key in dfs and col in df.columns:
        mapping = dict(zip(dfs[enum_key]["value"], dfs[enum_key]["description"]))
        df[f"{col}_desc"] = df[col].map(mapping).fillna(default)
    else:
        df[f"{col}_desc"] = default


# ====================== CATEGORICAL MAPPINGS ======================
# Tactical posture
if "postype" in dfs["battles"].columns:
    df = df.merge(dfs["battles"][["isqno", "postype"]], on="isqno", how="left")
else:
    df["postype"] = "NA"
safe_map("postype", "action_initiated", "Unknown")
df.rename(columns={"postype_desc": "tactical_posture"}, inplace=True)

# Surprise
safe_map("surpa", "element_of_surprise", "None")
df.rename(columns={"surpa_desc": "surprise_level"}, inplace=True)

# Air superiority
if "aeroa" in dfs["battles"].columns:
    df = df.merge(dfs["battles"][["isqno", "aeroa"]], on="isqno", how="left")
else:
    df["aeroa"] = 0
safe_map("aeroa", "air_superiority", "None")
df.rename(columns={"aeroa_desc": "air_superiority"}, inplace=True)


# ====================== SMALL TABLE PROCESSING ======================
# Belligerents one-hot encoding
if "belligerents" in dfs:
    bel = dfs["belligerents"]
    bel_cols = [c for c in bel.columns if c != "isqno"]
    if bel_cols:
        bel_onehot = pd.get_dummies(bel, columns=bel_cols, prefix="bel")
        bel_onehot = bel_onehot.groupby("isqno").sum().reset_index()
        df = df.merge(bel_onehot, on="isqno", how="left")
        print("Added belligerents one-hot")
    else:
        print("WARNING: belligerents.xlsx has no data columns")

# --- Clear Cut ---
if "clear_cut" in dfs:
    print("  clearCut.xlsx is an enum table – skipping merge")
else:
    print("WARNING: clearCut.xlsx not found")

# Top commanders (one-hot)
if "commanders" in dfs:
    cmd_df = dfs["commanders"]
    print(f"  commanders.xlsx columns: {list(cmd_df.columns)}")
    isqno_col = next((c for c in cmd_df.columns if c.lower() == "isqno"), None)
    cmd_col = next((c for c in cmd_df.columns if c.lower() in ["commanders", "commander", "name"]), None)
    if isqno_col and cmd_col:
        cmd = cmd_df[[isqno_col, cmd_col]].copy()
        cmd.rename(columns={isqno_col: "isqno", cmd_col: "commander_name"}, inplace=True)
        # Split multiple commanders if comma-separated
        cmd["commander_name"] = cmd["commander_name"].str.split(",").str[0].str.strip()
        top_cmd = cmd["commander_name"].value_counts().head(10).index
        cmd = cmd[cmd["commander_name"].isin(top_cmd)]
        cmd_onehot = pd.get_dummies(cmd, columns=["commander_name"], prefix="cmd")
        cmd_onehot = cmd_onehot.groupby("isqno").sum().reset_index()
        df = df.merge(cmd_onehot, on="isqno", how="left")
        print("Added top-10 commanders")
    else:
        print("WARNING: commanders.xlsx missing 'isqno' or commander column – skipping")

# Favors side (enum table)
if "favors_side" in dfs:
    print("  favors_side.xlsx is an enum table – skipping merge")
else:
    print("WARNING: favors_side.xlsx not found")


# ====================== FINAL CLEANING & SPLIT ======================
# Impute numeric missing values with median
num_cols = df.select_dtypes(include="number").columns
num_cols = num_cols.drop("isqno") if "isqno" in num_cols else num_cols
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# One-hot encode remaining text categories
cat_cols = [c for c in ["surprise_level", "air_superiority", "tactical_posture"] if c in df.columns]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

# Drop original raw code columns
drop_raw = ["terra", "surpa", "postype", "aeroa"]
df.drop(columns=[c for c in drop_raw if c in df.columns], inplace=True, errors="ignore")

df["force_ratio"] = 1.0  # placeholder

# Encode target variable
le = LabelEncoder()
df["target"] = le.fit_transform(df["direction"].astype(str))

# Prepare final feature matrix
feature_cols = [c for c in df.columns if c not in ["isqno", "direction", "target"]]
X = df[feature_cols]
y = df["target"]

print(f"\nFinal X shape: {X.shape}")
print(f"Features ({len(feature_cols)}): {sorted(feature_cols)}")

# Train / Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ====================== SAVE OUTPUTS ======================
out_path = os.path.join(MODEL_DIR, "data_split_enhanced.pkl")
joblib.dump({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "label_encoder": le,
    "feature_names": feature_cols
}, out_path)

pd.Series(feature_cols).to_csv(os.path.join(MODEL_DIR, "feature_list.csv"), index=False, header=["feature"])

print(f"\nSaved to: {out_path}")
print(f"Feature list saved to: models/feature_list.csv")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")