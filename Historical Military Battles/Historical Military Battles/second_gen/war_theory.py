"""
war_theory.py
WAR THEORY FEATURE SET - MILESTONE 2

Author: Jordan Waschow
Project: Historical Military Battles Prediction
Course: CSCI 710
Milestone: 2
Date: November 2025

Theoretical Foundation:
    • Force Ratio & Duration → Clausewitz & Lanchester
    • Surprise → Sun Tzu & Boyd (OODA Loop)
    • Air Superiority → Warden’s Five Rings & modern airpower doctrine
    • Terrain → Jomini’s decisive points
    • Supply Cut & Naval Power → Moltke, van Creveld (logistics) and Mahan (sea power)
    • Multi-phase → Soviet Deep Battle doctrine

Important Limitations:
    • This feature list is **hard-coded** and static.
    • It was designed specifically for Milestone 2 experiments.
    • **This module was NOT used in Milestone 3**, as the project moved toward
      automated feature selection, expanded interaction terms, and more dynamic
      modeling approaches.
    • The narrow focus (only 20 features) was intentional for interpretability
      but may limit model performance compared to using the full feature set.
"""

# ====================== THEORY-ALIGNED FEATURE LIST ======================
# This list represents the curated, theory-driven feature set used in Milestone 2.
# It deliberately prioritizes interpretability and alignment with military theory
# over using every available feature from preprocessing.
THEORY_FEATURES = [
    # Core quantitative
    "duration_days",           # Longer battles = defender advantage (attrition)
    "multi_phase",             # Complex battles favor better C2 and logistics
    "force_ratio",             # Lanchester square law approximation

    # High-Impact Tactical (from battle names)
    "supply_cut",              # Sieges, blockades, encirclement → decisive
    "naval_power",             # Amphibious, naval gunfire, carrier strikes

    # Surprise (element of surprise)
    "surprise_level_Complete surprise achieved by attacker",
    "surprise_level_Substantial surprise achieved by attacker",
    "surprise_level_Minor surprise achieved by attacker",
    "surprise_level_Neither side achieved surprise, or it did not affect the outcome",
    "surprise_level_None",

    # Air Superiority
    "air_superiority_Attacker had air superiority in the theater",
    "air_superiority_Neither side had theater air superiority",
    "air_superiority_None",

    # Tactical Posture (limited data → only Unknown used)
    "tactical_posture_Unknown",

    # Terrain (one-hot encoded from terra1/terra2/terra3)
    "terra_B",   # Built-up / Urban
    "terra_D",   # Desert
    "terra_F",   # Forest
    "terra_G",   # Grassland / Plain
    "terra_M",   # Mountain
    "terra_R",   # Rough / Broken
    "terra_U",   # Unknown
    "terra_W"    # Water / Marsh
]

# Print for verification
if __name__ == "__main__":
    print("WAR THEORY FEATURE SET (n=20)")
    print("=" * 50)
    for i, feat in enumerate(THEORY_FEATURES, 1):
        print(f"{i:2d}. {feat}")
    print(f"\nTotal features: {len(THEORY_FEATURES)}")
    print("This set is used in RF, LogReg, and XGBoost models.")