# Cassity-Paradox

**An Unbiased Pre-Battle Military Outcome Prediction Algorithm**

A comparative machine learning project that predicts interstate war outcomes using only **pre-engagement factors** from the Correlates of War (COW) dataset.

---

## 🎯 Project Objective

Develop a **transparent, bias-mitigated framework** for forecasting war winners based exclusively on quantifiable pre-battle variables such as force ratios, logistics, naval power, surprise, terrain, and leadership metrics.

The project tests whether classical military theory remains predictive in the modern machine learning era.

---

## Key Findings

**YES — Classical war theory remains highly relevant.**

- `supply_surprise` and `supply_cut` consistently ranked as the strongest predictors across all models.
- Strong empirical validation for Moltke / van Creveld ("Logistics is 90% of war") and Mahan ("Command of the sea").
- Supports **The Cassity Paradox** (coined by Jim Cassity).

---

## Project Structure

```markdown
Cassity-Paradox/
├── scripts/                          # Milestone 1 - Initial Baseline Runs
│   ├── preprocess.py
│   ├── logreg_model.py
│   ├── mlp_model.py
│   └── rf_model.py
│
├── second_gen/                       # Milestone 2 - Tactical Features & Theory Integration
│   ├── 00_load_split_v2.py
│   ├── 01_train_logreg_theory.py
│   ├── 02_train_rf_v2.py
│   ├── 03_train_mlp_v2.py
│   ├── tactical_factors.py
│   └── war_theory.py
│
├── final_milestone/                  # Milestone 3 - Final Optimized Models
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── compare_models.py
│   └── war_theory_validation.py
│
├── final_milestone/results/          # Final Model Outputs & Reports
│   ├── figures/                      # All visualization outputs
│   ├── data_ready.pkl
│   ├── logistic_regression_optimized.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── model_comparison.csv
│   ├── war_theory_final.csv
│   └── ...
│
├── results/                          # Archived outputs from Milestone 1 and 2
├── data/                             # Raw Excel data files
├── models/                           # Intermediate processed data
└── README.md
