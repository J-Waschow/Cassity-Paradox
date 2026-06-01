# Historical Military Battles Prediction

**A Data Science Project Exploring Classical War Theory through Machine Learning**

**Course:** CSCI 710  
**Author:** Jordan Waschow  
**Date:** December 2025

---

## Project Overview

This project investigates whether classical military theory — particularly **logistics** and **sea power** — remains predictive of battle outcomes when analyzed with modern machine learning techniques.

The project is structured across **three milestones**:
- **Milestone 1**: Baseline models using traditional features
- **Milestone 2**: Introduction of theory-driven tactical features (`supply_cut`, `naval_power`)
- **Milestone 3**: Final optimized models, interaction terms, and formal validation of war theory

**Key Finding**: Logistics disruption (`supply_cut` and `supply_surprise`) consistently emerged as the strongest predictors across all models, providing empirical support for classical military thinkers (Moltke, van Creveld, and Mahan).

---

## Project Structure
Historical Military Battles Prediction/
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
│   ├── compare_models.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
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
├── results/                          # Outputs from Milestone 1 and 2
│   └── (timestamped folders from early runs)
│
├── data/                             # Raw Excel data files
├── models/                           # Intermediate processed data
└── README.md


### Folder Explanation

- **`scripts/`** — Contains the original Milestone 1 baseline scripts (first runs).
- **`second_gen/`** — Milestone 2 development with tactical features and war theory integration.
- **`final_milestone/`** — Final polished scripts used for Milestone 3.
- **`final_milestone/results/`** — Contains the final model outputs, figures, and validation results from Milestone 3.
- **`results/`** — Archive of outputs from Milestone 1 and Milestone 2 runs.

---

## How to Run the Project

1. **Prepare Data**  
   Place all raw `.xlsx` files in the `data/` folder.

2. **Milestone 1** (Optional)  
   Run the scripts in the `scripts/` folder.

3. **Milestone 2**  
   Run the files in `second_gen/` in sequential order.

4. **Milestone 3** (Final Models)  
   ```bash
   cd final_milestone
   python train_logistic_regression.py
   python train_random_forest.py
   python train_xgboost.py
   python compare_models.py
   python war_theory_validation.py