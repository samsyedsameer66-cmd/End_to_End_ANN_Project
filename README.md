# Customer Churn Prediction — ANN Project

An end-to-end **Artificial Neural Network (ANN)** project that predicts whether a bank
customer will churn (exit) using the **Churn Modelling** dataset. The project follows a
structured workflow — problem definition → EDA → preprocessing → baseline model →
diagnosis → hyperparameter tuning → final model → evaluation → saving artifacts — based on
the ANN project methodology outlined in `docs/ANN_Project_Documentation.docx`.

## Problem Statement

Given demographic and account information about a bank's customers, predict whether a
customer will **exit (churn)** in the near future (`Exited`: 0 = stays, 1 = churns).

This is a **binary classification** problem with a **moderate class imbalance**
(~80% stayed / ~20% churned).

## Dataset

**Source:** [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) (Kaggle)

| Property | Value |
|---|---|
| Rows | 10,000 |
| Columns | 14 |
| Target | `Exited` (0/1) |
| Missing values | None |
| Duplicate rows | None |

Key features: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`,
`NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
`RowNumber`, `CustomerId`, and `Surname` are identifiers and are dropped before modeling.

## Project Workflow

| Step | What was done |
|---|---|
| 1. Problem definition | Identified as binary classification → sigmoid output, binary cross-entropy loss |
| 2. EDA | Checked shape, nulls, duplicates, feature types, and target distribution |
| 3. Data preparation | Dropped identifier columns, stratified train/val/test split (60/20/20), scaled numeric features (`StandardScaler`), one-hot encoded categorical features |
| 4. Baseline ANN | Small 2-hidden-layer network (32 → 16 → 1) trained with EarlyStopping |
| 5. Diagnosis | Baseline had strong accuracy but weak recall on churners → confirmed class imbalance issue |
| 6. Imbalance handling | Computed balanced `class_weight` for the loss function |
| 7. Hyperparameter tuning | Used **Optuna** (15 trials) to search layers, units, dropout, L1/L2 regularization, activation, optimizer, and learning rate — optimizing validation **F1-score** |
| 8. Final model | Rebuilt with best hyperparameters + He-normal init + BatchNorm + Dropout + class weights + EarlyStopping |
| 9. Evaluation | Reported train/val/test accuracy, classification report, and confusion matrix on the **untouched test set** |
| 10. Artifacts | Saved trained model (`model.keras`) and fitted preprocessing pipeline (`preprocessor.pkl`) |

## Results

| Model | Test Accuracy | Recall (churn class) | F1 (churn class) |
|---|---|---|---|
| Baseline ANN | ~0.86 | ~0.52 | ~0.61 |
| Tuned ANN (class weights + Optuna) | ~0.81 | ~0.74 | ~0.61 |

The tuned model trades a small amount of overall accuracy for a **large gain in recall** on
the churn class (the customers the bank actually cares about catching) — this reflects the
documentation's core lesson: *"accuracy alone can be misleading on imbalanced data; the best
model is the one that catches the important class reliably."*

## Repository Structure

```
.
├── data/
│   └── Churn_Modelling.csv        # raw dataset
├── saved_model/
│   ├── model.keras                # trained ANN
│   └── preprocessor.pkl           # fitted ColumnTransformer (scaler + one-hot encoder)
├── docs/
│   └── ANN_Project_Documentation.docx   # workflow reference followed for this project
├── Churn_ANN_Project.ipynb        # full end-to-end notebook (EDA → model → evaluation)
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook Churn_ANN_Project.ipynb
```

## Tech Stack

- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** TensorFlow / Keras
- **Preprocessing:** scikit-learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Imbalance handling:** class weights (scikit-learn), `imbalanced-learn`
- **Hyperparameter tuning:** Optuna

## Key Learnings Applied from the Documentation

- Binary classification → sigmoid output + binary cross-entropy loss
- Stratified splitting to avoid leakage and preserve class ratio across sets
- Scaling numeric features and one-hot encoding categoricals before feeding an ANN
- Diagnosing imbalance instead of trusting accuracy alone
- Using class weights and F1/recall-oriented tuning instead of assuming a 0.5 threshold is optimal
- Using learning curves (train vs. validation loss/accuracy) to check for over/underfitting
- Reserving the test set strictly for final, one-time evaluation

## Author

Built as part of an ANN project assignment, following the workflow provided in
`ANN_Project_Documentation_Steps_to_follow.docx`.
