# Customer Churn Prediction — ANN

End-to-end ANN pipeline predicting bank customer churn (`Exited`: 0/1) using the Churn
Modelling dataset. Built with TensorFlow/Keras, tuned with Optuna, and handles class
imbalance via class weighting.

## Results

| Model | Test Accuracy | Recall (churn class) | F1 (churn class) |
|---|---|---|---|
| Baseline ANN | ~0.86 | ~0.52 | ~0.61 |
| Tuned ANN (class weights + Optuna) | ~0.80 | ~0.72 | ~0.59 |

## Project Structure

```
.
├── data/
│   └── Churn_Modelling.csv
├── saved_model/
│   ├── model.keras
│   └── preprocessor.pkl
├── docs/
│   └── ANN_Project_Documentation.docx
├── train.py
├── requirements.txt
└── README.md
```

## Pipeline (`train.py`)

1. Load data, drop identifier columns (`RowNumber`, `CustomerId`, `Surname`)
2. Stratified train/val/test split (60/20/20)
3. Preprocess: `StandardScaler` on numeric features, `OneHotEncoder` on categorical features
4. Train a baseline ANN (2 hidden layers)
5. Compute balanced class weights (target is ~80/20 imbalanced)
6. Tune architecture/hyperparameters with Optuna (objective: validation F1-score)
7. Train the final model (best hyperparameters + He-init + BatchNorm + Dropout + L1/L2 + class weights)
8. Evaluate on train/val/test sets
9. Save `model.keras` and `preprocessor.pkl`

## Setup & Run (training)

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
python train.py
```

## Run the Streamlit App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `saved_model/model.keras` and
`saved_model/preprocessor.pkl` to already exist (already included in this repo).

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (make sure `app.py`, `saved_model/`, and `requirements.txt`
   are all included — `saved_model/` must NOT be in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Select this repository, branch `main`, and set **Main file path** to `app.py`.
5. Click **Deploy**.
6. Streamlit Cloud installs everything from `requirements.txt` automatically and gives you
   a public URL like `https://<your-app-name>.streamlit.app`.

## Tech Stack

pandas, numpy, scikit-learn, TensorFlow/Keras, imbalanced-learn, Optuna
