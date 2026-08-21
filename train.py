"""
Customer Churn Prediction - ANN training pipeline
Dataset: Churn_Modelling.csv
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE  # noqa: F401 (kept for optional use)
import optuna

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1_l2

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 21
DATA_PATH = "data/Churn_Modelling.csv"
MODEL_PATH = "saved_model/model.keras"
PREPROCESSOR_PATH = "saved_model/preprocessor.pkl"
N_TRIALS = 15

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    return ColumnTransformer(
        [
            ("Scaling", StandardScaler(), num_cols),
            ("Encoding", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def build_baseline_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_class_weights(y_train) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


def tune_hyperparameters(X_train_t, y_train, X_val_t, y_val, class_weight_dict, n_trials=N_TRIALS):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        lr_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = Sequential()
        model.add(Input(shape=(X_train_t.shape[1],)))
        for i in range(n_layers):
            units = trial.suggest_int(f"units{i}", 8, 64)
            dropout = trial.suggest_float(f"dropout{i}", 0.0, 0.5)
            reg = trial.suggest_float(f"reg{i}", 1e-5, 1e-2, log=True)
            model.add(Dense(units, activation=activation, kernel_regularizer=l1_l2(l1=reg, l2=reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))

        opt = Adam(learning_rate=lr_rate) if optimizer_name == "Adam" else RMSprop(learning_rate=lr_rate)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        es = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        model.fit(
            X_train_t, y_train,
            validation_data=(X_val_t, y_val),
            epochs=30, batch_size=batch_size,
            callbacks=[es], class_weight=class_weight_dict, verbose=0,
        )

        val_pred = np.where(model.predict(X_val_t, verbose=0) > 0.5, 1, 0)
        return f1_score(y_val, val_pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def build_final_model(input_dim: int, best_params: dict) -> Sequential:
    n_layers = best_params["n_layers"]
    activation = best_params["activation"]
    optimizer_name = best_params["optimizer"]
    lr_rate = best_params["learning_rate"]

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for i in range(n_layers):
        units = best_params[f"units{i}"]
        dropout = best_params[f"dropout{i}"]
        reg = best_params[f"reg{i}"]
        model.add(
            Dense(units, activation=activation, kernel_initializer="he_normal",
                  kernel_regularizer=l1_l2(l1=reg, l2=reg))
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))

    opt = Adam(learning_rate=lr_rate) if optimizer_name == "Adam" else RMSprop(learning_rate=lr_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Recall(name="recall")])
    return model


def evaluate(model, X, y, label: str):
    y_pred = np.where(model.predict(X, verbose=0) > 0.5, 1, 0)
    acc = accuracy_score(y, y_pred)
    print(f"\n{label} Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
    return acc


def main():
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print("=== Baseline model ===")
    baseline = build_baseline_model(X_train_t.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    baseline.fit(X_train_t, y_train, validation_data=(X_val_t, y_val),
                 epochs=50, batch_size=32, callbacks=[es], verbose=0)
    evaluate(baseline, X_test_t, y_test, "Baseline Test")

    class_weight_dict = get_class_weights(y_train)
    print("\nClass weights:", class_weight_dict)

    print("\n=== Hyperparameter tuning (Optuna) ===")
    study = tune_hyperparameters(X_train_t, y_train, X_val_t, y_val, class_weight_dict)
    print("Best params:", study.best_params)
    print("Best validation F1:", study.best_value)

    print("\n=== Final tuned model ===")
    final_model = build_final_model(X_train_t.shape[1], study.best_params)
    es_final = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    final_model.fit(
        X_train_t, y_train,
        validation_data=(X_val_t, y_val),
        epochs=100, batch_size=study.best_params["batch_size"],
        callbacks=[es_final], class_weight=class_weight_dict, verbose=2,
    )

    evaluate(final_model, X_train_t, y_train, "Train")
    evaluate(final_model, X_val_t, y_val, "Validation")
    evaluate(final_model, X_test_t, y_test, "Test")

    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    final_model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH} and preprocessor to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
