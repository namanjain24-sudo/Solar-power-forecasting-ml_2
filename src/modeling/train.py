from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

from src.data.load_data import load_data
from src.preprocessing.preprocessing import encode_features, split_data
from src.evaluation.metrics import evaluate_model, print_metrics


def train_model():
    print("Loading data...")
    df = load_data()

    # Encode categorical features
    df = encode_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nFeature types:\n", X_train.dtypes)


    # ── Train Model ──
    print("\nTraining RandomForest...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("✅ Model trained successfully.")

    # ── Evaluation ──
    preds = np.clip(model.predict(X_test), 0, None)
    metrics = evaluate_model(y_test, preds)
    print_metrics(metrics, "Holdout Evaluation")

    # ── Cross Validation ──
    print("\nRunning TimeSeriesSplit CV...")

    df_sorted = df.sort_values("DATE_TIME")

    # IMPORTANT: use same numeric columns only
    features = list(X_train.columns)
    X_full = df_sorted[features]
    y_full = df_sorted["DC_POWER"]

    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), 1):
        X_tr = X_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]
        y_tr = y_full.iloc[train_idx]
        y_val = y_full.iloc[val_idx]

        cv_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        cv_model.fit(X_tr, y_tr)
        cv_preds = np.clip(cv_model.predict(X_val), 0, None)

        fold_metrics = evaluate_model(y_val, cv_preds)

        print(f"Fold {fold} -> MAE: {fold_metrics['MAE']:.2f}, RMSE: {fold_metrics['RMSE']:.2f}")

    # ── Save model ──
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/solar_model.pkl")
    print("✅ Model saved at models/solar_model.pkl")

    # ── Save log ──
    log = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "metrics": metrics
    }

    with open("training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("✅ Training log saved")

    # ── Plot ──
    plt.figure()
    plt.plot(y_test.values[:200], label="Actual")
    plt.plot(preds[:200], label="Predicted")
    plt.legend()
    plt.title("Prediction vs Actual")
    plt.savefig("reports/prediction_vs_actual.png")

    print("✅ Training complete!")


if __name__ == "__main__":
    train_model()