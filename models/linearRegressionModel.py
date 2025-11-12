import os
import warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_DIR = os.path.join("data")
MODELS_DIR = os.path.join("models")
VISUALS_DIR = os.path.join("visuals", "models")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

def prepare_data_for_model(df: pd.DataFrame, feature_cols: list, target_shift: int = 1) -> pd.DataFrame:
    # required columns check
    required = set(feature_cols + ["Close"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    df = df.copy()

    df["Target_Close"] = df["Close"].shift(-target_shift)
    df = df.dropna(subset=list(feature_cols) + ["Target_Close"])

    return df

def time_based_train_test_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()
    return train_df, test_df


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> Tuple[LinearRegression, StandardScaler, Dict[str, float], pd.Series]:

    X_train = train_df[feature_cols].values
    y_train = train_df["Target_Close"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Target_Close"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    # Return model, scaler, metrics, and Series of predictions
    preds_series = pd.Series(y_pred, index=test_df.index, name="Predicted_Close")
    return model, scaler, metrics, preds_series


def save_artifacts(model: LinearRegression, scaler: StandardScaler, ticker: str):
    """Save model and scaler to models/ directory with ticker-specific names."""
    model_path = os.path.join(MODELS_DIR, f"{ticker}_linearreg_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")


def plot_predictions(test_df: pd.DataFrame, preds: pd.Series, ticker: str):
    """
    Plot actual vs predicted Close prices for the test period and save figure.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df["Target_Close"], label="Actual Next-Day Close", linewidth=1.25)
    plt.plot(preds.index, preds.values, label="Predicted Next-Day Close (LR)", linewidth=1.25)
    plt.title(f"{ticker} — Actual vs Predicted Next-Day Close (Linear Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_file = os.path.join(VISUALS_DIR, f"{ticker}_lr_predictions.png")
    plt.savefig(out_file, dpi=250)
    plt.show()
    print(f"Saved prediction plot to: {out_file}")


def process_file(file_path: str, feature_cols: list = None):
    """
    Single-file processing: load data, prepare, train, evaluate, save artifacts, plot.
    """
    if feature_cols is None:
        feature_cols = ["Return", "SMA_7", "SMA_30", "SMA_90", "Vol_30"]

    ticker = os.path.basename(file_path).replace(".csv", "")
    print(f"\n--- Modeling {ticker} ---")

    # Load processed CSV 
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Prepare data 
    df_model = prepare_data_for_model(df, feature_cols)

    if df_model.empty or len(df_model) < 50:
        print(f"Not enough data for {ticker} after preparing model dataset (rows: {len(df_model)}). Skipping.")
        return

    # Time-based train/test split
    train_df, test_df = time_based_train_test_split(df_model, train_frac=0.8)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # Train and evaluate
    model, scaler, metrics, preds = train_and_evaluate(train_df, test_df, feature_cols)

    # Print metrics
    print("Model evaluation metrics (on test set):")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")

    # Save model & scaler
    save_artifacts(model, scaler, ticker)

    # Plot predictions vs actual
    plot_predictions(test_df, preds, ticker)


def run_all_models_on_data():
    """
    Run model training for every CSV in DATA_DIR.
    """
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files:
        print("No CSV files found in data/. Run preprocessing first.")
        return

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        try:
            process_file(path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    run_all_models_on_data()