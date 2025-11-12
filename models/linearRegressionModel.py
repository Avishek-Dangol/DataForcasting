import os
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_DIR = os.path.join("data")
MODELS_DIR = os.path.join("models")
VISUALS_DIR = os.path.join("visuals")  
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


def prepare_data_for_model(df: pd.DataFrame, feature_cols: list, target_shift: int = 1) -> pd.DataFrame:
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
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


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

    preds = model.predict(X_test_scaled)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }

    preds_series = pd.Series(preds, index=test_df.index, name="Predicted_Close")
    return model, scaler, metrics, preds_series


def save_artifacts(model: LinearRegression, scaler: StandardScaler, ticker: str):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_linearreg_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")


def forecast_future(df: pd.DataFrame, model: LinearRegression, scaler: StandardScaler, feature_cols: list, days: int):
    """Forecast N days ahead using recursive prediction."""
    future_df = df.copy()
    preds = []

    for _ in range(days):
        X_last = future_df.iloc[-1][feature_cols].values.reshape(1, -1)
        X_last_scaled = scaler.transform(X_last)
        next_close = model.predict(X_last_scaled)[0]
        preds.append(next_close)

        new_row = future_df.iloc[-1].copy()
        new_row["Close"] = next_close

        # Update rolling features
        for col in feature_cols:
            if col.startswith("SMA_"):
                window = int(col.split("_")[1])
                new_row[col] = pd.concat([future_df["Close"], pd.Series([next_close])]).tail(window).mean()
            elif col == "Return":
                new_row[col] = (next_close - future_df["Close"].iloc[-1]) / future_df["Close"].iloc[-1]

        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.Series(preds, index=future_dates, name="Forecast_Close")

def process_file(file_path: str, feature_cols: list = None, forecast_days: int = 10):
    if feature_cols is None:
        feature_cols = ["Return", "SMA_7", "SMA_30", "SMA_90", "Vol_30"]

    ticker = os.path.basename(file_path).replace(".csv", "")
    print(f"\n--- Modeling {ticker} ---")

    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df_model = prepare_data_for_model(df, feature_cols)

    if df_model.empty or len(df_model) < 50:
        print(f"Not enough data for {ticker}. Skipping.")
        return

    train_df, test_df = time_based_train_test_split(df_model)
    model, scaler, metrics, preds = train_and_evaluate(train_df, test_df, feature_cols)

    print("Model evaluation metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    save_artifacts(model, scaler, ticker)

    future_preds = forecast_future(df_model, model, scaler, feature_cols, forecast_days)

    plt.figure(figsize=(12, 6))
    plt.plot(df_model.index, df_model["Close"], color="blue", linewidth=1.5, label="Historical Close")
    plt.plot(preds.index, preds.values, "orange", linewidth=1.8, label="Model Predictions")

    plt.plot(
        future_preds.index,
        future_preds.values,
        color="orange",
        linestyle="--",
        linewidth=2.0,
        label=f"{forecast_days}-Day Forecast",
    )

    plt.axvspan(df_model.index[-1], future_preds.index[-1], color="orange", alpha=0.08)

    plt.title(f"{ticker} Price Forecast ({forecast_days}-Day Extension)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(VISUALS_DIR, f"{ticker}_forecast.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved extended forecast chart to: {save_path}\n")


def run_all_models_on_data():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files:
        print("No CSV files found in data/. Run preprocessing first.")
        return
    
    forecast_days = int(input("\nHow many days into the future do you want to forecast? (e.g., 5, 10, 30): "))

    for fname in files:
        try:
            process_file(os.path.join(DATA_DIR, fname), forecast_days=forecast_days)
        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    run_all_models_on_data()