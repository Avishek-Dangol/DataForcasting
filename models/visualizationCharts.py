import os
import matplotlib.pyplot as plt
import pandas as pd

VISUALS_DIR = "visuals"  # base folder for all visuals

class StockChart:
    """
    Base class for plotting stock charts with common features:
    - Moving averages
    - Fibonacci retracement levels
    - High volatility highlighting
    """
    def __init__(self, df: pd.DataFrame, ticker: str, sma_windows=[30, 90], vol_col="Vol_30"):
        self.df = df.copy()
        self.ticker = ticker
        self.sma_windows = sma_windows
        self.vol_col = vol_col

        self.recent_high = df['Close'].max()
        self.recent_low = df['Close'].min()
        self.diff = self.recent_high - self.recent_low

        self.fib_levels = {
            '0.0%': self.recent_high,
            '23.6%': self.recent_high - 0.236 * self.diff,
            '38.2%': self.recent_high - 0.382 * self.diff,
            '50.0%': self.recent_high - 0.5 * self.diff,
            '61.8%': self.recent_high - 0.618 * self.diff,
            '78.6%': self.recent_high - 0.786 * self.diff,
            '100%': self.recent_low
        }

        # Ensure output directory exists
        self.output_dir = os.path.join(VISUALS_DIR, "technical_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_base_chart(self):
        """Plot Close price, moving averages, Fibonacci levels, and high volatility zones."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df['Close'], label="Close Price", color='black', linewidth=1)

        # Plot SMAs
        for window in self.sma_windows:
            col_name = f"SMA_{window}"
            if col_name in self.df.columns:
                plt.plot(self.df.index, self.df[col_name], label=f"{window}-Day SMA", linewidth=1.5)

        # Plot Fibonacci levels
        for level, value in self.fib_levels.items():
            plt.axhline(value, linestyle='--', alpha=0.5, label=f'Fib {level}')

        # Highlight high volatility zones
        if self.vol_col in self.df.columns:
            high_vol_threshold = self.df[self.vol_col].mean() + self.df[self.vol_col].std()
            high_vol_periods = self.df[self.df[self.vol_col] > high_vol_threshold]
            plt.scatter(high_vol_periods.index, high_vol_periods['Close'],
                        color='red', s=10, label='High Volatility')

        plt.title(f"{self.ticker} Price Chart", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()

class PredictionChart:
    def __init__(self, test_df, preds, ticker, model=None, scaler=None, feature_cols=None, forecast_days=10):
        self.test_df = test_df
        self.preds = preds
        self.ticker = ticker
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.forecast_days = forecast_days

    def forecast_future(self):
        """Generate N-day forecast continuation from last row."""
        if self.model is None or self.scaler is None or self.feature_cols is None:
            return pd.Series([], dtype=float)

        last_row = self.test_df.iloc[-1].copy()
        forecasts = []
        dates = []

        for i in range(self.forecast_days):
            X_last = last_row[self.feature_cols].values.reshape(1, -1)
            X_scaled = self.scaler.transform(X_last)
            next_close = self.model.predict(X_scaled)[0]

            next_date = self.test_df.index[-1] + timedelta(days=i + 1)
            forecasts.append(next_close)
            dates.append(next_date)

            # Update features recursively (simplified version)
            last_row["Close"] = next_close
            last_row["Return"] = (next_close - last_row["Close"]) / last_row["Close"]

        return pd.Series(forecasts, index=pd.to_datetime(dates), name="Forecast_Close")

    def show(self):
        """Show extended prediction chart with future continuation."""
        future_series = self.forecast_future()

        plt.figure(figsize=(12, 6))
        plt.plot(self.test_df.index, self.test_df["Target_Close"], color="blue", label="Actual Close", linewidth=1.6)
        plt.plot(self.preds.index, self.preds.values, "r--", label="Predicted Close", linewidth=1.8)

        if not future_series.empty:
            plt.plot(future_series.index, future_series.values, "r--", alpha=0.8, linewidth=2.0, label=f"{self.forecast_days}-Day Forecast")

        plt.title(f"{self.ticker} Forecast (Linear Regression)", fontsize=14, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = f"visuals/{self.ticker}_forecast.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Saved extended forecast analysis chart to: {save_path}")

class TrendAnalysisChart(StockChart):
    def show(self):
        self.plot_base_chart()
        out_file = os.path.join(self.output_dir, f"{self.ticker}_EDA.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved analysis chart to: {out_file}")

