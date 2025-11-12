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

class PredictionChart(StockChart):
    """Chart for plotting actual vs predicted values along with features"""
    def __init__(self, df: pd.DataFrame, ticker: str, preds: pd.Series, sma_windows=[30, 90], vol_col="Vol_30"):
        super().__init__(df, ticker, sma_windows, vol_col)
        self.preds = preds

    def show(self):
        self.plot_base_chart()
        # Plot predicted values
        plt.plot(self.preds.index, self.preds.values,
                 label="Predicted Next-Day Close (LR)", linewidth=1.25, color='green')

        plt.title(f"{self.ticker} — Actual vs Predicted Next-Day Close")
        out_file = os.path.join(VISUALS_DIR, "models", f"{self.ticker}_predictions.png")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file, dpi=250)
        plt.show()
        print(f"Saved prediction chart to: {out_file}")
    
class TrendAnalysisChart(StockChart):
    def show(self):
        self.plot_base_chart()
        out_file = os.path.join(self.output_dir, f"{self.ticker}_EDA.png")
        plt.savefig(out_file, dpi=300)
        plt.show()
        print(f"Saved EDA chart to: {out_file}")

