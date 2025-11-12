import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"

def plot_fibonacci_retracement(df, ticker):
    """
    Plots stock closing prices, moving averages, and Fibonacci retracement levels.
    """
    if 'Close' not in df.columns:
        print(f"Error: Missing 'Close' column in {ticker}.")
        return

    recent_high = df['Close'].max()
    recent_low = df['Close'].min()

    diff = recent_high - recent_low
    fib_levels = {
        '0.0%': recent_high,
        '23.6%': recent_high - 0.236 * diff,
        '38.2%': recent_high - 0.382 * diff,
        '50.0%': recent_high - 0.5 * diff,
        '61.8%': recent_high - 0.618 * diff,
        '78.6%': recent_high - 0.786 * diff,
        '100%': recent_low
    }

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label="Close Price", color='black', linewidth=1)
    plt.plot(df.index, df['SMA_30'], label="30-Day SMA", color='orange', linewidth=1.5)
    plt.plot(df.index, df['SMA_90'], label="90-Day SMA", color='purple', linewidth=1.5)
    
    # Add Fibonacci retracement levels
    for level, value in fib_levels.items():
        plt.axhline(value, linestyle='--', alpha=0.5, label=f'Fib {level}')

    # Highlight volatility zones visually
    if 'Vol_30' in df.columns:
        high_vol_threshold = df['Vol_30'].mean() + df['Vol_30'].std()
        high_vol_periods = df[df['Vol_30'] > high_vol_threshold]
        plt.scatter(high_vol_periods.index, high_vol_periods['Close'], color='red', s=10, label='High Volatility')

    plt.title(f"{ticker} Price Chart with Fibonacci Retracement Levels", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_dir = os.path.join("visuals", "technical_analysis")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{ticker}_Fibonacci.png", dpi=300)
    plt.show()
    print(f"Created Technical Analysis chart for {ticker} in '{output_dir}/'.")

def technicalAnalysis():
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Run preprocessing first.")
        return

    tickers_processed = 0

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "")
            file_path = os.path.join(DATA_DIR, filename)

            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            plot_fibonacci_retracement(df, ticker)
            tickers_processed += 1

    print(f"\nTechnical Analysis Complete: {tickers_processed} charts generated.")

if __name__ == "__main__":
    technicalAnalysis()
