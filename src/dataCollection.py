import yfinance as yf
import pandas as pd
import os
from datetime import date
from dateutil.relativedelta import relativedelta

def fetch_data(ticker, start, end):
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created main data directory: {output_dir}")

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        print(f"Error: Could not fetch data for {ticker}. Check the ticker symbol or range.")
        return pd.DataFrame()

    file_path = f"{output_dir}/{ticker}.csv"
    data.to_csv(file_path)
    print(f"Saved {ticker} data to {file_path}")

    return data

#testing block
if __name__ == "__main__":
    print("Standalone Data Collection Test")
    tickers = input("Enter tickers (e.g. AAPL, NVDA, BTC-USD): ").upper().split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    start_date = (date.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date = "2050-12-31"

    for ticker in tickers:
        fetch_data(ticker, start=start_date, end=end_date)
