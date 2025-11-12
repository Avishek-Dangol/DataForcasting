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
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data.reset_index(inplace=True) 
    file_path = f"{output_dir}/{ticker}.csv"
    data.to_csv(file_path, index=False)
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
