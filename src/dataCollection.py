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
        print(f"Error: Could not fetch data for {ticker}. Check the ticker symbol and data range.")
        return pd.DataFrame()

    file_path=f"{output_dir}/{ticker}.csv"    
    data.to_csv(file_path)

    return data

if __name__ == "__main__":
    print("Welcome to Data Collection. Enter tickers separated by a comma (e.g., AAPL, NVDA, BTC-USD):")
    ticker_input = input("Tickers: ").upper()

    TICKERS = [t.strip() for t in ticker_input.split(',') if t.strip()]

    TODAY = date.today()
    FIVE_YEARS_AGO = TODAY - relativedelta(years=5)
    
    START_DATE = FIVE_YEARS_AGO.strftime("%Y-%m-%d") 
    
    FUTURE_END_DATE = "2050-12-31" 
    
    downloaded_tickers = []
    
    for ticker in TICKERS:
        df = fetch_data(ticker, START_DATE, FUTURE_END_DATE)
        if not df.empty:
            downloaded_tickers.append(ticker)
            
    if downloaded_tickers:
        print(f"\n Data collection complete for: {', '.join(downloaded_tickers)}")
    else:
        print("\n No data was successfully collected.")