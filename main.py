# main.py
from src.dataCollection import fetch_data
from src.dataPreprocessing import process_all_data
from src.trendAnalysis import trend_analysis
from models.linearRegressionModel import run_all_models_on_data
from datetime import date
from dateutil.relativedelta import relativedelta
import os

def main():
    print("\nData Forecasting Project\n")

    tickers = input("Enter tickers (e.g., AAPL, NVDA, BTC-USD): ").upper().split(',')
    tickers = [t.strip() for t in tickers if t.strip()]

    start_date = (date.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date = date.today().strftime("%Y-%m-%d")
    
    os.makedirs("data", exist_ok=True)

    for ticker in tickers:
        fetch_data(ticker, start=start_date, end=end_date)

    process_all_data()

    trend_analysis()

    run_all_models_on_data()

    print("\nForcasting completed!")

if __name__ == "__main__":
    main()
