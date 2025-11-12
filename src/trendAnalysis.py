import os
import pandas as pd
from models.visualizationCharts import TrendAnalysisChart

DATA_DIR = "data"

def trend_analysis():
    tickers_processed = 0
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            ticker = filename.replace(".csv", "")
            
            chart = TrendAnalysisChart(df, ticker)
            chart.show()
            
            tickers_processed += 1
    print(f"\nTrend Analysis Complete: {tickers_processed} charts generated.")

if __name__ == "__main__":
    trend_analysis()
