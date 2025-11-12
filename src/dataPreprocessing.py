import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

def create_features(ticker, file_path):

    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print(f"--- Processing {ticker} (Initial rows: {len(df)})")

    df.dropna(inplace=True)
    
    df['Return'] = df['Close'].pct_change()
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_90'] = df['Close'].rolling(window=90).mean()
    df['Vol_30'] = df['Return'].rolling(window=30).std() * np.sqrt(252) 

    df.dropna(inplace=True)
    
    df.to_csv(file_path)
    
    print(f"Features created and saved for {ticker}. Final rows: {len(df)}")
    
    return df

if __name__ == "__main__":
    processed_count = 0

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "")
            file_path = os.path.join(DATA_DIR, filename)

            df_features = create_features(ticker, file_path)
            processed_count += 1

    print(f"\nBatch Processing Complete: {processed_count} files processed.")
