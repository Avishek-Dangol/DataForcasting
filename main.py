# main.py
import os
from src import dataCollection
from src import dataPreprocessing
from src import trendAnalysis
from models import linearRegressionModel

def main():
    print("Starting Independent Data Forecasting Project...\n")

    print("Step 1: Data Collection")
    dataCollection.main() 

    print("Step 2: Data Preprocessing")
    dataPreprocessing.main() 

    print("Step 3: Trend Analysis / EDA")
    trendAnalysis.trend_analysis()

    print("Step 4: Linear Regression Model Training & Prediction")
    linearRegressionModel.run_all_models_on_data()

if __name__ == "__main__":
    main()
