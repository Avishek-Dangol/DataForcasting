# main.py
import os
from src import data_collection
from src import data_preprocessing
from src import trend_analysis
from models import LinearRegressionModel

def main():
    print("Starting Independent Data Forecasting Project...\n")

    print("Step 1: Data Collection")
    data_collection.main() 

    print("Step 2: Data Preprocessing")
    data_preprocessing.main() 

    print("Step 3: Trend Analysis / EDA")
    trend_analysis.trend_analysis()

    print("Step 4: Linear Regression Model Training & Prediction")
    LinearRegressionModel.run_all_models_on_data()

if __name__ == "__main__":
    main()
