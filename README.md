# Data Forecasting Project

An end-to-end machine learning pipeline for **stock trend analysis** and **price forecasting**.  
This project automates data collection, technical analysis visualization, and multi-day forecasting using Linear Regression and time-series features.

---

## Overview

The **Data Forecasting Project** fetches financial market data (using Yahoo Finance), processes it for modeling, and produces analytical visuals that combine:
- Moving averages  
- Fibonacci retracements  
- Volatility highlights  
- Multi-day forecast extensions  

The system outputs professional-grade charts and predictive insights — ideal for learning, portfolio building, or lightweight quantitative research.

---

## Features

**Automated Data Collection** — Fetches data directly from Yahoo Finance.  
**Feature Engineering** — Computes SMAs, returns, volatility, and trend features.  
**Trend Visualization** — Plots moving averages, Fibonacci levels, and volatility.  
**Forecasting Module** — Predicts user-defined future days using trained ML models.  
**Model Persistence** — Saves trained model (`.pkl`) and scaler for reuse.  
**Organized Outputs** — Saves all visuals and models into structured folders.

---

## Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/yourusername/DataForecasting.git
cd DataForecasting
pip install -r requirements.txt
```

## Usuage

- Run the main program:

When prompted:

- Enter ticker(s):

Enter tickers (e.g., AAPL, NVDA, BTC-USD): NVDA


- Choose forecast horizon:

Enter number of days to forecast (e.g., 10): 10


- The program will:

**Download market data**

**Generate technical analysis charts**

**Train a regression model**

**Extend the chart with future forecasts**

View your results:

- Trend & analysis visuals → visuals/technical_analysis/

- Forecast charts → visuals/predictions/

- Trained models → models/


## Project Structure
DataForecasting/
│
├── main.py
├── requirements.txt
├── data/                      # Raw and processed stock data
├── models/                    # Trained ML models & scalers
│    ├── linearRegressionModel.py 
│    └── visualizationCharts.py         
├── visuals/
│   ├── technical_analysis/    # Trend charts (SMA, Fib, volatility)
│   └── predictions/           # Forecast extension charts
└── src/
    ├── dataCollection.py      # Data fetching 
    ├── dataPreprocessing.py   # Feature generation
    └── trendAnalysis.py
    
## Author
**Avishek Dangol**
- [dangolavishek202@gmail.com]
- [www.linkedin.com/in/avishek-dangol]
