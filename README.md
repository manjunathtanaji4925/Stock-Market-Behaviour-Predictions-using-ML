# Stock Market Behaviour Prediction using Machine Learning

🚀 **Live Demo:** [Open the Streamlit App](https://stock-market-behaviour-predictions-using-ml-3qht4bpvggfwypxnl7.streamlit.app/)

## Project Overview

This project is a machine learning-based web application that predicts stock market behaviour using historical stock price data.

The application uses historical data obtained from Yahoo Finance and applies two machine learning algorithms:

- Linear Regression
- Random Forest Regressor

The predictions and historical stock prices are visualized through an interactive Streamlit web application.

## Features

- Select stocks from predefined options
- Select a custom historical date range
- Fetch historical stock market data using Yahoo Finance
- Display historical stock data
- Predict stock prices for the next 30 days
- Compare Linear Regression and Random Forest predictions
- Display R² scores for model comparison
- Visualize actual and predicted stock prices
- Support for both Indian and US stocks

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Yahoo Finance
- VS Code

## Machine Learning Algorithms

### Linear Regression

Linear Regression is used to identify and model the linear relationship between the number of days and stock closing prices.

### Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that uses multiple decision trees to model non-linear relationships in the historical stock data.

## Stocks Available

The application currently supports:

- Apple (AAPL)
- Microsoft (MSFT)
- Google (GOOG)
- Amazon (AMZN)
- Tesla (TSLA)
- Reliance (RELIANCE.NS)
- TCS (TCS.NS)
- Infosys (INFY.NS)

## Project Structure

```text
Stock-Market-Behaviour-Prediction/
│
├── app.py
├── requirements.txt
└── README.md
