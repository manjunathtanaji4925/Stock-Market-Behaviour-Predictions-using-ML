import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  

st.title("📈 Stock Market Behaviour Prediction App")

# Predefined stock symbols
stock_options = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOG)": "GOOG",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS"
}

# Sidebar inputs
stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
stock_symbol = stock_options[stock_name]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# Load data
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df

data = load_data(stock_symbol, start_date, end_date)

# Show raw data
st.subheader(f"Raw Data for {stock_name}")
st.write(data.tail())

# Prepare data
data['Date'] = pd.to_datetime(data['Date'])
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']]
y = data['Close']

# --- Linear Regression Model ---
lr_model = LinearRegression()
lr_model.fit(X, y)
future_days = pd.DataFrame({'Days': np.arange(X['Days'].max() + 1, X['Days'].max() + 31)})
lr_preds = lr_model.predict(future_days)
lr_accuracy = r2_score(y, lr_model.predict(X)) * 100

# --- Random Forest Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_preds = rf_model.predict(future_days)
rf_accuracy = r2_score(y, rf_model.predict(X)) * 100

# --- Plotting ---
fig, ax = plt.subplots()
ax.plot(data['Date'], y, label="Actual Price",color="yellow")
future_dates = pd.date_range(data['Date'].max() + pd.Timedelta(days=1), periods=30)

# Predictions
ax.plot(future_dates, lr_preds, label="Linear Regression Prediction", linestyle='--',color="blue")
ax.plot(future_dates, rf_preds, label="Random Forest Prediction", linestyle='--',color="red")

# Currency symbol
if ".NS" in stock_symbol:  
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.0f}'))
else:  
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

# Final touches
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.subheader("📉 Future Price Prediction (Next 30 Days)")
st.pyplot(fig)

# --- Show accuracy comparison ---
st.markdown(f"### 🔎 Model Accuracies:")
st.write(f"**Linear Regression:** {lr_accuracy:.2f}%")
st.write(f"**Random Forest:** {rf_accuracy:.2f}%")



#python -m streamlit run app.py