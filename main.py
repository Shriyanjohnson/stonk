import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import io
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# Set API Key
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Store this securely

# Custom On-Balance Volume (OBV) function
def custom_on_balance_volume(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

# Fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = data['Close'].pct_change().rolling(14).mean()
    data['ATR'] = (data['High'] - data['Low']).rolling(14).mean()
    data = custom_on_balance_volume(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

# Fetch real-time stock price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d", interval="1m")
    return real_time_data['Close'][-1]

# Fetch EPS and earnings report
def fetch_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    eps = stock.info.get("trailingEps", None)
    earnings = stock.quarterly_earnings
    return eps, earnings

# Train Machine Learning Model
@st.cache_resource
def train_model(data, eps):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    if eps:
        data['EPS'] = eps  # Add EPS as a constant column
        features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'EPS']]
    
    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test) * 100

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    eps, earnings = fetch_fundamentals(symbol)
    model, accuracy = train_model(stock_data, eps)
    real_time_price = fetch_real_time_price(symbol)

    st.subheader(f"📈 Stock Data for {symbol}")
    st.write(f"### Real-Time Price: **${real_time_price:.2f}**")
    if eps:
        st.write(f"### EPS (Earnings Per Share): **${eps:.2f}**")
    else:
        st.write("### EPS: Not available")
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")

    # Charts
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "RSI", "OBV"))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['OBV'], mode='lines', name='OBV'), row=3, col=1)
    st.plotly_chart(fig)

    # Earnings Report Table
    if earnings is not None:
        st.write("### Earnings Report")
        st.dataframe(earnings)
    else:
        st.write("No earnings report available.")
