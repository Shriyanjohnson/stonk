import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta  # Using pandas_ta for technical indicators
import yfinance as yf
import requests
from datetime import datetime, timedelta

# API key for news
NEWS_API_KEY = '833b7f0c6c7243b6b751715b243e4802'  # Replace with your actual API key

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y', interval='1d')
    stock_data['RSI'] = ta.rsi(stock_data['Close'], length=14)
    stock_data['ATR'] = ta.atr(stock_data['High'], stock_data['Low'], stock_data['Close'], length=14)
    stock_data['OBV'] = ta.obv(stock_data['Close'], stock_data['Volume'])
    stock_data['SMA_20'] = ta.sma(stock_data['Close'], length=20)
    stock_data['SMA_50'] = ta.sma(stock_data['Close'], length=50)
    stock_data['Earnings'] = get_earnings(ticker)  # Fetch earnings data
    return stock_data.dropna()  # Drop rows with missing values

# Function to get earnings data
def get_earnings(ticker):
    url = f'https://financialmodelingprep.com/api/v3/earnings_calendar?symbol={ticker}&apikey={NEWS_API_KEY}'
    response = requests.get(url).json()
    if response:
        return response[0]['epsEstimated']  # Estimated EPS
    return 0  # Return 0 if earnings data is not found

# Function to train or update the model
def train_or_update_model(stock_data, model=None):
    features = stock_data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'Earnings']]
    labels = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)  # Predict if price goes up

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train the model if it's not already trained
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features_scaled, labels)  # Train the model
    else:
        model.partial_fit(features_scaled, labels)  # Update the model incrementally

    return model, features_scaled, labels

# Function to predict with the model
def predict(model, stock_data):
    features = stock_data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'Earnings']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    prediction = model.predict(features_scaled[-1].reshape(1, -1))  # Use last data point for prediction
    return 'Buy' if prediction == 1 else 'Sell'

# Streamlit app layout
st.title('Stock Prediction App')

ticker = st.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')
stock_data = fetch_stock_data(ticker)

st.subheader('Stock Data')
st.write(stock_data.tail())

# Train or update the model
model, X_test, y_test = train_or_update_model(stock_data)

# Predict the next action
prediction = predict(model, stock_data)
st.subheader(f'Prediction: {prediction}')

# Show news headlines related to the stock
st.subheader('Related News')
news = get_news(ticker)
for article in news:
    st.write(f"**{article['title']}**")
    st.write(f"[Read more]({article['url']})")

# Function to fetch related news
def get_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}'
    response = requests.get(url).json()
    return response['articles'][:5]  # Return the top 5 news articles

# Explain the model and features
st.subheader('Model Explanation')
st.write("""
    The Random Forest Classifier was used for predicting stock price movements. 
    It uses a set of technical indicators like RSI, ATR, OBV, and SMAs, as well as 
    earnings data to make predictions. This allows the model to consider both 
    historical price data and market sentiment (through earnings and news).
""")
