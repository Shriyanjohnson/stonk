import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import difflib  # For finding the closest matching symbols

# Set API Key directly
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Ensure you handle the API key securely

# Improved OBV function
def custom_on_balance_volume(df):
    df['OBV'] = np.where(df['Close'].diff() > 0, df['Volume'],
                         np.where(df['Close'].diff() < 0, -df['Volume'], 0)).cumsum()
    return df

# Improved RSI Calculation
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Improved ATR Calculation
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = calculate_rsi(data)
    data['ATR'] = calculate_atr(data)
    data = custom_on_balance_volume(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

# Enhanced ML Model Training
def train_model(data):
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = ['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']
    X = StandardScaler().fit_transform(data[features])
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GridSearchCV(RandomForestClassifier(), {'n_estimators': [50, 100, 200]}, cv=5).fit(X_train, y_train).best_estimator_
    return model, model.score(X_test, y_test) * 100

# Enhanced Prediction Logic
def generate_recommendation(data, sentiment_score, model):
    latest_features = data.iloc[-1][['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']].values.reshape(1, -1)
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.6 else "Put"  # Adjusted threshold for better accuracy
    if sentiment_score > 0.3 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.3 and option == "Call":
        option = "Put"
    return option, round(data['Close'].iloc[-1] / 10) * 10, datetime.datetime.now().date()

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy = train_model(stock_data)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
    st.subheader(f"📈 {symbol} Option Recommendation")
    st.write(f"**Option:** {option}")
    st.write(f"**Strike Price:** ${strike_price}")
    st.write(f"**Expiration Date:** {expiration}")
    st.write(f"**Model Accuracy:** {accuracy:.2f}%")
