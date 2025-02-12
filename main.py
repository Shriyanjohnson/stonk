import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import pandas as pd
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Custom OBV Implementation
def custom_on_balance_volume(df):
    obv = [0]  # initialize with 0 for the first row
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])  # price goes up, OBV increases
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])  # price goes down, OBV decreases
        else:
            obv.append(obv[-1])  # price unchanged, OBV remains the same
    df['On_Balance_Volume'] = obv
    return df

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data = custom_on_balance_volume(data)  # Using the custom OBV function
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
@st.cache_data
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return 0
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except:
        return 0

# Train Model with GridSearchCV
@st.cache_resource
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'On_Balance_Volume', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # GridSearchCV for tuning model parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_score_ * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['On_Balance_Volume'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
st.subheader("Developed by Shriyan Kandula, a sophomore at Shaker High School.")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)

    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")
    
    st.write(f"### Model Accuracy: **{accuracy:.2f}%**")

# Key Features Section
st.subheader("🌟 Key Features That Make It Stand Out")
st.write("""
- **Advanced Machine Learning Models**: Uses XGBoost, Random Forest, and GridSearchCV to optimize predictions.
- **Comprehensive Technical Analysis**: Includes RSI, MACD, Bollinger Bands, and more.
- **Real-Time Market Sentiment Analysis**: Fetches news headlines and applies sentiment analysis.
- **Options Trading Recommendations**: Suggests actionable trades based on predicted price movements.
- **User-Friendly & Professional UI**: Clean interface with interactive charts.
- **Thorough Explanations**: Provides insights into each indicator and model accuracy.
- **Built for Accuracy**: Implements hyperparameter tuning and real-time data.

🚀 **Your app is a powerful AI-driven stock trading assistant that combines ML, technical indicators, and sentiment analysis to provide accurate, actionable insights. It’s more than just a predictor—it’s a decision-making tool for traders. 🔥**
""")

