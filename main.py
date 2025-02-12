import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import datetime
from textblob import TextBlob
from newsapi import NewsApiClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom OBV Implementation (since 'ta.volume' import might be causing issues)
def custom_on_balance_volume(close_prices, volumes):
    obv = [0]  # Start with the first value as 0
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif close_prices[i] < close_prices[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])  # If no change in price, OBV remains the same
    return obv

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # OBV Custom Calculation
    data['OBV'] = custom_on_balance_volume(data['Close'], data['Volume'])

    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
def fetch_sentiment(symbol):
    try:
        api_key = "833b7f0c6c7243b6b751715b243e4802"  # Your NewsAPI Key
        if not api_key:
            return 0
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        return 0

# Train Model with Cross-Validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50', 'OBV']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    model.fit(X_train, y_train)
    return model, cv_scores.mean() * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50'], latest_data['OBV']]])
    
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
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]

    st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.markdown(f"""
        <div class="recommendation">
            <h3>Option Recommendation: **{option}**</h3>
            <p>Strike Price: **${strike_price}**</p>
            <p>Expiration Date: **{expiration}**</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

    # Explanation of Indicators
    st.markdown("### **RSI (Relative Strength Index)**")
    st.write("RSI measures the speed and change of price movements. Values above 70 suggest the asset is overbought, while values below 30 suggest it is oversold.")

    st.markdown("### **MACD (Moving Average Convergence Divergence)**")
    st.write("MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.")

    st.markdown("### **OBV (On-Balance Volume)**")
    st.write("OBV is a momentum indicator that uses volume flow to predict changes in stock price. Increasing OBV indicates buying pressure, while decreasing OBV indicates selling pressure.")

    st.markdown("### **Volatility**")
    st.write("Volatility measures the degree of variation in the stock's price over a period. High volatility suggests larger price swings, while low volatility indicates more stable pricing.")

    st.markdown("### **SMA-20 and SMA-50 (Simple Moving Averages)**")
    st.write("SMA-20 represents the short-term trend, while SMA-50 represents the long-term trend. When the SMA-20 crosses above the SMA-50, it signals a potential buying opportunity.")

    st.markdown("### **Sentiment Score**")
    st.write(f"The sentiment score of {symbol} based on recent news articles is: **{sentiment_score:.2f}**. Positive sentiment may indicate bullishness, while negative sentiment may indicate bearishness.")

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
