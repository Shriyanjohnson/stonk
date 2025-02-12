import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolume
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# Custom HTML & CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #ecf0f1;
            font-size: 36px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subtitle {
            text-align: center;
            color: #bdc3c7;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .current-price, .recommendation {
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            color: #2c3e50;
        }
        .recommendation h3 {
            color: #e74c3c;
            font-size: 22px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['OBV'] = OnBalanceVolume(data['Close'], data['Volume']).on_balance_volume()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
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

# Train Model with Cross-Validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'OBV', 'Volatility', 'SMA_20', 'SMA_50']]
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
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['OBV'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
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

    # Displaying technical indicators and their descriptions
    st.subheader("Technical Indicators")
    
    # RSI Indicator
    st.write(f"**RSI (Relative Strength Index)**: This measures the speed and change of price movements. RSI above 70 indicates the stock is overbought, and below 30 indicates it is oversold. **Current RSI: {stock_data['RSI'].iloc[-1]:.2f}**")
    
    # MACD Indicator
    st.write(f"**MACD (Moving Average Convergence Divergence)**: This measures the difference between two moving averages to identify potential buy or sell signals. **Current MACD: {stock_data['MACD'].iloc[-1]:.2f}**")
    
    # OBV Indicator
    st.write(f"**On-Balance Volume (OBV)**: This tracks buying and selling pressure. An increasing OBV indicates buying pressure, and a decreasing OBV suggests selling pressure. **Current OBV: {stock_data['OBV'].iloc[-1]:.2f}**")
    
    # Volatility
    st.write(f"**Volatility**: This measures price fluctuations over a period of time. High volatility indicates greater risk and more potential for larger price changes. **Current Volatility: {stock_data['Volatility'].iloc[-1]:.4f}**")
    
    # SMA Indicators
    st.write(f"**SMA-20 (20-Day Simple Moving Average)**: A short-term trend indicator. **Current SMA-20: {stock_data['SMA_20'].iloc[-1]:.2f}**")
    st.write(f"**SMA-50 (50-Day Simple Moving Average)**: A longer-term trend indicator. **Current SMA-50: {stock_data['SMA_50'].iloc[-1]:.2f}**")
    
    # Sentiment Analysis
    st.write(f"**Sentiment Analysis**: Sentiment from news articles about the stock. Positive sentiment increases the likelihood of a Call option. **Current Sentiment Score: {sentiment_score:.2f}**")
    
    # Model Accuracy
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

    # Displaying the recommendation
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
    
    st.markdown(f"""
        <div class="recommendation">
            <h3>Option Recommendation: **{option}**</h3>
            <p>Strike Price: **${strike_price}**</p>
            <p>Expiration Date: **{expiration}**</p>
        </div>
    """, unsafe_allow_html=True)

    # Allow user to download the stock data
    st.download_button(
        label="Download Stock Data",
        data=stock_data.to_csv(),
        file_name=f"{symbol}_stock_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
