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
import requests  # To fetch all available tickers dynamically

# Set API Key directly
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Ensure you handle the API key securely

# Function to fetch stock data
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

# Function to fetch all tickers from an exchange (e.g., NASDAQ)
def fetch_all_tickers():
    url = "https://query1.finance.yahoo.com/v7/finance/screener"
    params = {
        "region": "US",
        "lang": "en",
        "count": 5000,  # Change this count based on how many tickers you want to load at once
    }
    response = requests.get(url, params=params)
    data = response.json()
    tickers = [stock['symbol'] for stock in data['finance']['result'][0]['quotes']]
    return tickers

# Function to filter out stocks based on market criteria
def filter_trending_stocks(tickers):
    trending_stocks = []

    for symbol in tickers:
        try:
            stock_data = fetch_stock_data(symbol)
            recent_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-7]) / stock_data['Close'].iloc[-7]  # 7-day price change
            if recent_change > 0.05:  # 5% increase is considered an uptrend
                trending_stocks.append((symbol, 'Uptrend', recent_change))
            elif recent_change < -0.05:  # 5% decrease is considered a downtrend
                trending_stocks.append((symbol, 'Downtrend', recent_change))
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            continue
    
    trending_stocks.sort(key=lambda x: abs(x[2]), reverse=True)  # Sort by the strongest trend
    return trending_stocks[:5]  # Show the top 5 trending stocks

# Function to check if ticker is valid
def is_valid_ticker(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d") is not None
    except:
        return False

# Function to fetch sentiment from news API
def fetch_sentiment(symbol):
    newsapi = NewsApiClient(api_key=API_KEY)
    try:
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page_size=5)
        sentiment_score = 0
        for article in articles['articles']:
            text = article['title'] + " " + article['description']
            sentiment_score += TextBlob(text).sentiment.polarity
        return sentiment_score / len(articles['articles']) if len(articles['articles']) > 0 else 0
    except Exception as e:
        st.error(f"Error fetching news sentiment for {symbol}: {e}")
        return 0

# Function to generate stock options recommendation
def generate_recommendation(stock_data, sentiment_score, model, symbol):
    # Example placeholder logic for generating recommendations
    latest_price = stock_data['Close'].iloc[-1]
    strike_price = latest_price * 1.05 if sentiment_score > 0 else latest_price * 0.95
    expiration = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    option = 'Call' if sentiment_score > 0 else 'Put'
    return option, strike_price, expiration, stock_data.iloc[-1]

# Function to train model on stock data
def train_model(stock_data):
    X = stock_data[['RSI', 'ATR', 'SMA_20', 'SMA_50']]  # Example features
    y = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)  # Target: 1 if next day price is higher

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model accuracy
    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy, X_test, y_test

# Function to track model performance
def track_performance(model, X_test, y_test):
    accuracy = model.score(X_test, y_test) * 100
    return accuracy

# Function to fetch real-time price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d")
    return real_time_data['Close'].iloc[-1]

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰\n Made by Shriyan Kandula")

# Disclaimer message
st.write("**Note:** The predictions and recommendations made by this tool may not be entirely accurate and should be considered as one of many factors when making investment decisions.")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    # Check if symbol is valid
    if is_valid_ticker(symbol):
        stock_data = fetch_stock_data(symbol)
        sentiment_score = fetch_sentiment(symbol)
        model, accuracy, X_test, y_test = train_model(stock_data)
        option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, symbol)

        # Performance Metrics: Tracking model performance on test data
        test_accuracy = track_performance(model, X_test, y_test)

        # Fetch and display the real-time stock price
        real_time_price = fetch_real_time_price(symbol)

        st.subheader(f"📈 Option Recommendation for {symbol}")
        st.write(f"**Recommended Option:** {option}")
        st.write(f"**Strike Price:** ${strike_price}")
        st.write(f"**Expiration Date:** {expiration}")
        st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
        st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
        st.write(f"### Real-Time Price: **${real_time_price:.2f}**")

        st.download_button("Download Stock Data", data=stock_data.to_csv(index=True), file_name=f"{symbol}_stock_data.csv", mime="text/csv")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
        fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                     low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
        st.plotly_chart(fig)

        # Displaying the indicator values and their significance
        st.subheader("📊 Indicator Analysis")
        st.write(f"**RSI (Relative Strength Index):** {latest_data['RSI']:.2f} - RSI values above 70 may suggest an overbought condition (uptrend possible reversal), and below 30 may indicate an oversold condition (uptrend likely).")
        st.write(f"**ATR (Average True Range):** {latest_data['ATR']:.2f} - A high ATR value suggests higher volatility, potentially signaling larger price movements.")
        st.write(f"**SMA_20 (20-period Simple Moving Average):** {latest_data['SMA_20']:.2f} - A rising SMA_20 could indicate an uptrend, while a falling SMA_20 could suggest a downtrend.")
        st.write(f"**SMA_50 (50-period Simple Moving Average):** {latest_data['SMA_50']:.2f} - A crossover above the 50-day SMA could indicate bullish movement (uptrend).")
        st.write(f"**OBV (On-Balance Volume):** {latest_data['OBV']:.2f} - A rising OBV suggests increasing buying pressure (uptrend), while a falling OBV signals a downtrend.")

    else:
        st.error(f"Invalid ticker symbol: {symbol.upper()}. Please check the symbol and try again.")
        suggestions = suggest_valid_symbols(symbol)
        if suggestions:
            st.write(f"Did you mean one of these? {', '.join(suggestions)}")
        else:
            st.write("No close matches found.")

# New Feature: Trending Stocks
st.subheader("🚀 Trending Stocks")
tickers = fetch_all_tickers()  # Fetch all tickers dynamically
trending_stocks = filter_trending_stocks(tickers)
for stock in trending_stocks:
    st.write(f"**{stock[0]}**: {stock[1]} ({stock[2]*100:.2f}%)")
    stock_data = fetch_stock_data(stock[0])
    sentiment_score = fetch_sentiment(stock[0])
    model, accuracy, X_test, y_test = train_model(stock_data)
    option, strike_price, expiration, _ = generate_recommendation(stock_data, sentiment_score, model, stock[0])
    st.write(f"  - **Recommended Option:** {option} | **Strike Price:** ${strike_price} | **Expiration Date:** {expiration}")
