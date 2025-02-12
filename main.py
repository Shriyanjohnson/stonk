import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit UI Title
st.title("📈 AI Stock Options Predictor")

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
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
        return sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles) if articles else 0
    except:
        return 0

# Function to train model
def train_model(data):
    data['Target'] = np.where(data['Close'].diff().shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'EMA_20', 'EMA_50']]
    labels = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test) * 100

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['EMA_20'], latest_data['EMA_50']]])
    option = "Call" if model.predict(latest_features)[0] == 1 else "Put"
    if sentiment_score > 0.2 and option == "Put": option = "Call"
    elif sentiment_score < -0.2 and option == "Call": option = "Put"
    return option, round(latest_data['Close'] / 10) * 10, (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

# Stock input
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, test_accuracy = train_model(stock_data)
    current_price = stock_data['Close'].iloc[-1]
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
    
    st.metric("Current Price", f"${current_price:.2f}")
    st.markdown(f"**Option Recommendation:** {option}\n\n**Strike Price:** ${strike_price}\n\n**Expiration Date:** {expiration}")
    st.metric("Model Accuracy", f"{test_accuracy:.2f}%")
    
    # Plot stock data
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"{symbol} Stock Price", "RSI", "MACD"))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], name='Market Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='purple')), row=3, col=1)
    fig.update_layout(title=f"{symbol} Stock Data", template="plotly_dark")
    st.plotly_chart(fig)
    
    # Download option
    csv = stock_data.to_csv(index=True)
    st.download_button("Download Data", csv, file_name=f"{symbol}_data.csv", mime="text/csv")
    
    st.markdown("**Disclaimer:** This app is for educational purposes only.")
