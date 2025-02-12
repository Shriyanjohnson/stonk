import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import plotly.graph_objects as go

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score from NewsAPI
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("newsapi_key")
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0  
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception:
        return 0  

# Function to train machine learning model with cross-validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    accuracy = cv_scores.mean() * 100
    model.fit(X_train, y_train)
    return model, accuracy, X_test, y_test

# Streamlit UI with enhanced layout
st.set_page_config(page_title="AI Stock Options Predictor", layout="wide")
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select a Section", ("Stock Info", "Recommendations", "Charts"))

if tab == "Stock Info":
    st.title("📈 Stock Data")
    symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    if symbol:
        st.sidebar.markdown("### Stock Data")
        stock_data = fetch_stock_data(symbol)
        st.dataframe(stock_data.tail())

elif tab == "Recommendations":
    st.title("💡 Option Recommendations")
    if symbol:
        sentiment_score = fetch_sentiment(symbol)
        model, accuracy, X_test, y_test = train_model(stock_data)
        st.write(f"Model Accuracy: {accuracy}%")
        st.write(f"Sentiment Score: {sentiment_score}")
        option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
        st.write(f"Recommendation: **{option}** with strike price **{strike_price}**")

elif tab == "Charts":
    st.title("📊 Stock Price Charts")
    if symbol:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close']))
        st.plotly_chart(fig)
    
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
