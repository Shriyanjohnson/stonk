import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import numpy as np
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime

# Fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data.dropna(inplace=True)
    return data

# News sentiment analysis
def fetch_sentiment(symbol):
    api_key = "833b7f0c6c7243b6b751715b243e4802"
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
    articles = all_articles.get('articles', [])
    if not articles:
        return 0
    sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
    return sentiment_score

# Machine learning model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'MACD', 'Volatility']]
    labels = data['Target']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    return model
