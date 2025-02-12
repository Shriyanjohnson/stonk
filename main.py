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
    data = stock.history(period="90d")  # Last 90 days for better analysis
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch sentiment score from NewsAPI
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("newsapi_key")  # Get API key from environment variable
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0  # Default to neutral if no articles are found
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception:
        return 0  # Default to neutral if API fails

# Function to train machine learning model with cross-validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']  # Split the data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest model
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Cross-validation for better generalization
    accuracy = cv_scores.mean() * 100  # Cross-validation accuracy
    model.fit(X_train, y_train)
    return model, accuracy, X_test, y_test

# Function to generate option recommendation with sentiment adjustment
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]  # Probability of being a 'Call'
    option = "Call" if prediction_prob > 0.5 else "Put"
    # Adjust recommendation based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()  # Friday expiry
  
