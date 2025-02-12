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
from plotly.subplots import make_subplots

st.title("💰 AI Stock Options Predictor 💰")
st.write("An AI-powered tool for predicting stock movement and generating options strategies.")

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="90d")
        data['RSI'] = RSIIndicator(data['Close']).rsi()
        data['MACD'] = MACD(data['Close']).macd()
        data['Volatility'] = data['Close'].pct_change().rolling(10).std()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to fetch sentiment score
def fetch_sentiment(symbol):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return 0
    try:
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        return np.mean([TextBlob(article['title']).sentiment.polarity for article in articles]) if articles else 0
    except Exception:
        return 0

# Train model
def train_model(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(data[['Close', 'RSI', 'MACD', 'Volatility', 'EMA_20', 'EMA_50']], 
                                                        data['Target'], test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean() * 100
    model.fit(X_train, y_train)
    return model, accuracy, X_test, y_test

# Generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1][['Close', 'RSI', 'MACD', 'Volatility', 'EMA_20', 'EMA_50']].values.reshape(1, -1)
    prediction_prob = model.predict_proba(latest_data)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    strike_price = round(data['Close'].iloc[-1] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    return option, strike_price, expiration_date

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    if stock_data is not None:
        sentiment_score = fetch_sentiment(symbol)
        model, accuracy, X_test, y_test = train_model(stock_data)
        current_price = stock_data['Close'].iloc[-1]
        option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
        
        st.metric(label=f"Current Price of {symbol}", value=f"${current_price:.2f}")
        st.subheader(f"Option Recommendation: {option}")
        st.write(f"Strike Price: ${strike_price}, Expiration Date: {expiration}")
        st.write(f"Model Accuracy (Cross-Validation): {accuracy:.2f}%")
        st.write(f"Model Test Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

        # Visualization
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            subplot_titles=("Stock Price", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                     low=stock_data['Low'], close=stock_data['Close'], name='Market Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='purple')), row=3, col=1)
        fig.update_layout(template="plotly_dark", height=700)
        st.plotly_chart(fig)
        
        csv = stock_data.to_csv(index=True)
        st.download_button("Download Stock Data (CSV)", data=csv, file_name=f"{symbol}_stock_data.csv", mime="text/csv")

st.write("*Disclaimer: This app is for educational purposes only and should not be considered financial advice.*")
