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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

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
    
    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
def fetch_sentiment(symbol):
    try:
        api_key = "YOUR_NEWSAPI_KEY"
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except:
        return 0

# Train Model with Cross-Validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Generate "thinking" process and prediction
def generate_prediction_with_thinking(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    # Reasoning
    reasoning = f"Stock Prediction: The model suggests a **{option}** option because the following indicators are influencing the decision:\n"
    
    reasoning += f"- **RSI**: {latest_data['RSI']:.2f}, which indicates whether the stock is overbought or oversold. (Above 70 is overbought, below 30 is oversold)\n"
    reasoning += f"- **MACD**: {latest_data['MACD']:.2f}, a trend-following momentum indicator showing the relationship between two moving averages.\n"
    reasoning += f"- **Volatility**: {latest_data['Volatility']:.4f}, indicating how volatile the stock has been over the past 10 days.\n"
    reasoning += f"- **SMA-20 vs SMA-50**: The short-term and long-term moving averages suggest a trend.\n"

    # Modify prediction based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
        reasoning += "\nSentiment analysis suggests positive market sentiment, which may influence the decision to go with a Call option."
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
        reasoning += "\nSentiment analysis suggests negative market sentiment, which may influence the decision to go with a Put option."

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date, reasoning

# Streamlit UI
st.title("AI Stock Options Predictor with Thinking Process")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, X_test, y_test = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]
    st.write(f"Current Price of {symbol}: ${current_price:.2f}")

    option, strike_price, expiration, reasoning = generate_prediction_with_thinking(stock_data, sentiment_score, model)

    st.write(f"Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")

    st.write("### Thinking Process:")
    st.write(reasoning)

    # Accuracy Evaluation
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
