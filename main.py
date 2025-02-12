import yfinance as yf
import streamlit as st
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import os

# Custom OBV Implementation
def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i - 1]:
            obv.append(obv[-1] - data['Volume'][i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data.index, name='OBV')

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Adding Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['OBV'] = calculate_obv(data)  # Using custom OBV function
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    return data

# Fetch sentiment score from news
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        return 0

# Train model with hyperparameter tuning
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'OBV', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    accuracy = grid_search.best_score_ * 100
    test_accuracy = model.score(X_test, y_test) * 100
    
    return model, accuracy, test_accuracy

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
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, training_accuracy, test_accuracy = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]
    st.write(f"Current Price of {symbol}: **${current_price:.2f}**")

    # Displaying technical indicators with explanations
    st.write(f"**RSI (Relative Strength Index)**: {stock_data['RSI'].iloc[-1]:.2f}")
    st.write("RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions. An RSI above 70 indicates overbought, below 30 indicates oversold.")

    st.write(f"**MACD (Moving Average Convergence Divergence)**: {stock_data['MACD'].iloc[-1]:.2f}")
    st.write("MACD is used to spot changes in the strength, direction, momentum, and duration of a trend. A positive MACD indicates upward momentum, and a negative MACD indicates downward momentum.")

    st.write(f"**OBV (On-Balance Volume)**: {stock_data['OBV'].iloc[-1]:.2f}")
    st.write("OBV uses volume flow to predict changes in stock price. An increasing OBV indicates buying pressure, while a decreasing OBV indicates selling pressure.")

    st.write(f"**Volatility**: {stock_data['Volatility'].iloc[-1]:.2f}")
    st.write("Volatility is the statistical measure of the dispersion of returns. High volatility often indicates a potential for larger price swings, while low volatility indicates a steadier price.")

    st.write(f"**SMA 20 (20-day Simple Moving Average)**: {stock_data['SMA_20'].iloc[-1]:.2f}")
    st.write("The 20-day SMA is used to smooth out short-term fluctuations and identify trends over the past month. A rising SMA 20 indicates an uptrend, while a falling SMA indicates a downtrend.")

    st.write(f"**SMA 50 (50-day Simple Moving Average)**: {stock_data['SMA_50'].iloc[-1]:.2f}")
    st.write("The 50-day SMA is a medium-term trend indicator. It helps identify the general direction of a stock price over the past 2 months.")

    # Option Recommendation
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
    st.write(f"Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")

    # Display model accuracy
    st.write(f"Model Accuracy (Training): **{training_accuracy:.2f}%**")
    st.write(f"Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")
    st.write("The model’s accuracy is derived from both training data (how well it performs on known data) and test data (how well it generalizes to new, unseen data).")

    # Adding download link for data (CSV)
    csv_data = stock_data.to_csv()
    st.download_button("Download Stock Data", csv_data, file_name=f"{symbol}_stock_data.csv", mime="text/csv")

    # Footer with your name and school
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    st.markdown("Created by: Shriyan Kandula, a sophomore at Shaker High School", unsafe_allow_html=True)

 
