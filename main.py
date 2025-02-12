import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import requests
from textblob import TextBlob

# NewsAPI Key (replace with your own)
NEWSAPI_KEY = "833b7f0c6c7243b6b751715b243e4802"

# Fetching news and sentiment for the stock
def fetch_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    articles = response.json()['articles']
    
    sentiment_score = 0
    for article in articles:
        blob = TextBlob(article['title'])
        sentiment_score += blob.sentiment.polarity  # Sum of polarity (positive/negative)
    
    return sentiment_score / len(articles) if articles else 0

# Custom On-Balance Volume (OBV) function
def custom_on_balance_volume(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

# Save the trained model
def save_model(model, filename="stock_model.pkl"):
    joblib.dump(model, filename)

# Load the trained model
def load_model(filename="stock_model.pkl"):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

# Fetch stock data
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

# Fetch real-time stock price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d", interval="1m")
    return real_time_data['Close'][-1]

# Train or update the model
def train_or_update_model(data, model=None):
    # Prepare the new data for training
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # If no model exists, initialize a new model
    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(features_scaled, labels)
    else:
        model.fit(features_scaled, labels)  # For RandomForest, it will retrain with all data each time

    save_model(model)  # Save the updated model

    accuracy = model.score(features_scaled, labels) * 100
    return model, accuracy

# Generate options prediction
def generate_options_prediction(real_time_price, model, features):
    prediction = model.predict(features)
    if prediction == 1:
        predicted_movement = "Up"
        strike_price = real_time_price * 1.05  # Strike price 5% above current price
        option_type = "Call"
    else:
        predicted_movement = "Down"
        strike_price = real_time_price * 0.95  # Strike price 5% below current price
        option_type = "Put"

    # Find the nearest Friday
    today = datetime.date.today()
    days_to_friday = (4 - today.weekday()) % 7
    expiration_date = today + datetime.timedelta(days=days_to_friday)

    return predicted_movement, strike_price, expiration_date, option_type

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    real_time_price = fetch_real_time_price(symbol)
    
    # Load the existing model and update it
    model = load_model()
    model, accuracy = train_or_update_model(stock_data, model)
    
    sentiment_score = fetch_sentiment(symbol)  # Fetch Sentiment

    st.subheader(f"📈 Stock Data for {symbol}")
    st.write(f"### Real-Time Price: **${real_time_price:.2f}**")
    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    st.write(f"### Sentiment Score: **{sentiment_score:.2f}**")

    # Generate options prediction
    features = stock_data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    predicted_movement, strike_price, expiration_date, option_type = generate_options_prediction(real_time_price, model, features)

    st.write(f"### Predicted Price Movement: **{predicted_movement}**")
    st.write(f"### Suggested Option: **{option_type}**")
    st.write(f"### Strike Price: **${strike_price:.2f}**")
    st.write(f"### Expiration Date: **{expiration_date}**")

    # Plot stock data
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "RSI", "OBV"))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['OBV'], mode='lines', name='OBV'), row=3, col=1)
    st.plotly_chart(fig)
