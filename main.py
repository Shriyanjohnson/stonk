import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set API Key for NewsAPI
API_KEY = "833b7f0c6c7243b6b751715b243e4802"  # Store this securely

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

# Fetch sentiment score from news articles
@st.cache_data
def fetch_sentiment(symbol):
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return 0

# Train or Update Machine Learning Model (SGDClassifier for incremental learning)
def train_or_update_model(data, model=None):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    labels = data['Target']

    # Check if data is empty or has NaN values
    if features.isnull().sum().any() or labels.isnull().sum().any():
        raise ValueError("Data contains NaN values, which cannot be processed.")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize the model if it doesn't exist
    if model is None:
        model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
        model.fit(features_scaled, labels)  # Initial fit on the data
    else:
        model.partial_fit(features_scaled, labels, classes=[0, 1])  # Update the model incrementally

    return model

# Generate Recommendation (Call/Put) based on Model Prediction
def generate_recommendation(data, sentiment_score, model, symbol):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['ATR'], latest_data['OBV'], latest_data['SMA_20'], latest_data['SMA_50']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"
    
    # Adjust based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    return option, strike_price, expiration_date, latest_data

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

# Initialize model variable
model = None

if symbol:
    try:
        stock_data = fetch_stock_data(symbol)
        sentiment_score = fetch_sentiment(symbol)

        # Update or train the model with the current stock data
        model = train_or_update_model(stock_data, model)

        # Generate option recommendation
        option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, symbol)

        # Fetch and display the real-time stock price
        real_time_price = stock_data['Close'].iloc[-1]

        st.subheader(f"📈 Option Recommendation for {symbol}")
        st.write(f"**Recommended Option:** {option}")
        st.write(f"**Strike Price:** ${strike_price}")
        st.write(f"**Expiration Date:** {expiration}")
        st.write(f"### 🔥 Real-Time Price: **${real_time_price:.2f}**")

        # Plot the stock price and RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI'))
        fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                     low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
