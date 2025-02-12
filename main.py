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
from sklearn.preprocessing import StandardScaler

# Custom HTML & CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #ecf0f1;
            font-size: 36px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subtitle {
            text-align: center;
            color: #bdc3c7;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .current-price, .recommendation {
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            color: #2c3e50;
        }
        .recommendation h3 {
            color: #e74c3c;
            font-size: 22px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

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
    
    # Try importing OnBalanceVolume, otherwise use custom implementation
    try:
        from ta.volume import OnBalanceVolume
        data['OBV'] = OnBalanceVolume(data['Close'], data['Volume']).on_balance_volume()
    except ImportError:
        st.write("OnBalanceVolume import failed. Using custom implementation.")
        data = calculate_obv(data)  # Use the custom OBV implementation
    
    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Custom OBV calculation (manual)
def calculate_obv(data):
    obv = [0]  # Start with the first value as zero
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            obv.append(obv[-1] + data['Volume'][i])  # Price went up, add volume
        elif data['Close'][i] < data['Close'][i - 1]:
            obv.append(obv[-1] - data['Volume'][i])  # Price went down, subtract volume
        else:
            obv.append(obv[-1])  # No price change, carry forward the previous OBV
    data['OBV'] = obv
    return data

# Function to fetch sentiment score
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return 0
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except:
        return 0

# Train Model with Cross-Validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50', 'OBV']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    model.fit(X_train, y_train)
    return model, cv_scores.mean() * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50'], latest_data['OBV']]])
    
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
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]

    st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.markdown(f"""
        <div class="recommendation">
            <h3>Option Recommendation: **{option}**</h3>
            <p>Strike Price: **${strike_price}**</p>
            <p>Expiration Date: **{expiration}**</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

    # Downloadable data
    csv = stock_data.to_csv(index=True)
    st.download_button(label="Download Stock Data", data=csv, file_name=f"{symbol}_data.csv", mime="text/csv")

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
