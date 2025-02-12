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
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

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
    data = stock.history(period="1y")  # Increased period to 1 year
    
    if len(data) < 30:
        return None  # Return None if not enough data
    
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
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return 0  # Return neutral sentiment if no key is provided
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5).get('articles', [])
        if not articles:
            return 0
        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    model.fit(X_train, y_train)
    return model, cv_scores.mean() * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
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
    if stock_data is None:
        st.error("Not enough data to proceed with prediction. Please check the stock symbol or try again later.")
    else:
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
        
        # Visualizing Stock Data and Indicators in Charts
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1, subplot_titles=('Stock Price & EMAs', 'Volume & OBV', 'RSI & Sentiment'))

        # Stock Price and EMAs
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], mode='lines', name='20-Day SMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='50-Day SMA'), row=1, col=1)

        # Volume and OBV
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'].cumsum(), mode='lines', name='OBV'), row=2, col=1)

        # RSI and Sentiment
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=[sentiment_score] * len(stock_data), mode='lines', name='Sentiment', line=dict(dash='dash')), row=3, col=1)

        fig.update_layout(title=f"{symbol} Stock Data and Indicators", height=900)
        st.plotly_chart(fig)

# Footer Information
st.markdown('<div class="footer">Made with ❤️ by Shriyan Kandula for educational purposes.</div>', unsafe_allow_html=True)
