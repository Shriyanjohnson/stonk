import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolume
from ta.trend import ADXIndicator
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    data['OBV'] = OnBalanceVolume(data['Close'], data['Volume']).on_balance_volume()
    data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    
    # Drop NaN values
    data.dropna(inplace=True)
    
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
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return 0

# Function to train model using ensemble stacking
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50', 'OBV', 'ADX']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # TimeSeries Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
    
    # Base models for stacking
    base_models = [
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', random_state=42))
    ]
    
    # Stacked model
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
    
    # Train the stacked model
    stacked_model.fit(X_train, y_train)
    
    return stacked_model, X_test, y_test

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50'], latest_data['OBV'], latest_data['ADX']]])
    
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    # Adjust based on sentiment score
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
    model, X_test, y_test = train_model(stock_data)

    current_price = stock_data['Close'].iloc[-1]
    
    # Display Current Price
    st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

    # Display recommendation
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)
    st.markdown(f"""
        <div class="recommendation">
            <h3>Option Recommendation: **{option}**</h3>
            <p>Strike Price: **${strike_price}**</p>
            <p>Expiration Date: **{expiration}**</p>
        </div>
    """, unsafe_allow_html=True)

    # Display model performance
    accuracy = model.score(X_test, y_test) * 100
    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    
    # Downloadable data
    st.download_button(
        label="Download Data",
        data=stock_data.to_csv(index=False),
        file_name=f"{symbol}_stock_data.csv",
        mime="text/csv"
    )

    # Plotting technical indicators
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Price & SMAs
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], mode='lines', name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)

    st.plotly_chart(fig)

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
