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
import xgboost as xgb
from textblob import TextBlob
from newsapi import NewsApiClient

# Set API Key (ensure to keep this secure)
API_KEY = "YOUR_NEWS_API_KEY"

# News API for sentiment analysis
newsapi = NewsApiClient(api_key=API_KEY)

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

# Fetch real-time stock price
def fetch_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    real_time_data = stock.history(period="1d", interval="1m")
    return real_time_data['Close'][-1]

# Fetch stock fundamentals (EPS)
def fetch_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    financials = stock.financials
    eps = financials.loc['Earnings Per Share'][0]  # Get the EPS value
    return eps

# Sentiment analysis from news
def fetch_sentiment(symbol):
    articles = newsapi.get_everything(q=symbol, language="en", sort_by="publishedAt", page_size=5)
    headlines = [article['title'] for article in articles['articles']]
    sentiment_score = 0
    for headline in headlines:
        sentiment_score += TextBlob(headline).sentiment.polarity
    return sentiment_score

# Option Pricing using Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Generate options prediction
def generate_options_prediction(real_time_price, model, features, volatility, expiration_days):
    prediction = model.predict(features)
    
    # Choose option type based on prediction
    option_type = 'call' if prediction == 1 else 'put'

    # Strike price based on prediction and volatility
    strike_price = real_time_price * (1 + (volatility * np.random.uniform(0.02, 0.1)))  # Random volatility effect

    # Expiration date (Friday of current week)
    today = datetime.date.today()
    days_until_friday = (4 - today.weekday()) % 7  # Days until Friday
    expiration_date = today + datetime.timedelta(days=days_until_friday)
    
    # Option price using Black-Scholes Model
    T = (expiration_date - today).days / 365.0  # Time to expiration in years
    r = 0.01  # Assumed risk-free rate (can be adjusted)
    sigma = volatility  # Current volatility (historically calculated)
    strike_price = round(strike_price, 2)

    option_price = black_scholes(real_time_price, strike_price, T, r, sigma, option_type)
    
    return option_type, strike_price, expiration_date, option_price

# Train or update the model
def train_or_update_model(data, eps, model=None):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    
    if eps:
        data['EPS'] = eps
        features = data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 'EPS']]
    
    labels = data['Target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if model is None:
        model = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(features_scaled, labels)
    else:
        model.fit(features_scaled, labels)

    joblib.dump(model, 'stock_model_xgb.pkl')
    
    accuracy = model.score(features_scaled, labels) * 100
    return model, accuracy

# Streamlit UI
st.set_page_config(page_title="💰 AI Stock Options Predictor 💰", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("💰 AI Stock Options Predictor 💰")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    eps = fetch_fundamentals(symbol)  # Fetch EPS
    sentiment_score = fetch_sentiment(symbol)  # Fetch Sentiment

    # Train or update model
    model = load_model()  # Load model
    model, accuracy = train_or_update_model(stock_data, eps, model)

    real_time_price = fetch_real_time_price(symbol)
    volatility = stock_data['ATR'].mean()  # Historical volatility (can refine this)

    option_type, strike_price, expiration_date, option_price = generate_options_prediction(
        real_time_price, model, stock_data[['Close', 'RSI', 'ATR', 'OBV', 'SMA_20', 'SMA_50']], volatility, expiration_days=7
    )

    # Display Results
    st.subheader(f"📈 Stock Data for {symbol}")
    st.write(f"Real-Time Price: **${real_time_price:.2f}**")
    st.write(f"Predicted Option Type: **{option_type.capitalize()}**")
    st.write(f"Strike Price: **${strike_price:.2f}**")
    st.write(f"Expiration Date: **{expiration_date.strftime('%Y-%m-%d')}**")
    st.write(f"Option Price (Black-Scholes): **${option_price:.2f}**")
    st.write(f"Model Accuracy: **{accuracy:.2f}%**")
    st.write(f"Market Sentiment: **{sentiment_score:.2f}**")
    
    # Visualize stock price and technical indicators
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "RSI", "OBV"))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['OBV'], mode='lines', name='OBV'), row=3, col=1)
    st.plotly_chart(fig)
