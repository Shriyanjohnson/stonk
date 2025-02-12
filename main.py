import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import pandas as pd
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Custom OBV Implementation
def custom_on_balance_volume(df):
    obv = [0]  # initialize with 0 for the first row
    for i in range(1, len(df)):
        if df['Close'][i] > df['Close'][i - 1]:
            obv.append(obv[-1] + df['Volume'][i])  # price goes up, OBV increases
        elif df['Close'][i] < df['Close'][i - 1]:
            obv.append(obv[-1] - df['Volume'][i])  # price goes down, OBV decreases
        else:
            obv.append(obv[-1])  # price unchanged, OBV remains the same
    df['On_Balance_Volume'] = obv
    return df

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    
    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data = custom_on_balance_volume(data)  # Using the custom OBV function
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Handle NaN values
    data.dropna(inplace=True)
    return data

# Function to fetch sentiment score
@st.cache_data
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

# Function to fetch earnings data
@st.cache_data
def fetch_earnings_data(symbol):
    stock = yf.Ticker(symbol)
    earnings_data = stock.earnings_reports
    if earnings_data.empty:
        return None
    return earnings_data

# Function to analyze earnings impact and generate numerical features
def analyze_earnings_impact(earnings_data, stock_data):
    if earnings_data is None or earnings_data.empty:
        return 0, 0  # No earnings impact features available
    
    latest_earnings = earnings_data.iloc[-1]
    actual_earnings = latest_earnings['Actual EPS']
    expected_earnings = latest_earnings['Expected EPS']
    
    # Earnings surprise (Actual - Expected)
    earnings_surprise = actual_earnings - expected_earnings
    
    earnings_date = latest_earnings['Earnings Date']
    
    # Calculate price reaction after earnings (percentage change in stock price)
    if earnings_date in stock_data.index:
        price_reaction = stock_data.loc[earnings_date:]['Close'].pct_change().mean() * 100  # % change in price after earnings
    else:
        price_reaction = 0  # If earnings date is not within the stock data range

    return earnings_surprise, price_reaction

# Train Model with earnings features included
@st.cache_resource
def train_model(data, earnings_surprise, price_reaction):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)

    # Add earnings features to the dataset
    data['Earnings Surprise'] = earnings_surprise
    data['Price Reaction'] = price_reaction

    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'On_Balance_Volume', 'SMA_20', 'SMA_50', 'Earnings Surprise', 'Price Reaction']]
    labels = data['Target']
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # GridSearchCV for tuning model parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_score_ * 100, X_test, y_test

# Generate recommendation with earnings features
def generate_recommendation(data, sentiment_score, model, earnings_surprise, price_reaction):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['On_Balance_Volume'], latest_data['SMA_20'], latest_data['SMA_50'], earnings_surprise, price_reaction]])

    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date, latest_data

# Streamlit UI
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    earnings_data = fetch_earnings_data(symbol)
    
    earnings_surprise, price_reaction = analyze_earnings_impact(earnings_data, stock_data)
    
    model, accuracy, X_test, y_test = train_model(stock_data, earnings_surprise, price_reaction)

    current_price = stock_data['Close'].iloc[-1]

    st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

    # Generate the option recommendation
    option, strike_price, expiration, latest_data = generate_recommendation(stock_data, sentiment_score, model, earnings_surprise, price_reaction)

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
    
    # Displaying Earnings Data and Analysis
    st.subheader("Earnings Report Impact")
    if earnings_data is not None and not earnings_data.empty:
        st.write(f"**Latest Earnings Surprise**: {earnings_surprise:.2f}")
        st.write(f"**Price Reaction After Earnings**: {price_reaction:.2f}%")
    else:
        st.write("No recent earnings data available.")

    # Displaying Technical Indicators
    st.subheader('Technical Indicators and Their Current Values')
    st.write(f"**RSI** (Relative Strength Index): {latest_data['RSI']:.2f} - An RSI above 70 indicates overbought conditions, while below 30 indicates oversold.")
    st.write(f"**MACD** (Moving Average Convergence Divergence): {latest_data['MACD']:.2f} - Indicates momentum. Positive values suggest upward momentum.")
    st.write(f"**On-Balance Volume (OBV)**: {latest_data['On_Balance_Volume']:.2f} - Shows the cumulative buying and selling pressure.")
    st.write(f"**SMA-20** (Simple Moving Average - 20 days): {latest_data['SMA_20']:.2f}")
    st.write(f"**SMA-50** (Simple Moving Average - 50 days): {latest_data['SMA_50']:.2f}")
    st.write(f"**Volatility** (Average True Range): {latest_data['Volatility']:.2f} - A measure of price fluctuations over a given period.")

    # Visualizations
    st.subheader('Stock Data and Technical Indicators')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('Stock Price', 'RSI Indicator', 'MACD Indicator'))

    fig.add_trace(go.Candlestick(x=stock_data.index,
                open=stock_data['Open'], high=stock_data['High'],
                low=stock_data['Low'], close=stock_data['Close'], name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'), row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

