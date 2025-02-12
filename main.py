import yfinance as yf
import streamlit as st
import pandas_ta as ta
import pandas as pd
import numpy as np
import datetime
import os
import io
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # Calculate Technical Indicators using pandas_ta
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data = custom_on_balance_volume(data)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    data.dropna(inplace=True)
    return data

# Fetch sentiment score from news articles
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

# Train Machine Learning Model
@st.cache_resource
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)

    features = data[['Close', 'RSI', 'MACD', 'ATR', 'OBV', 'SMA_20', 'SMA_50']]
    labels = data['Target']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_score_ * 100, X_test, y_test

# Option Recommendation Function
def generate_recommendation(data, sentiment_score, model, symbol):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['ATR'],
                                 latest_data['OBV'], latest_data['SMA_20'], latest_data['SMA_50']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

    analysis = ""
    if latest_data['RSI'] > 70:
        analysis = "The stock is overbought, suggesting a possible price pullback."
    elif latest_data['RSI'] < 30:
        analysis = "The stock is oversold, potentially signaling an upward movement."
    elif latest_data['MACD'] > 0:
        analysis = "Positive momentum detected, a potential upward trend ahead."
    elif latest_data['MACD'] < 0:
        analysis = "Negative momentum detected, possibly signaling a downward trend."

    return option, strike_price, expiration_date, latest_data, analysis

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)

    option, strike_price, expiration, latest_data, analysis = generate_recommendation(stock_data, sentiment_score, model, symbol)

    st.subheader(f"📈 Option Recommendation for {symbol}")
    st.write(f"**Recommended Option:** {option}")
    st.write(f"**Strike Price:** ${strike_price}")
    st.write(f"**Expiration Date:** {expiration}")
    st.write(f"**AI Analysis:** {analysis}")

    st.write(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

    # Explanation of Indicators
    st.subheader('📊 Technical Indicators Explained')
    st.write("""
    - **RSI (Relative Strength Index):** Measures overbought/oversold conditions (Above 70 = overbought, Below 30 = oversold).
    - **MACD (Moving Average Convergence Divergence):** Shows momentum direction; positive = uptrend, negative = downtrend.
    - **ATR (Average True Range):** Measures volatility; higher values indicate larger price swings.
    - **OBV (On-Balance Volume):** Tracks volume flow to predict price movement.
    - **SMA-20 & SMA-50 (Simple Moving Averages):** Help identify trends; when SMA-20 crosses above SMA-50, it's a bullish signal.
    """)

    # Downloadable Data
    csv = stock_data.to_csv(index=True)
    st.download_button("Download Stock Data", data=csv, file_name=f"{symbol}_stock_data.csv", mime="text/csv")

    # Visualizations
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Stock Price', 'RSI', 'MACD'))
    fig.add_trace(go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                                 low=stock_data['Low'], close=stock_data['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'), row=3, col=1)
    st.plotly_chart(fig)
