import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score

# Function to fetch stock data
def fetch_stock_data(symbol, period='90d'):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)  # Customizable period
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data['SMA_20'] = data['Close'].rolling(20).mean()  # Adding 20-day Simple Moving Average
    data['SMA_50'] = data['Close'].rolling(50).mean()  # Adding 50-day Simple Moving Average
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch sentiment score from NewsAPI
def fetch_sentiment(symbol):
    try:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            st.error("API Key for NewsAPI is not set. Please set the API key in your environment.")
            return 0

        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0

        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")
        return 0

# Function to train machine learning model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)

    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Improved model with hyperparameters
    model.fit(X_train, y_train)
    
    # Cross-validation for better accuracy
    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean() * 100
    return model, accuracy

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])

    prediction = model.predict(latest_features)[0]
    option = "Call" if prediction == 1 else "Put"

    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

    return option, strike_price, expiration_date

# Streamlit UI Improvements
st.title("💰 AI Stock Options Predictor 💰")
st.image("https://media.istockphoto.com/id/184276818/photo/us-dollars-stack.webp?b=1&s=170667a&w=0&k=20&c=FgRD0szcZ1Z-vpMZtkmMl5m1lmjVxQ2FYr5FUzDfJmM=", caption="Let's Make Some Money!", use_container_width=True)

symbol = st.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"])  # Dropdown for stock selection

if symbol:
    st.text(f"Fetching data for {symbol}...")
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    
    # Load or train model
    if os.path.exists('trained_model.pkl'):
        model = joblib.load('trained_model.pkl')
        accuracy = model.score(stock_data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']], stock_data['Target']) * 100
    else:
        model, accuracy = train_model(stock_data)
        joblib.dump(model, 'trained_model.pkl')

    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.write(f"### Option Recommendation: **{option}**", color='green' if option == "Call" else 'red')
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")

    # Display Model Accuracy
    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")
    
    # Display stock data in a table
    st.dataframe(stock_data.tail(10), width=600)

    # Plot stock data using Plotly
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Market Data'))
    fig.update_layout(title=f"{symbol} Stock Price Chart", xaxis_title='Date', yaxis_title='Stock Price', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Allow for sentiment threshold adjustment
    sentiment_threshold = st.slider("Sentiment Threshold", min_value=-1.0, max_value=1.0, value=0.2, step=0.1)
    st.write(f"Sentiment threshold set to: {sentiment_threshold}")

    st.markdown("""
    **Disclaimer:** This application is for informational purposes only and does not constitute financial advice.
    Please conduct your own due diligence before making any investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
