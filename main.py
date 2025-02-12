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

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")  # Last 90 days for better analysis
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch sentiment score from NewsAPI
def fetch_sentiment(symbol):
    try:
        # Use environment variable for API key
        api_key = os.getenv("newsapi_key")  # Get API key from environment variable
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0  # Default to neutral if no articles are found

        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception:
        return 0  # Default to neutral if API fails

# Function to train machine learning model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put

    features = data[['Close', 'RSI', 'MACD', 'Volatility']]
    labels = data['Target']

    # Initialize Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    
    accuracy = model.score(features, labels) * 100
    return model, accuracy

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility']]])

    prediction = model.predict(latest_features)[0]
    option = "Call" if prediction == 1 else "Put"

    # Adjust recommendation based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

    return option, strike_price, expiration_date

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
st.image("https://media.istockphoto.com/id/184276818/photo/us-dollars-stack.webp?b=1&s=170667a&w=0&k=20&c=FgRD0szcZ1Z-vpMZtkmMl5m1lmjVxQ2FYr5FUzDfJmM=", 
         caption="Let's Make Some Money!", use_column_width=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy = train_model(stock_data)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")

    # Display Model Accuracy
    st.markdown(f"### 🔥 Model Accuracy: **{accuracy:.2f}%**")

    # Plot stock data
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Market Data'))
    fig.update_layout(title=f"{symbol} Stock Price Chart",
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Disclaimer:** This application is for informational purposes only and does not constitute financial advice.
    Please conduct your own due diligence before making any investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
