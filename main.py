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

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='833b7f0c6c7243b6b751715b243e4802')

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="120d")  # Last 120 days for better analysis
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch news sentiment
def fetch_sentiment(symbol):
    try:
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0  # Default to neutral

        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception:
        return 0  # If NewsAPI fails, assume neutral sentiment

# Function to train ML model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put

    features = data[['Close', 'RSI', 'MACD', 'Volatility']]
    labels = data['Target']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(features, labels)

    # Calculate accuracy using the training data
    accuracy = model.score(features, labels)

    return model, accuracy

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility']]])

    prediction = model.predict(latest_features)[0]
    option = "Call" if prediction == 1 else "Put"

    # Adjust based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"

    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()

    return option, strike_price, expiration_date

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")

# Display image of money
st.image("https://media.istockphoto.com/id/184276818/photo/us-dollars-stack.webp?b=1&s=170667a&w=0&k=20&c=FgRD0szcZ1Z-vpMZtkmMl5m1lmjVxQ2FYr5FUzDfJmM=", 
         caption="Let's Make Some Money!", use_column_width=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy = train_model(stock_data)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    # Display Model Accuracy at the Top
    st.markdown(f"### 🔥 Model Accuracy: **{accuracy * 100:.2f}%**")

    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"### Strike Price: **${strike_price}**")
    st.write(f"### Expiration Date: **{expiration}**")

# Footer
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
