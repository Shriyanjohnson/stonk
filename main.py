import yfinance as yf
import streamlit as st
from ta import add_all_ta_features
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import requests

# Function to fetch stock data and calculate technical indicators
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="60d")  # Last 60 days of stock data for better analysis
    data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')
    return data

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='833b7f0c6c7243b6b751715b243e4802')  # Your API Key

# Function to fetch market sentiment based on news headlines
def fetch_sentiment(symbol):
    try:
        # Fetch the latest news articles for the given symbol
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)

        # Extract the titles of the articles
        articles = all_articles['articles']
        if not articles:
            return 0  # No articles found, return neutral sentiment

        # Analyze sentiment based on articles (simple logic: positive or negative)
        positive_sentiment = 0
        negative_sentiment = 0

        for article in articles:
            title = article['title']
            description = article['description']
            content = article['content']

            # Basic sentiment check: You can replace this with a more sophisticated method
            if 'positive' in title.lower() or 'good' in description.lower() or 'rise' in content.lower():
                positive_sentiment += 1
            elif 'negative' in title.lower() or 'bad' in description.lower() or 'drop' in content.lower():
                negative_sentiment += 1

        # If there are more positive articles, sentiment is positive, else negative
        sentiment_score = positive_sentiment - negative_sentiment
        return sentiment_score

    except requests.exceptions.RequestException as e:
        # Handle request errors (e.g., network issues)
        print(f"Error fetching news: {e}")
        return 0  # Default to neutral sentiment if there was an issue

    except KeyError as e:
        # Handle any KeyError (e.g., missing fields in the response)
        print(f"Error parsing news data: {e}")
        return 0  # Default to neutral sentiment if parsing fails

    except Exception as e:
        # Catch any other errors
        print(f"Unexpected error: {e}")
        return 0  # Default to neutral sentiment for any other error

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to predict the stock option recommendation (Call or Put)
def generate_recommendation(data, sentiment_score):
    # Calculate technical features (e.g., RSI)
    rsi = calculate_rsi(data)
    latest_rsi = rsi.iloc[-1]

    # Use a RandomForest classifier to predict the recommendation
    # Placeholder logic to use stock data and sentiment
    features = [
        data['Close'].mean(),     # Average close price (used for simplicity)
        latest_rsi,               # RSI indicator
        sentiment_score           # Sentiment from news
    ]

    # For simplicity, we're using a Random Forest with dummy data (expand this with actual training)
    model = RandomForestClassifier()
    model.fit([[features[0], features[1], features[2]]], [1])  # Dummy training for now
    prediction = model.predict([features])

    option = "Call" if prediction[0] == 1 else "Put"

    # Calculate strike price as nearest 10 of the last closing price
    strike_price = round(data['Close'].iloc[-1] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()  # Next Friday

    return option, strike_price, expiration_date

# Streamlit UI
st.title("AI Stock Options Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    # Fetch stock data and news sentiment
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    
    # Generate recommendation
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score)
    
    # Display the recommendation
    st.write(f"Option Recommendation: {option}")
    st.write(f"Strike Price: ${strike_price}")
    st.write(f"Expiration Date: {expiration}")
