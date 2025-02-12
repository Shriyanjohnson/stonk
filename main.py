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

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="max")  # Fetch all available historical data
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Function to fetch sentiment score from NewsAPI
def fetch_sentiment(symbol):
    try:
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

# Function to train machine learning model with cross-validation
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put
    features = data[['Close', 'RSI', 'MACD', 'Volatility', 'SMA_20', 'SMA_50']]
    labels = data['Target']  # Split the data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest model
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Cross-validation for better generalization
    accuracy = cv_scores.mean() * 100  # Cross-validation accuracy
    model.fit(X_train, y_train)
    return model, accuracy, X_test, y_test

# Function to generate option recommendation with sentiment adjustment
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    prediction_prob = model.predict_proba(latest_features)[0][1]  # Probability of being a 'Call'
    option = "Call" if prediction_prob > 0.5 else "Put"
    # Adjust recommendation based on sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()  # Friday expiry
    return option, strike_price, expiration_date

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")
st.image("https://media.istockphoto.com/id/184276818/photo/us-dollars-stack.webp?b=1&s=170667a&w=0&k=20&c=FgRD0szcZ1Z-vpMZtkmMl5m1lmjVxQ2FYr5FUzDfJmM=", caption="Let's Make Some Money!", use_container_width=True)

# Stock input and data retrieval
symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model, accuracy, X_test, y_test = train_model(stock_data)  # Train model

    # Generate option recommendation
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    # Displaying Option Recommendation and Information
    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")

    # Display Model Accuracy
    st.markdown(f"### 🔥 Model Accuracy (Cross-Validation): **{accuracy:.2f}%**")

    # Test accuracy on unseen data
    test_accuracy = model.score(X_test, y_test) * 100
    st.write(f"### Model Test Accuracy on Unseen Data: **{test_accuracy:.2f}%**")

    # Display Current Stock Price
    current_price = stock_data['Close'][-1]
    st.write(f"### Current Price of {symbol}: **${current_price:.2f}**")

    # Display Sentiment Score
    st.write(f"### Sentiment Score for {symbol}: **{sentiment_score:.2f}**")
    if sentiment_score > 0:
        st.markdown("#### **Positive News Sentiment!** 📈")
    elif sentiment_score < 0:
        st.markdown("#### **Negative News Sentiment!** 📉")
    else:
        st.markdown("#### **Neutral News Sentiment.** 🤔")

    # Create the figure with subplots (Candlestick, RSI, MACD)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol} Stock Price", "RSI", "MACD"),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Add Candlestick trace
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Market Data'
    ), row=1, col=1)

    # Add Moving Averages (SMA & EMA) for stock data
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='red')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='green')
    ), row=1, col=1)

    # Add RSI trace
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')
    ), row=2, col=1)

    # Add MACD trace
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='blue')
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Analysis",
        xaxis_title='Date',
        yaxis_title='Stock Price',
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(range=[0, 100], row=2, col=1)  # RSI range
    fig.update_yaxes(range=[stock_data['MACD'].min() - 1, stock_data['MACD'].max() + 1], row=3, col=1)  # MACD range

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

    # Add download button for the data
    csv = stock_data.to_csv(index=True)  # Convert data to CSV format
    st.download_button(
        label="Download Stock Data",
        data=csv,
        file_name=f"{symbol}_stock_data.csv",
        mime="text/csv",
    )

    # Disclaimer
    st.markdown(""" **Disclaimer:** This application is for informational purposes only and does not constitute financial advice. Please conduct your own due diligence before making any investment decisions. """)

    # Footer
    st.markdown("---")
    st.markdown("### Created by **Shriyan K**")
