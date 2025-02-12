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

# Cache the data fetching and indicator calculation to speed things up
@st.cache
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="6mo")  # Use a smaller time period (e.g., "6mo" or "90d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)  # Remove NaN values
    return data

# Other functions (fetch sentiment, train model, generate recommendation) go here...

# Streamlit UI
st.title("💰 AI Stock Options Predictor 💰")

symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    with st.spinner('Fetching data...'):
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

    # Show numerical values of indicators over time
    st.subheader(f"### Indicator Values for {symbol} Over Time")
    st.dataframe(stock_data[['Close', 'RSI', 'MACD', 'SMA_20', 'SMA_50']])

    # Add download button for the data
    csv = stock_data.to_csv(index=True)  # Convert data to CSV format
    st.download_button(
        label="Download Stock Data",
        data=csv,
        file_name=f"{symbol}_stock_data.csv",
        mime="text/csv",
    )
