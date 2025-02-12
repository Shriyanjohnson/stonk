import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import numpy as np
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime

# Fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()
    data.dropna(inplace=True)
    return data

# News sentiment analysis
def fetch_sentiment(symbol):
    api_key = "833b7f0c6c7243b6b751715b243e4802"
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
    articles = all_articles.get('articles', [])
    if not articles:
        return 0
    sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
    return sentiment_score

# Machine learning model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
    features = data[['Close', 'RSI', 'MACD', 'Volatility']]
    labels = data['Target']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, labels)
    return model

# Streamlit app
st.title("Stock Prediction and Analysis")
symbol = st.text_input("Enter Stock Symbol", "AAPL")
stock_data = fetch_stock_data(symbol)

# Display Stock data and charts
st.write(stock_data.tail())
fig = go.Figure()
fig.add_trace(go.Candlestick(x=stock_data.index,
                             open=stock_data['Open'],
                             high=stock_data['High'],
                             low=stock_data['Low'],
                             close=stock_data['Close'],
                             name='Candlesticks'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
fig.update_layout(title=f"{symbol} Stock Price, RSI",
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# Stock prediction
model = train_model(stock_data)
predicted_price = model.predict(stock_data[['Close', 'RSI', 'MACD', 'Volatility']])[-1]
st.write(f"Predicted Movement: **{'Up' if predicted_price == 1 else 'Down'}**")

# News sentiment analysis
sentiment_score = fetch_sentiment(symbol)
st.write(f"News Sentiment Score: **{sentiment_score:.2f}**")

# Risk analysis
volatility = stock_data['Price Change'].std() * 100
st.write(f"Stock Volatility (Risk): **{volatility:.2f}%**")

# Download option
st.download_button("Download Stock Data", data=stock_data.to_csv(), file_name=f"{symbol}_data.csv", mime="text/csv")
