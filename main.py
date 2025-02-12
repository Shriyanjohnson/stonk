import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom HTML & CSS Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #ecf0f1;
            font-size: 36px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subtitle {
            text-align: center;
            color: #bdc3c7;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .current-price, .recommendation {
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 24px;
            color: #2c3e50;
        }
        .recommendation h3 {
            color: #e74c3c;
            font-size: 22px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to fetch stock data and calculate additional indicators
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="90d")
        
        # Technical Indicators
        data['RSI'] = RSIIndicator(data['Close']).rsi()
        data['MACD'] = MACD(data['Close']).macd()
        data['Stochastic'] = StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        bb = BollingerBands(data['Close'])
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['EMA_50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
        data['EMA_200'] = EMAIndicator(data['Close'], window=200).ema_indicator()
        data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['Volatility'] = data['Close'].pct_change().rolling(10).std()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Drop NaN values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to fetch sentiment score
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
    except Exception as e:
        st.error(f"Error fetching news sentiment: {e}")
        return 0

# Train Model with Cross-Validation
def train_model(data):
    try:
        data['Price Change'] = data['Close'].diff()
        data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)
        
        features = data[['Close', 'RSI', 'MACD', 'Stochastic', 'BB_Upper', 'BB_Lower', 'ADX', 'EMA_50', 'EMA_200', 'OBV', 'Volatility', 'SMA_20', 'SMA_50']]
        labels = data['Target']
        
        # Standardize Features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    
        # Split Data for Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        model.fit(X_train, y_train)
        return model, cv_scores.mean() * 100, X_test, y_test
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, 0, None, None

# Generate Option Recommendation Based on Latest Data and Sentiment
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Stochastic'], latest_data['BB_Upper'], latest_data['BB_Lower'], latest_data['ADX'], latest_data['EMA_50'], latest_data['EMA_200'], latest_data['OBV'], latest_data['Volatility'], latest_data['SMA_20'], latest_data['SMA_50']]])
    
    # Predict Probability of Price Going Up
    prediction_prob = model.predict_proba(latest_features)[0][1]
    option = "Call" if prediction_prob > 0.5 else "Put"

    # Adjust Option Based on Sentiment
    if sentiment_score > 0.2 and option == "Put":
        option = "Call"
    elif sentiment_score < -0.2 and option == "Call":
        option = "Put"
    
    strike_price = round(latest_data['Close'] / 10) * 10
    expiration_date = (datetime.datetime.now() + datetime.timedelta((4 - datetime.datetime.now().weekday()) % 7)).date()
    
    return option, strike_price, expiration_date

# Streamlit UI Components
st.markdown('<div class="title">💰 AI Stock Options Predictor 💰</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Developed by Shriyan Kandula, a sophomore at Shaker High School.</div>', unsafe_allow_html=True)

# Stock Symbol Input
symbol = st.text_input("Enter Stock Symbol", "AAPL")

if symbol:
    stock_data = fetch_stock_data(symbol)
    if stock_data is not None:
        sentiment_score = fetch_sentiment(symbol)
        model, accuracy, X_test, y_test = train_model(stock_data)
    
        if model:
            current_price = stock_data['Close'].iloc[-1]
            st.markdown(f'<div class="current-price">Current Price of {symbol}: **${current_price:.2f}**</div>', unsafe_allow_html=True)

            option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

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
            
            # Interactive Charts for Technical Indicators
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("Price and Indicators", "Volume & OBV", "RSI and Sentiment"))

            # Price and Indicators
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Price", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], name="SMA 20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], name="SMA 50", line=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_50'], name="EMA 50", line=dict(color='red')), row=1, col=1)

            # Volume & OBV
            fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name="Volume", marker=dict(color='grey')), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['OBV'], name="OBV", line=dict(color='purple')), row=2, col=1)

            # RSI and Sentiment
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name="RSI", line=dict(color='black')), row=3, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=[sentiment_score] * len(stock_data), name="Sentiment", line=dict(color='red', dash='dash')), row=3, col=1)

            fig.update_layout(height=800, title=f"Technical Indicators for {symbol}")
            st.plotly_chart(fig)

            # Provide CSV Download for Stock Data
            st.download_button("Download Stock Data as CSV", stock_data.to_csv(), file_name=f"{symbol}_stock_data.csv")
            
        else:
            st.warning("Model training failed. Please try again.")
else:
    st.info("Please enter a valid stock symbol.")

# Footer
st.markdown("""
    <div class="footer">
        Created by **Shriyan Kandula** | 💻 Stock Predictions & Insights
    </div>
""", unsafe_allow_html=True)
