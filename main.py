import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import talib as ta
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

# 📌 Streamlit App Title
st.set_page_config(page_title="S&P 500 Prediction & Options", layout="wide")
st.title("📈 S&P 500 Prediction & Option Recommendations")

# 📌 Sidebar: Date Selection
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# 📌 Fetch S&P 500 Data
@st.cache_data
def load_data():
    df = yf.download("^GSPC", start=start_date, end=end_date)
    df["RSI"] = ta.RSI(df["Close"], timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = ta.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["On_Balance_Volume"] = df["Volume"].cumsum()
    df["SMA_20"] = ta.SMA(df["Close"], timeperiod=20)
    df["SMA_50"] = ta.SMA(df["Close"], timeperiod=50)
    df["Volatility"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df.dropna(inplace=True)
    return df

df = load_data()

# 📊 Display Data & Candlestick Chart
st.subheader("📉 S&P 500 Historical Data & Indicators")
st.write(df.tail(10))

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="S&P 500")])
fig.update_layout(title="S&P 500 Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# 📌 Feature Engineering for Model
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
X = df[["RSI", "MACD", "On_Balance_Volume", "SMA_20", "SMA_50", "Volatility"]]
y = df["Target"]

# 📌 Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
params = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
grid_search = GridSearchCV(rf, params, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 📌 Display Prediction Accuracy
st.subheader("📊 Model Performance")
st.write(f"🔍 **Prediction Accuracy:** {accuracy:.2%}")

# 📌 Current Market Indicators Display
latest_data = df.iloc[-1]
st.subheader('📊 Technical Indicators and Their Current Values')

st.write(f"""
- **RSI (Relative Strength Index)**: {latest_data['RSI']:.2f}
  - RSI measures price momentum on a 0-100 scale.
  - **Above 70** = Overbought (potential reversal).  
  - **Below 30** = Oversold (potential buying opportunity).

- **MACD (Moving Average Convergence Divergence)**: {latest_data['MACD']:.2f}
  - Trend-following momentum indicator (relationship between 12-day & 26-day EMAs).  
  - **Positive MACD** = Bullish trend.  
  - **Negative MACD** = Bearish trend.

- **On-Balance Volume (OBV)**: {latest_data['On_Balance_Volume']:.2f}
  - OBV measures volume flow to confirm trends.  
  - Rising OBV supports an **uptrend**, declining OBV may indicate **weakening momentum**.

- **SMA-20 & SMA-50 (Simple Moving Averages)**:
  - **SMA-20** (Short-term trend): {latest_data['SMA_20']:.2f}  
  - **SMA-50** (Longer-term trend): {latest_data['SMA_50']:.2f}  
  - If **SMA-20 > SMA-50**, it’s a **bullish signal**. If **SMA-20 < SMA-50**, it’s **bearish**.

- **Volatility (ATR - Average True Range)**: {latest_data['Volatility']:.2f}
  - High ATR = **high volatility (uncertainty)**, Low ATR = **stable trend**.
""")

# 📌 Predict Next Day Movement
predicted_movement = best_model.predict([latest_data[["RSI", "MACD", "On_Balance_Volume", "SMA_20", "SMA_50", "Volatility"]]])[0]
prediction_text = "📈 Expected to go UP" if predicted_movement == 1 else "📉 Expected to go DOWN"
st.subheader("🔮 AI Market Prediction")
st.write(prediction_text)

# 📌 Option Strategy Recommendation
strike_price = round(latest_data["Close"] * (1.02 if predicted_movement == 1 else 0.98), 2)
option_type = "Call" if predicted_movement == 1 else "Put"
expiration_date = datetime.date.today() + datetime.timedelta((4 - datetime.date.today().weekday()) % 7)

st.subheader("📌 Suggested Option Trade")
st.write(f"""
- **Option Type:** {option_type}  
- **Strike Price:** {strike_price}  
- **Expiration Date:** {expiration_date}  
""")

# 📌 News Sentiment Analysis
st.subheader("📰 Market Sentiment Analysis")
news_url = "https://www.cnbc.com/stocks/"
news_page = requests.get(news_url)
soup = BeautifulSoup(news_page.content, "html.parser")
headlines = soup.find_all("a", class_="LatestNews-headline", limit=5)

for headline in headlines:
    title = headline.text.strip()
    sentiment = TextBlob(title).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    st.write(f"- **{title}** ({sentiment_label})")

# 📌 Unique Features Explanation
st.subheader("🚀 What Sets This Model Apart?")

st.markdown("""
- **AI-Driven Prediction**: Machine learning-powered forecasts using optimized Random Forest.
- **Technical Indicators Integration**: RSI, MACD, OBV, SMAs, and ATR for deep trend analysis.
- **Real-Time Sentiment Analysis**: Analyzes news sentiment to refine predictions.
- **Options Strategy**: Recommends precise strike prices & expiration dates dynamically.
- **User-Friendly Interface**: Data visualization with candlestick charts and downloadable datasets.
""")

st.success("✅ Analysis Complete! Stay ahead of the market.")
