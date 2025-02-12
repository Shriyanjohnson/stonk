import yfinance as yf
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from textblob import TextBlob
from newsapi import NewsApiClient
import datetime
@@ -14,8 +15,10 @@
# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")  # Last 90 days for better analysis
    data = stock.history(period="120d")  # Last 120 days for better analysis
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['MACD'] = MACD(data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(10).std()  # 10-day rolling volatility
    data.dropna(inplace=True)  # Remove NaN values
    return data

@@ -25,30 +28,33 @@ def fetch_sentiment(symbol):
        all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
        articles = all_articles.get('articles', [])
        if not articles:
            return 0
            return 0  # Default to neutral

        sentiment_score = sum(TextBlob(article['title']).sentiment.polarity for article in articles) / len(articles)
        return sentiment_score
    except Exception:
        return 0  # Default to neutral if API fails
        return 0  # If NewsAPI fails, assume neutral sentiment

# Function to train ML model
def train_model(data):
    data['Price Change'] = data['Close'].diff()
    data['Target'] = np.where(data['Price Change'].shift(-1) > 0, 1, 0)  # 1 = Call, 0 = Put

    features = data[['Close', 'RSI']]
    features = data[['Close', 'RSI', 'MACD', 'Volatility']]
    labels = data['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(features, labels)
    return model
    # Calculate accuracy using the training data
    accuracy = model.score(features, labels)
    return model, accuracy

# Function to generate option recommendation
def generate_recommendation(data, sentiment_score, model):
    latest_data = data.iloc[-1]
    latest_features = np.array([[latest_data['Close'], latest_data['RSI']]])
    latest_features = np.array([[latest_data['Close'], latest_data['RSI'], latest_data['MACD'], latest_data['Volatility']]])

    prediction = model.predict(latest_features)[0]
    option = "Call" if prediction == 1 else "Put"
@@ -65,15 +71,26 @@ def generate_recommendation(data, sentiment_score, model):
    return option, strike_price, expiration_date

# Streamlit UI
st.title("AI Stock Options Predictor")
st.title("💰 AI Stock Options Predictor 💰")
# Display image of money
st.image("https://media.istockphoto.com/id/184276818/photo/us-dollars-stack.webp?b=1&s=170667a&w=0&k=20&c=FgRD0szcZ1Z-vpMZtkmMl5m1lmjVxQ2FYr5FUzDfJmM=", 
         caption="Let's Make Some Money!", use_column_width=True)

symbol = st.text_input("Enter Stock Symbol", "AAPL")
if symbol:
    stock_data = fetch_stock_data(symbol)
    sentiment_score = fetch_sentiment(symbol)
    model = train_model(stock_data)
    model, accuracy = train_model(stock_data)
    option, strike_price, expiration = generate_recommendation(stock_data, sentiment_score, model)

    st.write(f"Option Recommendation: **{option}**")
    st.write(f"Strike Price: **${strike_price}**")
    st.write(f"Expiration Date: **{expiration}**")
    # Display Model Accuracy at the Top
    st.markdown(f"### 🔥 Model Accuracy: **{accuracy * 100:.2f}%**")
    st.write(f"### Option Recommendation: **{option}**")
    st.write(f"### Strike Price: **${strike_price}**")
    st.write(f"### Expiration Date: **{expiration}**")
# Footer
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
