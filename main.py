import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Function to fetch stock data
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="90d")
    return data

# Streamlit UI with enhanced layout
st.set_page_config(page_title="AI Stock Options Predictor", layout="wide")

# Colorful background for the main page and sidebar
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #ff6347, #00b3b3);
            color: white;
        }
        .css-1v3fvcr {
            background-color: #ff5733;
        }
        .dataframe tbody tr:nth-child(odd) {
            background-color: #f4f4f9;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #e6f7ff;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select a Section", ("Stock Info", "Recommendations", "Charts"))

if tab == "Stock Info":
    st.title("📈 Stock Data")
    symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    if symbol:
        stock_data = fetch_stock_data(symbol)
        st.dataframe(stock_data.tail())

elif tab == "Recommendations":
    st.title("💡 Option Recommendations")
    if symbol:
        st.write("Here are the recommendations based on your stock data.")
        # Add more logic for generating recommendations here.

elif tab == "Charts":
    st.title("📊 Stock Price Charts")
    if symbol:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close']))
        st.plotly_chart(fig)

# Footer Section
st.markdown("---")
st.markdown("### Created by **Shriyan K**")
