import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import math
import openai
from datetime import datetime
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from yahooquery import search
import requests
from transformers import pipeline  # FinBERT for sentiment analysis
from prophet import Prophet  # Predictive modeling
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor  # Alternative ML Model


# Set app to wide mode
st.set_page_config(layout="wide")

# ========== Technical Indicator Helpers ==========
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_fibonacci_level(series):
    min_price = series.min()
    max_price = series.max()
    current_price = series.iloc[-1]
    return ((current_price - min_price) / (max_price - min_price)) * 100

# Function to fetch stock news from TradingView
def fetch_tradingview_news(ticker):
    url = f"https://www.tradingview.com/symbols/{ticker}/news/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if status is not 200
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('div', class_='tv-screener__description')

        if not news_items:
            return ["No news found or TradingView changed its structure."]

        return [item.get_text(strip=True) for item in news_items]

    except requests.exceptions.RequestException as e:
        return [f"Error fetching news: {e}"]

# Function to analyze sentiment using FinBERT
@st.cache_resource  # Cache the model to avoid repeated downloads
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=-1)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

def analyze_sentiment(news_list):
    sentiment_pipeline = load_sentiment_model()
    
    if sentiment_pipeline is None:
        return ["Error loading FinBERT model."]
    
    sentiments = []
    for news in news_list:
        try:
            sentiment = sentiment_pipeline(news)[0]
            sentiments.append(sentiment['label'])
        except Exception as e:
            sentiments.append(f"Error analyzing sentiment: {e}")
    
    return sentiments

# Load API key from secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# Ensure session_state variable is initialized
if "stock_analysis" not in st.session_state:
    st.session_state.stock_analysis = None  # or some default value

# Function to analyze stock using GPT
def analyze_stock_with_gpt(ticker, stock_data, news):
    if stock_data is None or stock_data.empty:
        return "Error: No stock data available."

    if "y" not in stock_data.columns:
        return f"Error: 'Close' column not found. Available columns: {list(stock_data.columns)}"

    prompt = f"""
    - Stock: {ticker}
    - Last 30-day closing prices: {stock_data['y'].tail(30).tolist()}
    - News Sentiment: {news}
    
    Provide an analysis including:
    - General sentiment (bullish, bearish, or neutral).
    - Technical insights based on price trends.
    - A short forecast for the stock.
    """

    # Updated API call with OpenAI's latest version
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" or "gpt-4" based on your preference
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content  # Extract text response correctly

# Function to get stock ticker from company name
def get_ticker_from_name(search_input):
    """Fetch possible tickers from Yahoo Finance dynamically as the user types."""
    try:
        # Use YahooQuery's search function to find matches
        results = search(search_input)  # Search for tickers matching the company name
        matches = results.get("quotes", [])

        if not matches:
            return {}

        # Extract tickers and names
        ticker_options = {item["shortname"]: item["symbol"] for item in matches if "symbol" in item and "shortname" in item}

        return ticker_options

    except Exception as e:
        st.error(f"Error fetching ticker: {e}")
        return {}

# Function to fetch stock data
@st.cache_data
def fetch_data(ticker):
    try:
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="10y")  # Fetch 10 years of historical data
        info = stock.info  # Fetch stock information

        # Ensure data and info are not empty
        if data.empty:
            st.error(f"No historical data found for ticker: {ticker}")
            return None, None
        if not info:
            st.error(f"No stock information found for ticker: {ticker}")
            return data, None

        return data, info

    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return None, None

# Train Prophet model and forecast
def get_stock_data(ticker):
    try:
        # Fetch stock data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")  # Retrieve 5 year of stock data
        data = data.reset_index()[["Date", "Close"]]  # Keep Date and Close columns
        data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)  # Prophet expects 'ds' for dates and 'y' for values

        # Ensure 'ds' is a datetime object (remove timezone)
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)  # Remove timezone

        # Check for missing values and drop them
        data = data.dropna(subset=['ds', 'y'])

        # Check for duplicate dates and remove them
        data = data.drop_duplicates(subset=['ds'])

        # Ensure 'y' column is numeric (float)
        data['y'] = pd.to_numeric(data['y'], errors='coerce')

        # Drop rows with NaN 'y' values after coercion
        data = data.dropna(subset=['y'])

        # Sort the data by 'ds' (Prophet expects the dates to be in ascending order)
        data = data.sort_values(by='ds').reset_index(drop=True)

        return data
    except Exception as e:
        raise ValueError(f"Invalid ticker symbol: {ticker}. Error: {str(e)}")

def predict_future(ticker):
    # Fetch stock data
    df = get_stock_data(ticker)

    # Ensure the required columns are present in the dataframe
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("The dataframe is missing required columns 'ds' and 'y'")

    # Debug: Print the first few rows of the dataframe
    print("DataFrame before Prophet fitting:")
    print(df.head())

    # Create and fit the Prophet model
    model = Prophet()

    # Fit the model to the data inside a try-except block
    try:
        model.fit(df)
    except Exception as e:
        raise ValueError(f"Error fitting the Prophet model: {str(e)}")

    # Make future predictions (e.g., next 30 days)
    future = model.make_future_dataframe(periods=60)  # Correct usage
    forecast = model.predict(future)

    # Check if forecast contains necessary columns
    if "ds" in forecast.columns and "yhat" in forecast.columns:
        # Clean the forecast data by dropping NaN values
        forecast_cleaned = forecast[["ds", "yhat"]].dropna()

        # Plot the forecast data if it's available and valid
        if not forecast_cleaned.empty:
            st.line_chart(forecast_cleaned.set_index("ds"))  # Plot predictions
        else:
            st.error("No valid forecast data available to plot.")
    else:
        st.error("Forecast data is missing necessary columns ('ds' and 'yhat').")

    return forecast

# -------------------------------
# 3️⃣ FUNDAMENTAL & TECHNICAL AI ANALYSIS
# -------------------------------

# Analyse fundamentals and technicals and score the stock using prophet
def analyze_fundamentals(ticker):
    try:
        # Fetch stock data
        data, info = fetch_data(ticker)

        if data is None or info is None:
            return "Error: Unable to fetch stock data or information."

        # Predict future prices using Prophet
        forecast = predict_future(ticker)

        # Define or compute the 'analysis' variable
        analysis = f"Forecast generated for {ticker}. Predicted future prices and trends are available."
        return analysis

    except Exception as e:
        st.error(f"Error analyzing fundamentals: {e}")
        return None
    

# Monte Carlo Simulation
def monte_carlo_simulation(data, n_simulations=1000, n_days=252, log_normal=False, volatility=None):
    daily_returns = data['Close'].pct_change().dropna()
    mean_return = daily_returns.mean()
    vol = volatility if volatility else daily_returns.std()

    simulations = np.zeros((n_simulations, n_days))
    last_price = data['Close'].iloc[-1]

    for i in range(n_simulations):
        prices = [last_price]
        for j in range(n_days):
            shock = np.random.normal(0, vol)
            drift = mean_return
            if log_normal:
                prices.append(prices[-1] * np.exp(drift + shock))
            else:
                prices.append(prices[-1] * (1 + drift + shock))
        simulations[i] = prices[1:]

    return simulations

# Calculate RSI and MACD
def calculate_indicators(data):
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    # Calculate MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Moving Average (SMA)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    
    return data

# Generate Buy/Sell signals
def generate_signals(data):
    data = calculate_indicators(data)
    
    # Define signals
    data['Buy Signal'] = (data['RSI'] < 30) & (data['MACD'] > data['MACD_signal']) 
                           
    
    data['Sell Signal'] = (data['RSI'] > 70) & (data['MACD'] < data['MACD_signal'])

    
    # Filter to only keep the most significant signals
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0
    signals.loc[data['Buy Signal'], 'Signal'] = 1
    signals.loc[data['Sell Signal'], 'Signal'] = -1

    return signals

# Train a simple machine learning model using past signals
def train_model(data):
    data = calculate_indicators(data)
    
    # Features: RSI, MACD, Close, SMA50
    features = ['RSI', 'MACD', 'Close', 'SMA50']
    data = data.dropna(subset=features)
    X = data[features]
    y = data['Signal'].dropna()
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model: RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Return the model for future predictions
    return model, X_test, y_test, y_pred

# Function to fetch VIX data
def get_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d")  # Get the latest data
    return vix_data["Close"].iloc[-1] if not vix_data.empty else None

# Function to determine VIX Indicator signal
def vix_indicator(vix_value):
    if vix_value is None:
        return "No Data", "gray"
    elif vix_value < 15:
        return "🟢 BUY", "green"
    elif 15 <= vix_value <= 25:
        return "🟡 NEUTRAL", "yellow"
    else:
        return "🔴 SELL", "red"

menu = st.sidebar.radio(
    "Select a Section",
    [
        "Market Analysis | Buy Signals",
        "Stock Info",
        "Historical Analysis",
        "Monte Carlo Simulations",
        "Export Data",
        "Sector Map",
        "Refined Strategy (RSI with Trend)",
    ]
)

# Exit App Button
if st.sidebar.button("❌ Exit App"):
    os._exit(0)

# Sector Map Section
if menu == "Sector Map":
    st.title("📊 Sector Map - Finviz")
    
    finviz_url = "https://finviz.com/map.ashx?t=sec&st=w1"
    
    # Display a clickable link instead of an iframe
    st.markdown(f"[🔗 Click here to view the Finviz Sector Map]({finviz_url})", unsafe_allow_html=True)

# Stock Info Section
if menu == "Stock Info":
    st.title("📊 Stock Info and Metrics")
    st.markdown("### Search for a Stock Ticker or Company Name")

    # Input box for user to type the stock name, with a default value
    search_input = st.text_input("Enter Stock Ticker or Name", "Apple")

    # Ensure session state is initialized for selected_ticker
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None

    if search_input:
        # Dynamically fetch matching tickers
        ticker_options = get_ticker_from_name(search_input)

        with st.expander("### Matching Companies:"):
            if ticker_options:
                for company_name, ticker in ticker_options.items():
                    if st.button(f"{ticker} - {company_name}"):
                        st.session_state.selected_ticker = ticker  # Store in session state
                        st.rerun()  # Rerun script to update display immediately
            else:
                st.warning("No matching stocks found. Please refine your search.")
    else:
        st.info("Please enter a stock name or ticker to search.")

    # Display stock information
    with st.expander("Sector and Industry", expanded=False):
        if st.session_state.selected_ticker:
            st.write(f"**Selected Ticker:** {st.session_state.selected_ticker}")
            # Here you can fetch stock data and display additional information
        else:
            st.warning("No stock selected. Please choose a ticker.")

        col1, col2, col3 = st.columns(3)  # Initialize columns

        with col1:
            if st.session_state.selected_ticker:
                _, info = fetch_data(st.session_state.selected_ticker)
                if info:
                    st.markdown(f"<h4>Name: {st.session_state.selected_ticker} - {info.get('longName', 'Company Name Not Found')}</h4>", unsafe_allow_html=True)
                else:
                    st.warning("Stock information not found.")
            else:
                st.warning("No stock selected.")
        with col2:
            if st.session_state.selected_ticker:
                _, info = fetch_data(st.session_state.selected_ticker)
                if info:
                    st.markdown(f"<h4>Sector: {info.get('sector', 'Sector Not Found')}</h4>", unsafe_allow_html=True)
                else:
                    st.warning("No stock selected.")
        with col3:
            if st.session_state.selected_ticker:
                _, info = fetch_data(st.session_state.selected_ticker)
                if info:
                    st.markdown(f"<h4>Industry: {info.get('industry', 'Industry Not Found')}</h4>", unsafe_allow_html=True)
                else:
                    st.warning("No stock selected.")
        
    # Initialize info with a default value
    info = None  

    with st.expander("Company Info", expanded=False):
        if st.session_state.selected_ticker:
            _, info = fetch_data(st.session_state.selected_ticker)
            if info:
                st.write(info)
            else:
                st.warning("No stock selected.")

    # Check if key exists and value is valid before using it
    def safe_metric(value, divisor=1, suffix="", percentage=False):
            """Safely formats a metric value for Streamlit display."""
            try:
                if value is None:
                    return "N/A"
                if isinstance(value, (int, float)):
                    if math.isnan(value):  # Handle NaN values
                        return "N/A"
                    if percentage:
                        return f"{value:.2%}"
                    return f"${value / divisor:.2f}{suffix}" if divisor > 1 else f"${value:.2f}"
                return "N/A"
            except Exception as e:
                return f"Error: {e}"  # Return error message instead of crashing

    if isinstance(info, dict) and info:  # Ensure 'info' is a dictionary and not empty
        freecash_flow = info.get('freeCashflow', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        peg_ratio = info.get('trailingPegRatio', 'N/A')
        earnings_growth = info.get('earningsGrowth', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        freecash_flow = info.get('freeCashflow', 'N/A')
        netincome = info.get('netIncomeToCommon', 'N/A')
        grossmargin = info.get('grossMargins', 'N/A')
        operatingmargin = info.get('operatingMargins', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        institutional_ownership = info.get('heldPercentInstitutions', 'N/A')
        insider_ownership = info.get('heldPercentInsiders', 'N/A')
    else:
        st.warning("Stock information not found.")

    if isinstance(info, dict) and info:  # Ensure 'info' is a dictionary and not empty
        pe_ratio = info.get('trailingPE', 'N/A')
        peg_ratio = info.get('trailingPegRatio', 'N/A')
        earnings_growth = info.get('earningsGrowth', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        freecash_flow = info.get('freeCashflow', 'N/A')
        netincome = info.get('netIncomeToCommon', 'N/A')
        grossmargin = info.get('grossMargins', 'N/A')
        operatingmargin = info.get('operatingMargins', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        institutional_ownership = info.get('heldPercentInstitutions', 'N/A')
        insider_ownership = info.get('heldPercentInsiders', 'N/A')
        trailingeps = info.get('epsCurrentYear', 'N/A')
        forwardeps = info.get('forwardEps', 'N/A')
        revenue = info.get('totalRevenue', 'N/A')
        totaldebt = info.get('totalDebt', 'N/A')
        totalcash = info.get('totalCash', 'N/A')
        revenuegrowth = info.get('revenueGrowth', 'N/A')
    else:
        st.warning("Stock information not found.")

    with st.expander("Stock Overview", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="📈 Market Cap", value=safe_metric(info['marketCap'], 1e9, "B") if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Free Cash Flow", value=safe_metric(freecash_flow, 1e9, "B") if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Net Income", value=safe_metric(netincome, 1e9, "B") if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Gross Margin", value=safe_metric(grossmargin, percentage=True) if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Operating Margin", value=safe_metric(operatingmargin, percentage=True) if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Profit Margin", value=safe_metric(profit_margin, percentage=True) if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Earnings Growth", value=safe_metric(earnings_growth, percentage=True) if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Dividend Yield", value=safe_metric(info.get('dividendYield'), percentage=True) if isinstance(info, dict) and 'marketCap' in info else "N/A")

        with col2:
            # Safe conversion function to handle None and invalid values
            def safe_float(value):
                try:
                    return float(value) if value not in [None, 'N/A', '', 'NaN'] else None
                except (ValueError, TypeError):
                    return None

            if info is not None:    
                # Fetch and safely convert values from `info`
                pe_ratio = safe_float(info.get('trailingPE', None))
                pb_ratio = safe_float(info.get('priceToBook', None))
                ps_ratio = safe_float(info.get('priceToSalesTrailing12Months', None))
                roe = safe_float(info.get('returnOnEquity', None))
                forward_pe = safe_float(info.get('forwardPE', None))
                totaldebt = safe_float(info.get('totalDebt', None))
                totalcash = safe_float(info.get('totalCash', None))

                # Ensure info is a dictionary
                if isinstance(info, dict):
                    peg_ratio = info.get('pegRatio', None)  # Fetch pegRatio safely
                else:
                    peg_ratio = None  # Default to None if info is invalid

                # Convert safely
                try:
                    peg_ratio = float(peg_ratio) if peg_ratio not in [None, 'N/A', '', 'NaN'] else "N/A"
                except (ValueError, TypeError):
                    peg_ratio = "N/A"  # If conversion fails, set as "N/A"

                # Assign color based on value
                if isinstance(peg_ratio, (int, float)):
                    peg_color = "green" if peg_ratio < 1 else "orange" if 1 <= peg_ratio <= 2 else "red"
                else:
                    peg_color = "gray"

                # Categorize P/E Ratio
                if isinstance(pe_ratio, (int, float)) and pe_ratio is not None and not math.isnan(pe_ratio):
                    pe_color = "green" if pe_ratio < 15 else "orange" if 15 <= pe_ratio <= 25 else "red"
                    st.markdown(f"<span style='color:{pe_color}; font-size:25px;'>📈 P/E Ratio: {pe_ratio:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:gray; font-size:25px;'>📈 P/E Ratio: N/A</span>", unsafe_allow_html=True)

                # Categorize Forward P/E Ratio
                if isinstance(forward_pe, (int, float)) and forward_pe is not None and not math.isnan(forward_pe):
                    pe_color = "green" if forward_pe < 15 else "orange" if 15 <= forward_pe <= 25 else "red"
                    st.markdown(f"<span style='color:{pe_color}; font-size:25px;'>📈 Forward P/E Ratio: {forward_pe:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:gray; font-size:25px;'>📈 Forward P/E Ratio: N/A</span>", unsafe_allow_html=True)
                
                # Categorize P/S Ratio
                if isinstance(ps_ratio, (int, float)) and ps_ratio is not None and not math.isnan(ps_ratio):
                    ps_color = "green" if ps_ratio < 1 else "orange" if 1 <= ps_ratio <= 2 else "red"
                    st.markdown(f"<span style='color:{ps_color}; font-size:25px;'>📈 P/S Ratio: {ps_ratio:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:gray; font-size:25px;'>📈 P/S Ratio: N/A</span>", unsafe_allow_html=True)

                # Categorize P/B Ratio
                if isinstance(pb_ratio, (int, float)) and pb_ratio is not None and not math.isnan(pb_ratio):
                    pb_color = "green" if pb_ratio < 1 else "orange" if 1 <= pb_ratio <= 2 else "red"
                    st.markdown(f"<span style='color:{pb_color}; font-size:25px;'>📈 P/B Ratio: {pb_ratio:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:gray; font-size:25px;'>📈 P/B Ratio: N/A</span>", unsafe_allow_html=True)

                #Categorize ROE Ratio
                if isinstance(roe, (int, float)) and roe is not None and not math.isnan(roe):
                    roe_color = "green" if roe > 0.15 else "orange" if 0.05 <= roe <= 0.15 else "red"
                    st.markdown(f"<span style='color:{roe_color}; font-size:25px;'>📈 ROE Ratio: {roe:.2%}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:gray; font-size:25px;'>📈 ROE Ratio: N/A</span>", unsafe_allow_html=True)

                # Categorize Debt to Equity Ratio
                if isinstance(totaldebt, (int, float)) and isinstance(totalcash, (int, float)) and totaldebt > 0 and totalcash > 0:
                    debt_to_equity = totaldebt / totalcash
                    debt_color = "green" if debt_to_equity < 1 else "orange" if 1 <= debt_to_equity <= 2 else "red"
                    st.markdown(f"<span style='color:{debt_color}; font-size:25px;'>📈 Debt to Equity Ratio: {debt_to_equity:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.metric(label="📈 Debt to Equity Ratio", value="N/A")

                st.write("")  # Empty line

                # Display Total Debt & Total Cash
                st.metric(label="📈 Total Debt", value=f"${totaldebt / 1e9:.2f}B" if isinstance(totaldebt, (int, float)) else "N/A")
                st.metric(label="📈 Total Cash", value=f"${totalcash / 1e9:.2f}B" if isinstance(totalcash, (int, float)) else "N/A")

            else:
                # Handle the case where info is None
                st.write("Error: 'info' object is not properly initialized or is None.")

        with col3:
            st.metric(label="📈 Trailing EPS", value=f"${trailingeps:.2f}" if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Forward EPS", value=f"${forwardeps:.2f}" if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Revenue", value=f"${revenue / 1e9:.2f}B" if revenue and isinstance(revenue, (int, float)) else "N/A")
            st.metric(label="📈 Revenue Growth", value=f"{revenuegrowth:.2%}" if revenuegrowth and isinstance(revenuegrowth, (int, float)) else "N/A")
            st.metric(label="📈 Institutional Ownership", value=f"{institutional_ownership:.2%}" if isinstance(info, dict) and 'marketCap' in info else "N/A")
            st.metric(label="📈 Insider Ownership", value=f"{insider_ownership:.2%}" if isinstance(info, dict) and 'marketCap' in info else "N/A")
    
        
        
    #Using AI to analyse the stock based on the uper metrics
    st.markdown("📊 AI-Powered Stock Analysis", )

    with st.expander("Stock Analysis", expanded=True):
            # Fetch and display stock data if a ticker is selected
            if st.session_state.selected_ticker:
                    data, info = fetch_data(st.session_state.selected_ticker)
                    st.write(f"**Selected Ticker:** {st.session_state.selected_ticker}")
            
            if "stock_data" not in st.session_state:
                st.session_state.stock_data = None  # Default to None or an empty DataFrame
            
            # Button to analyze stock
            if st.button("Analyze Stock"):
                # Fetch and store stock data
                stock_data = get_stock_data(ticker)
                news = analyze_sentiment(ticker)

                # Store results in session state
                st.session_state.stock_data = stock_data
                #st.session_state.stock_analysis = analyze_stock_with_gpt(ticker, stock_data, news)
                st.session_state.news_sentiment = news
                st.session_state.stock_analysis = analyze_fundamentals(ticker)

            # Display historical stock price chart if available
            if st.session_state.stock_data is not None:
                st.subheader("📈 Stock Price History")
                st.bar_chart(st.session_state.stock_data["y"])


# Historical Analysis Section
if menu == "Historical Analysis":

    # Function to fetch stock data
    @st.cache_data
    def fetch_data(ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(period="10y")  # Fetch 10 years of data
        return data

    # TradingView widget function
    def tradingview_chart(symbol):
        with st.expander("📉 Click to Expand TradingView Chart"):
            st.markdown(f"""
                <iframe 
                    src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_1&symbol={symbol}&interval=D&theme=dark&style=1&hide_top_toolbar=1&hide_side_toolbar=0&allow_symbol_change=1&save_image=0&watchlist=stocks&calendar=1"
                    width="100%" height="400px" style="border: none;">
                </iframe>
            """, unsafe_allow_html=True)

    # App UI
    st.title("📉 Historical Analysis")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")

    # Fetch data
    data = fetch_data(ticker)

    # Display TradingView Chart
    tradingview_chart(ticker)

    # Ensure Close price exists
    if "Close" in data.columns and not data["Close"].isnull().all():
        # Title
        st.markdown("### 📊 Year Returns")

        # Create Columns
        col1, col2, col3, col4, col5 = st.columns(5)

        # Fill Columns with Metrics
        with col1:
            st.metric("📈 1-Year Return", f"{data['Close'].pct_change().tail(252).sum() * 100:.2f}%")
        with col2:
            st.metric("📈 3-Year Return", f"{data['Close'].pct_change().tail(252 * 3).sum() * 100:.2f}%")
        with col3:
            st.metric("📈 5-Year Return", f"{data['Close'].pct_change().tail(252 * 5).sum() * 100:.2f}%")
        with col4:
            st.metric("📈 10-Year Return", f"{data['Close'].pct_change().sum() * 100:.2f}%")
        with col5:
            import pytz  # Import timezone library
            # Get the timezone from data.index
            tz = data.index.tz
            # Convert the comparison timestamp to the same timezone
            start_of_year = pd.Timestamp.now(tz=tz).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            # Now compare with the timezone-aware index
            st.metric("📈 Year to Date", f"{data['Close'][data.index >= start_of_year].pct_change().sum() * 100:.2f}%")

                           
        #Create 2 Columns, Multiple Lines in Each Column
        col1, col2 = st.columns(2)

        # Get the current price
        current_price = data['Close'].iloc[-1]

        # Function to style the trend (bullish or bearish)
        def style_trend(trend):
            if trend == "Bullish":
                return f"<b style='color:green;'>{trend}</b>"
            else:
                return f"<b style='color:red;'>{trend}</b>"


        # Moving Averages
        with col1:
            st.write("")  # Empty line
            st.write("")  # Empty line
            st.markdown("### 📊 Metrics and Averages Analysis")
            # Calculate moving averages
            ma_10 = data['Close'].rolling(window=10).mean().iloc[-1]
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            ma_100 = data['Close'].rolling(window=100).mean().iloc[-1]
            ma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
            
            # Evaluate trend (Bullish or Bearish)
            ma_10_trend = "Bullish" if current_price > ma_10 else "Bearish"
            ma_50_trend = "Bullish" if current_price > ma_50 else "Bearish"
            ma_100_trend = "Bullish" if current_price > ma_100 else "Bearish"
            ma_200_trend = "Bullish" if current_price > ma_200 else "Bearish"

            # Display metrics with trend evaluation
            st.markdown(f"📊 10-Day Moving Average: {ma_10:.2f} ({style_trend(ma_10_trend)})", unsafe_allow_html=True)
            st.markdown(f"📊 50-Day Moving Average: {ma_50:.2f} ({style_trend(ma_50_trend)})", unsafe_allow_html=True)
            st.markdown(f"📊 100-Day Moving Average: {ma_100:.2f} ({style_trend(ma_100_trend)})", unsafe_allow_html=True)
            st.markdown(f"📊 200-Day Moving Average: {ma_200:.2f} ({style_trend(ma_200_trend)})", unsafe_allow_html=True)
            #Display the Rsi with Bulish or Bearish trend
            data = calculate_indicators(data)  # Ensure RSI is calculated
            rsi = data['RSI'].iloc[-1]
            rsi_trend = "Bullish" if rsi < 30 else "Bearish" if rsi > 70 else "Neutral"
            st.markdown(f"📈 RSI: {rsi:.2f} ({style_trend(rsi_trend)})", unsafe_allow_html=True)
            #Display the MACD with Bulish or Bearish trend
            macd = data['MACD'].iloc[-1]
            macd_trend = "Bullish" if macd > 0 else "Bearish"
            st.markdown(f"📈 MACD: {macd:.2f} ({style_trend(macd_trend)})", unsafe_allow_html=True)

            st.write("")  # Empty line
            
            # Display the Volume and Average Volume
            average_volume_20 = data['Volume'].tail(200).mean()
            # Check if the current close price is higher than the previous close (Bullish)
            data['Bullish'] = data['Close'] > data['Close'].shift(1)

            # Check if the current close price is lower than the previous close (Bearish)
            data['Bearish'] = data['Close'] < data['Close'].shift(1)

            # Define your style_trend color for Bullish and Bearish (you can adjust the color as per your existing style)
            bullish_color = "green"  # Assuming style_trend for bullish is green
            bearish_color = "red"    # Assuming style_trend for bearish is red

            # Determine the trend
            trend = 'Bullish' if data['Bullish'].iloc[-1] else 'Bearish'
            trend_color = bullish_color if data['Bullish'].iloc[-1] else bearish_color

            # Display if today is bullish or bearish with the style_trend color
            st.markdown(f"""
                <p style="color:{trend_color}; font-weight:bold;">
                    📊 Today's Price Movement: {trend}
                </p>
            """, unsafe_allow_html=True)

            # Example for 10-Day Moving Average
            is_bullish_10_day = data['Close'].iloc[-1] > data['Close'].iloc[-11]
            st.markdown(f"📊 10-Day Moving Average: {data['Close'].iloc[-1]:.2f} ({style_trend(is_bullish_10_day)})", unsafe_allow_html=True)

            # Example for 50-Day Moving Average
            is_bullish_50_day = data['Close'].iloc[-1] > data['Close'].iloc[-51]
            st.markdown(f"📊 50-Day Moving Average: {data['Close'].iloc[-1]:.2f} ({style_trend(is_bullish_50_day)})", unsafe_allow_html=True)

            # Example for 200-Day Moving Average
            is_bullish_200_day = data['Close'].iloc[-1] > data['Close'].iloc[-201]
            st.markdown(f"📊 200-Day Moving Average: {data['Close'].iloc[-1]:.2f} ({style_trend(is_bullish_200_day)})", unsafe_allow_html=True)

            # Evaluate if the current price movement is accompanied by higher volume than the average volume
            is_bullish_high_volume = (data['Bullish'].iloc[-1] and data['Volume'].iloc[-1] > average_volume_20)
            is_bearish_high_volume = (data['Bearish'].iloc[-1] and data['Volume'].iloc[-1] > average_volume_20)

            # Display whether the movement has high volume
            st.write(f"📊 Bullish with High Volume: {'Yes' if is_bullish_high_volume else 'No'}")
            st.write(f"📊 Bearish with High Volume: {'Yes' if is_bearish_high_volume else 'No'}")

        with col2:
            with st.container():  # Wrap the code in a container to avoid extra spacing
                vix_value = get_vix()
                signal, color = vix_indicator(vix_value)
                
                st.write("")  # Empty line
                st.write("")  # Empty line
                st.subheader("📊 VIX Indicator")
                if vix_value:
                    st.metric(label="VIX Index Level", value=round(vix_value, 2))
                    st.markdown(f"<h3 style='color:{color};'>{signal}</h3>", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Could not fetch VIX data.")

            #Display the Fibonacci Retracement
            st.write("")  # Empty line
            st.markdown(f"📈 Fibonacci Retracement Levels:")

            period = st.selectbox('Select period', ['1 Day', '1 Week', '1 Month', '3 Months', '1 Year'], index=4)

            if period == '1 Day':
                data_filtered = data.tail(1)
            elif period == '1 Week':
                data_filtered = data.tail(5)
            elif period == '1 Month':
                data_filtered = data.tail(30)
            elif period == '3 Months':
                data_filtered = data.tail(90)
            elif period == '1 Year':
                data_filtered = data.tail(252)

            high = data_filtered['High'].max()
            low = data_filtered['Low'].min()

            # Calculate Fibonacci retracement levels as usual
            fib_23_6 = low + (high - low) * 0.236
            fib_38_2 = low + (high - low) * 0.382
            fib_50 = low + (high - low) * 0.5
            fib_61_8 = low + (high - low) * 0.618
            fib_100 = high
            # Display the Fibonacci retracement levels
            st.markdown(f"   - 100%: {fib_100:.2f}")
            st.markdown(f"   - 61.8%: {fib_61_8:.2f}")
            st.markdown(f"   - 50.0%: {fib_50:.2f}")
            st.markdown(f"   - 38.2%: {fib_38_2:.2f}")
            st.markdown(f"   - 23.6%: {fib_23_6:.2f}")
    else:
        st.error("No valid price data available for calculations.")

# Monte Carlo Simulations Section
if menu == "Monte Carlo Simulations":
    st.title("🎲 Monte Carlo Simulations")
    
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    data, info = fetch_data(ticker)

    # User Inputs
    n_simulations = st.slider("Number of Simulations", 100, 10000, 5000)
    n_years = st.slider("Projection Period (Years)", 0, 10, 3)
    n_months = st.slider("Additional Months", 0, 11, 1)  # New: Add months
    log_normal = st.checkbox("Use Log-Normal Distribution")
    manual_vol = st.checkbox("Manually Adjust Volatility")

    volatility = None
    if manual_vol:
        volatility = st.slider("Set Volatility (%)", 0.5, 5.0, data['Close'].pct_change().std() * 100) / 100

    # Run Monte Carlo Simulation
    # Convert years & months into total trading days
    total_days = (n_years * 252) + (n_months * 21)  # ✅ Includes months
    # Run Monte Carlo Simulation with new total_days
    simulations = monte_carlo_simulation(data, n_simulations, total_days, log_normal, volatility)
    last_price = data['Close'].iloc[-1]
    final_prices = simulations[:, -1]

    # Compute Statistics
    mean_price = np.mean(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_25 = np.percentile(final_prices, 25)
    percentile_95 = np.percentile(final_prices, 95)
    prob_price_increase = np.sum(final_prices > last_price) / len(final_prices) * 100

    # Display Key Metrics
    st.markdown("### 📊 Simulation Results")
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2.5, 2])

    with col1:
        st.metric(label="🔹 Current Price", value=f"${last_price:.2f}")

    with col2:
        st.metric(label="📈 Expected Price", value=f"${mean_price:.2f}")

    with col3:
        st.metric(label="🚀 Discount-Case (25%)", value=f"${percentile_25:.2f}")

    with col4:
        st.metric(label="🚀 Best-Case (95%)", value=f"${percentile_95:.2f}")

    with col5:
        st.metric(label="📉 Worst-Case (5%)", value=f"${percentile_5:.2f}")

    st.markdown(f"**Probability of Price Increase: {prob_price_increase:.2f}%**")

    st.subheader("Simulated Price Paths")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample a subset of simulations (max 100)
    sample_size = min(n_simulations, 100)
    sample_indices = np.random.choice(n_simulations, sample_size, replace=False)

    for i in sample_indices:
        ax.plot(simulations[i], alpha=0.3, linewidth=0.8)

    # Compute and plot mean path + confidence intervals
    mean_path = np.mean(simulations, axis=0)
    percentile_5_path = np.percentile(simulations, 5, axis=0)
    percentile_95_path = np.percentile(simulations, 95, axis=0)

    ax.plot(mean_path, color="black", linewidth=2, label="Mean Projection")
    ax.fill_between(range(simulations.shape[1]), percentile_5_path, percentile_95_path, color='gray', alpha=0.3, label="5%-95% Confidence Interval")

    # Set labels and title
    ax.set_title("Monte Carlo Simulations")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()

    # Pass figure to Streamlit
    with st.expander("📊 Click to Expand Simulation Chart"):
        st.pyplot(fig)

if menu == "Market Analysis | Buy Signals":
    st.title("📈 Market Analysis | Buy Signals")

    # Define tickers
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "VIX": "^VIX"
    }

    # Display current market prices
    with st.expander("📊 Current Market Prices", expanded=True):
        cols = st.columns(len(tickers))
        for i, (name, symbol) in enumerate(tickers.items()):
            data = yf.Ticker(symbol).history(period="1d")
            current_price = data["Close"].iloc[-1] if not data.empty else "N/A"

            # Handle VIX differently
            if symbol == "^VIX":
                vix_value = current_price
                if isinstance(vix_value, float):
                    category, color = vix_indicator(vix_value)
                    cols[i].write(f"<p style='color:{color}; font-size:22px;'>{name}: {vix_value:.2f}</p>", unsafe_allow_html=True)
                    cols[i].markdown(f'<p style="color:{color}; font-size:22px;">{category}</p>', unsafe_allow_html=True)
                else:
                    cols[i].write(f"**{name}**: {current_price}")
            else:
                # Format S&P 500 and Nasdaq 100
                cols[i].metric(label=name, value=f"${current_price:,.2f}" if isinstance(current_price, float) else current_price)

    # Display market indicators
    with st.expander("📈 Market Indicators (S&P 500 & Nasdaq 100)"):
        col1, col2 = st.columns(2)

        def show_indicators(ticker, title):
            data = yf.Ticker(ticker).history(period="10y")
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return

            close = data["Close"]

            # Find the first available trading day of the year
            try:
                # Ensure start_of_year is timezone-aware and matches the timezone of close.index
                start_of_year = pd.Timestamp.now(tz=close.index.tz).replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                start_price = close.loc[close.index >= start_of_year].iloc[0] if not close.loc[close.index >= start_of_year].empty else close.iloc[0]
            except Exception as e:
                st.error(f"Error during selection: {e}")
                start_price = close.iloc[0]

            # Calculate YTD % return
            ytd = ((close.iloc[-1] / start_price) - 1) * 100

            # Calculate RSI, MACD, and Fibonacci levels
            rsi = compute_rsi(close)
            macd, signal = compute_macd(close)
            fib_level_3y = compute_fibonacci_level(close[-252*3:])  # Last 3 years
            fib_level_5y = compute_fibonacci_level(close[-252*5:])  # Last 5 years
            fib_level_10y = compute_fibonacci_level(close)  # Last 10 years

            #Calculate and categorize MACD signal into a variable
            macd_signal = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
            macd_color = "green" if macd_signal == "Bullish" else "red"

            #Calculate and categorize RSI signal for markdown incorporation
            rsi_signal = None
            rsi_color = None
            if rsi < 30:
                rsi_signal = "Bullish"
                rsi_color = "green"
            elif rsi > 70:
                rsi_signal = "Bearish"
                rsi_color = "red"
            else:
                rsi_signal = "Neutral"
                rsi_color = "yellow"
            if isinstance(rsi, (int, float)):
                rsi = round(rsi, 2)
            else:
                rsi = "N/A"

            #Calculate and classify the YTD % if with Correction, Bear Market, Crash or Bull Market
            ytd_signal = None
            ytd_color = None
            if ytd > 0:
                ytd_signal = "Bull Market"
                ytd_color = "green"
            elif -10 < ytd <= 0:
                ytd_signal = "Correction"
                ytd_color = "orange"
            elif -20 < ytd <= -10:
                ytd_signal = "Bear Market"
                ytd_color = "red"
            else:
                ytd_signal = "Crash"
                ytd_color = "darkred"

            st.subheader(title)
            st.markdown(f"""
            - **52 Week High**: ${close[-252:].max():,.2f}
            - **52 Week Low**: ${close[-252:].min():,.2f}
            - **RSI**: {rsi:.2f} (<span style='color:{rsi_color}; font-size:18px;'> {rsi_signal}</span>)
            - **MACD Signal**: {signal.iloc[-1]:.2f} (<span style='color:{macd_color}; font-size:18px;'> {macd_signal}</span>)
            - **YTD %**: {ytd:.2f}% (<span style='color:{ytd_color}; font-size:18px;'> {ytd_signal}</span>)
            - **1D %**: {close.pct_change().iloc[-1]*100:.2f}%
            - **5D %**: {close.pct_change(5).iloc[-1]*100:.2f}%
            - **1M %**: {close.pct_change(21).iloc[-1]*100:.2f}%
            - **6M %**: {close.pct_change(126).iloc[-1]*100:.2f}%
            - **1Y %**: {close.pct_change(252).iloc[-1]*100:.2f}%
            - **5Y %**: {close.pct_change(1260).iloc[-1]*100:.2f}%
            - **Fibonacci Level (3Y Range)**: {fib_level_3y:.2f}%
            - **Fibonacci Level (5Y Range)**: {fib_level_5y:.2f}%
            - **Fibonacci Level (10Y Range)**: {fib_level_10y:.2f}%
            """, unsafe_allow_html=True)

        with col1:
            show_indicators("^GSPC", "S&P 500 Indicators")

        with col2:
            show_indicators("^NDX", "Nasdaq 100 Indicators")

        
        
    with st.expander("📈 Monthly Performance Analysis", expanded=True):
        @st.cache_data
        def fetch_monthly_returns(ticker):
            # Fetch data from Yahoo Finance
            data = yf.download(ticker, period="10y", interval="1d", progress=False)
            
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return pd.DataFrame()

            # Resample to monthly frequency (closing prices)
            monthly_data = data['Close'].resample('M').ffill()
            
            # Calculate percentage change to get returns
            monthly_returns = monthly_data.pct_change().dropna()

            # Check if monthly_returns is a Series and convert to DataFrame
            if isinstance(monthly_returns, pd.Series):
                df = monthly_returns.to_frame(name='Monthly Return')
            else:
                df = monthly_returns.rename(columns={monthly_returns.columns[0]: 'Monthly Return'})

            # Add 'Year' and 'Month' columns for easy filtering
            df['Year'] = df.index.year
            df['Month'] = df.index.month

            return df

        def analyze_monthly_performance(monthly_returns):
            current_year = datetime.now().year
            current_month = datetime.now().month

            current_month_data = monthly_returns[
                (monthly_returns['Year'] == current_year) & 
                (monthly_returns['Month'] == current_month)
            ]

            if current_month_data.empty:
                current_performance = None
            else:
                current_performance = current_month_data['Monthly Return'].values[0]

            historical_max = monthly_returns['Monthly Return'].max()
            historical_min = monthly_returns['Monthly Return'].min()

            if current_performance is not None:
                if current_performance == historical_max:
                    category = 'Highest'
                elif current_performance == historical_min:
                    category = 'Lowest'
                else:
                    category = 'Neutral'
            else:
                category = 'No Data'

            return current_performance, historical_max, historical_min, category

        def display_monthly_performance(ticker, title):
            monthly_returns = fetch_monthly_returns(ticker)
            if monthly_returns.empty or 'Monthly Return' not in monthly_returns.columns:
                st.error(f"Could not fetch data for {ticker}")
                return

            current_performance, historical_max, historical_min, category = analyze_monthly_performance(monthly_returns)

            st.subheader(f"{title} - Monthly Performance")

            if current_performance is not None:
                st.write(f"**Current Month Performance**: {current_performance * 100:.2f}%")
                st.write(f"**Historical Max Monthly Return**: {historical_max * 100:.2f}%")
                st.write(f"**Historical Min Monthly Return**: {historical_min * 100:.2f}%")
                st.write(f"**Category**: {category}")
            else:
                st.write("No data available for the current month.")

        def display_yearly_performance(ticker, title):
            import yfinance as yf
            import pandas as pd
            from datetime import datetime

            # Fetch historical data for the past 10 years
            data = yf.download(ticker, period="10y", interval="1d", progress=False)
            if data.empty:
                st.error(f"Could not fetch data for {ticker}")
                return

            # Resample to yearly frequency and calculate yearly returns
            yearly_data = data['Close'].resample('Y').ffill()
            yearly_returns = yearly_data.pct_change().dropna()
            yearly_returns.index = yearly_returns.index.year

            # Ensure column is named properly
            if isinstance(yearly_returns, pd.Series):
                yearly_returns = yearly_returns.to_frame()
                yearly_returns.columns = ['Yearly Return']
            else:
                yearly_returns.columns = ['Yearly Return']

            # Get current year
            current_year = datetime.now().year

            if current_year in yearly_returns.index:
                current_performance = yearly_returns.loc[current_year, 'Yearly Return']
            else:
                current_performance = None

            historical_max = yearly_returns['Yearly Return'].max()
            historical_min = yearly_returns['Yearly Return'].min()

            # Categorize current performance
            if current_performance is not None:
                if current_performance == historical_max:
                    category = 'Highest'
                elif current_performance == historical_min:
                    category = 'Lowest'
                else:
                    category = 'Neutral'
            else:
                category = 'No Data'

            st.subheader(f"{title} - Yearly Performance")

            if current_performance is not None:
                st.write(f"**Current Year Performance**: {current_performance * 100:.2f}%")
                st.write(f"**Historical Max Yearly Return**: {historical_max * 100:.2f}%")
                st.write(f"**Historical Min Yearly Return**: {historical_min * 100:.2f}%")
                st.write(f"**Category**: {category}")
            else:
                st.write("No data available for the current year.")


    st.title("Market Performance Analysis - Last 10 Years")

    # S&P 500
    display_monthly_performance("^SPX", "S&P 500")
    display_yearly_performance("^SPX", "S&P 500")

    # Nasdaq 100
    display_monthly_performance("^NDX", "Nasdaq 100")
    display_yearly_performance("^NDX", "Nasdaq 100")


# Export Data Section
if menu == "Export Data":
    st.title("📥 Export Data")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    data, info = fetch_data(ticker)
    
    st.download_button("Download Data as CSV", data.to_csv(index=True), file_name=f"{ticker}_historical_data.csv")

# Refined Strategy Section (RSI with Trend Confirmation)
if menu == "Refined Strategy (RSI with Trend)":
    st.title("📊 Refined RSI Buy and Sell Strategy with Trend Confirmation")

    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    data, info = fetch_data(ticker)

    # Get RSI Strategy Signals with Trend Confirmation
    signals = generate_signals(data)

    # Plotting the stock price and buy/sell signals
    st.subheader("Stock Price and Buy/Sell Signals")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Stock Price', color='blue')
    ax.plot(data['SMA50'], label='50-Day SMA', color='orange')

    # Mark Buy and Sell signals on the plot
    ax.plot(signals.index[signals['Signal'] == 1], data['Close'][signals['Signal'] == 1], '^', markersize=10, color='green', lw=0, label='Buy Signal')
    ax.plot(signals.index[signals['Signal'] == -1], data['Close'][signals['Signal'] == -1], 'v', markersize=10, color='red', lw=0, label='Sell Signal')
    
    ax.set_title(f"{ticker} Stock Price with Buy/Sell Signals (RSI with Trend)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    st.pyplot(fig)

# Machine Learning Strategy
if menu == "Machine Learning Strategy":
    st.title("📊 Machine Learning Buy/Sell Strategy")

    ticker = st.text_input("Enter Stock Ticker for ML Strategy", "AAPL")
    data, info = fetch_data(ticker)

    # Get RSI Strategy Signals with Trend Confirmation
    signals = generate_signals(data)

    # Train Machine Learning Model
    model, X_test, y_test, y_pred = train_model(data)

    # Plotting the stock price and ML-based buy/sell signals
    st.subheader("Stock Price and ML-based Buy/Sell Signals")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Stock Price', color='blue')
    ax.plot(data['SMA50'], label='50-Day SMA', color='orange')

    # Mark ML-based Buy and Sell signals
    ax.plot(X_test.index, data['Close'].iloc[X_test.index][y_pred == 1], '^', markersize=10, color='green', lw=0, label='ML Buy Signal')
    ax.plot(X_test.index, data['Close'].iloc[X_test.index][y_pred == -1], 'v', markersize=10, color='red', lw=0, label='ML Sell Signal')

    ax.set_title(f"{ticker} Stock Price with ML Buy/Sell Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    st.pyplot(fig)