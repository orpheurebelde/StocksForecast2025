import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from yahooquery import search

# Set app to wide mode
st.set_page_config(layout="wide")

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
    """Fetch historical stock data and company info from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="10y")  # Fetch 10 years of data
    info = stock.info 

    if data.index.tz is not None:  # Remove timezone info if present
        data.index = data.index.tz_localize(None)

    return data, info

def dcf_valuation(ticker, years=10, manual_growth=None, manual_terminal_growth=None):
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get EPS: Prefer Forward EPS but fallback to Trailing EPS
        eps = stock.info.get("forwardEps", stock.info.get("trailingEps", 0))

        # Get shares outstanding
        shares_outstanding = info.get("sharesOutstanding", 1)

        # Get Cash Flow Statement
        cashflow_df = stock.cashflow

        # Extract FCF components safely
        try:
            operating_cash_flow = cashflow_df.loc["Total Cash From Operating Activities"].iloc[0]
            capital_expenditures = cashflow_df.loc["Capital Expenditures"].iloc[0]
            fcf = operating_cash_flow - capital_expenditures
        except (KeyError, IndexError):
            fcf = None  # Set to None if missing

        # Use last known FCF instead of EPS if missing
        if fcf is None:
            fcf = eps * shares_outstanding  # Approximate using EPS

        # Convert to FCF per share
        fcf_per_share = fcf / shares_outstanding

        # Get analyst growth estimate or allow manual input
        analyst_growth = info.get("earningsGrowth", 0.10)  # Default to 10% if missing
        growth_rate = manual_growth if manual_growth else analyst_growth

        # Use default terminal growth rate if not provided manually
        terminal_growth = manual_terminal_growth if manual_terminal_growth else 0.04

        # Set Discount Rate (WACC approximation), default to 8%
        discount_rate = info.get("costOfCapital", 0.08)  # Default to 8% if missing

        # Calculate future cash flows
        future_cash_flows = [fcf_per_share * (1 + growth_rate) ** i for i in range(1, years + 1)]
        
        # Discount future cash flows to present value
        dcf_value = sum([cf / (1 + discount_rate) ** i for i, cf in enumerate(future_cash_flows, 1)])

        # Add Terminal Value (TV) using the Gordon Growth Model
        terminal_value = (future_cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        terminal_value_discounted = terminal_value / ((1 + discount_rate) ** years)
        
        # Final DCF Value per share
        intrinsic_value = (dcf_value + terminal_value_discounted)

        return round(intrinsic_value, 2)

    except Exception as e:
        st.error(f"❌ Error calculating DCF for {ticker}: {e}")
        return None  # Return None instead of crashing

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
    data['Buy Signal'] = ((data['RSI'] < 30) & 
                           (data['MACD'] > data['MACD_signal']) & 
                           (data['Close'] > data['SMA50']))
    
    data['Sell Signal'] = ((data['RSI'] > 70) & 
                            (data['MACD'] < data['MACD_signal']) & 
                            (data['Close'] < data['SMA50']))
    
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
        return "🟢 BUY - Low Volatility (Bullish)", "green"
    elif 15 <= vix_value <= 25:
        return "🟡 NEUTRAL - Moderate Volatility", "yellow"
    else:
        return "🔴 SELL - High Volatility (Bearish)", "red"

# Sidebar Menu
st.sidebar.title("📈 Stock Info Dashboard")
menu = st.sidebar.radio("Select a Section", ["Stock Info", "Historical Analysis", "Monte Carlo Simulations", "Export Data", "Sector Map", "Refined Strategy (RSI with Trend)"])

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
    st.markdown("### Search for a Stock Ticker")
    st.markdown("You can search by Ticker or Company Name (e.g., AAPL, Apple Inc.)")

    # Input box for user to type the stock name, with a default value
    search_input = st.text_input("Enter Stock Ticker or Name", "Apple")

    # Initialize selected ticker
    selected_ticker = None

    if search_input:
        # Dynamically fetch matching tickers
        ticker_options = get_ticker_from_name(search_input)

        if ticker_options:
            # Display matching options as clickable buttons
            st.markdown("### Matching Companies:")
            for company_name, ticker in ticker_options.items():
                if st.button(f"{ticker} - {company_name}"):
                    selected_ticker = ticker
                    break
        else:
            st.warning("No matching stocks found. Please refine your search.")
    else:
        st.info("Please enter a stock name or ticker to search.")

    # Fetch and display stock data if a ticker is selected
    if selected_ticker:
        data, info = fetch_data(selected_ticker)

        # Display stock information
        st.markdown(f"### 📈 {selected_ticker} - {info.get('longName', 'Company Name Not Found')}")
        st.markdown(f"#### Sector: {info.get('sector', 'Sector Not Found')}")
        st.markdown(f"#### Industry: {info.get('industry', 'Industry Not Found')}")
        with st.expander("Company Info", expanded=False):
            st.write(info)

    if info and 'trailingPE' in info and 'earningsGrowth' in info:
        pe_ratio = info.get['trailingPE', 'N/A']
        peg_ratio = info.get('trailingPegRatio', 'N/A')
        earnings_growth = info.get['earningsGrowth']
        forward_pe = info.get('forwardPE', 'N/A')
        freecash_flow = info.get('freeCashflow', 'N/A')
        netincome = info.get('netIncomeToCommon', 'N/A')
        grossmargin = info.get('grossMargins', 'N/A')
        operatingmargin = info.get('operatingMargins', 'N/A')
        profit_margin = info.get('profitMargins', 'N/A')
        institutional_ownership = info.get('heldPercentInstitutions', 'N/A')
        insider_ownership = info.get('heldPercentInsiders', 'N/A')
        trailingeps = info.get('trailingEps', 'N/A')
        forwardeps = info.get('forwardEps', 'N/A')
        revenue = info.get('totalRevenue', 'N/A')
        totaldebt = info.get('totalDebt', 'N/A')
        totalcash = info.get('totalCash', 'N/A')
        revenuegrowth = info.get('revenueGrowth', 'N/A')
        # Calculate DCF Value
        dcf_value = dcf_valuation(ticker)

        with st.expander("Stock Overview", expanded=True):
            col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="📈 Market Cap", value=f"${info['marketCap'] / 1e9:.2f}B")
            st.metric(label="📈 Free Cash Flow", value=f"${freecash_flow / 1e9:.2f}B" if freecash_flow and isinstance(freecash_flow, (int, float)) else "N/A")
            st.metric(label="📈 Net Income", value=f"${netincome / 1e9:.2f}B" if netincome and isinstance(netincome, (int, float)) else "N/A")
            st.metric(label="📈 Gross Margin", value=f"{grossmargin:.2%}" if grossmargin and isinstance(grossmargin, (int, float)) else "N/A")
            st.metric(label="📈 Operating Margin", value=f"{operatingmargin:.2%}" if operatingmargin and isinstance(operatingmargin, (int, float)) else "N/A")
            st.metric(label="📈 Profit Margin", value=f"{profit_margin:.2%}" if profit_margin and isinstance(profit_margin, (int, float)) else "N/A")
            st.metric(label="📈 Earnings Growth", value=f"{earnings_growth:.2%}" if earnings_growth and isinstance(earnings_growth, (int, float)) else "N/A")
            st.metric(label="📈 Dividend Yield", value=f"{info['dividendYield']:.2%}" if info.get('dividendYield') and isinstance(info['dividendYield'], (int, float)) else "N/A")
        with col2:
            st.metric(label="📊 Current Price", value=f"${info['currentPrice']:.2f}")
            try:
                peg_ratio = float(peg_ratio)  # Convert to float if possible
                peg_color = "green" if peg_ratio < 1 else "orange" if 1 <= peg_ratio <= 2 else "red"
            except (ValueError, TypeError):
                peg_color = "gray"  # Default to gray if conversion fails
            if isinstance(peg_ratio, (int, float)):
                st.markdown(f'<div style="color: {peg_color}; font-size: 25px;"><b>📈 PEG Ratio: {peg_ratio:.2f}</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: gray; font-size: 25px;"><b>📈 PEG Ratio: N/A</b></div>', unsafe_allow_html=True)
            #Categorize P/E Ratio, adding Green, Orange and Red colors
            if pe_ratio < 15:       
                st.markdown(f"<span style='color:green; font-size:25px;'>📈 P/E Ratio: {pe_ratio:.2f}</span>", unsafe_allow_html=True)
            elif 15 <= pe_ratio <= 25:
                st.markdown(f"<span style='color:orange; font-size:25px;'>📈 P/E Ratio: {pe_ratio:.2f}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-size:25px;'>📈 P/E Ratio: {pe_ratio:.2f}</span>", unsafe_allow_html=True)
            #Categorize Forward P/E Ratio, adding Green, Orange and Red colors
            if forward_pe < 15:       
                st.markdown(f"<span style='color:green; font-size:25px;'>📈 Forward P/E Ratio: {forward_pe:.2f}</span>", unsafe_allow_html=True)
            elif 15 <= forward_pe <= 25:
                st.markdown(f"<span style='color:orange; font-size: 25px;'>📈 Forward P/E Ratio: {forward_pe:.2f}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-size:25px;'>📈 Forward P/E Ratio: {forward_pe:.2f}</span>", unsafe_allow_html=True)
            #Categorize Debt to Equity Ratio, adding Green, Orange and Red colors
            if totaldebt and totalcash and isinstance(totaldebt, (int, float)) and isinstance(totalcash, (int, float)):
                debt_to_equity = totaldebt / totalcash
                if debt_to_equity < 1:
                    st.markdown(f"<span style='color:green; font-size:25px;'>📈 Debt to Equity Ratio: {debt_to_equity:.2f}</span>", unsafe_allow_html=True)
                elif 1 <= debt_to_equity <= 2:
                    st.markdown(f"<span style='color:orange; font-size: 25px;'>📈 Debt to Equity Ratio: {debt_to_equity:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red;font-size: 25px;'>📈 Debt to Equity Ratio: {debt_to_equity:.2f}</span>", unsafe_allow_html=True)
            else:
                st.metric(label="📈 Debt to Equity Ratio", value="N/A")
            st.write("")  # Empty line
            st.metric(label="📈 Total Debt", value=f"${totaldebt / 1e9:.2f}B" if totaldebt and isinstance(totaldebt, (int, float)) else "N/A")
            st.metric(label="📈 Total Cash", value=f"${totalcash / 1e9:.2f}B" if totalcash and isinstance(totalcash, (int, float)) else "N/A")
            st.metric(label="📉 DCF Valuation", value=f"${dcf_value:,.2f}")
        with col3:
            st.metric(label="📈 Trailing EPS", value=f"${trailingeps:.2f}")
            st.metric(label="📈 Forward EPS", value=f"${forwardeps:.2f}")
            st.metric(label="📈 Revenue", value=f"${revenue / 1e9:.2f}B" if revenue and isinstance(revenue, (int, float)) else "N/A")
            st.metric(label="📈 Revenue Growth", value=f"{revenuegrowth:.2%}" if revenuegrowth and isinstance(revenuegrowth, (int, float)) else "N/A")
            st.metric(label="📈 Institutional Ownership", value=f"{institutional_ownership:.2%}")
            st.metric(label="📈 Insider Ownership", value=f"{insider_ownership:.2%}")
    

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


# Export Data Section
if menu == "Export Data":
    st.title("📥 Export Data")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    data, info = fetch_data(ticker)
    
    st.download_button("Download Data as CSV", data.to_csv(index=True), file_name=f"{ticker}_historical_data.csv")

# Refined Strategy Section (RSI with Trend)
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