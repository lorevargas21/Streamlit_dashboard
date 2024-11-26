#Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import requests
import urllib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Extrat information of Yahoo finance, using professor Minh code 
class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

#==============================================================================
# Header and update button 
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Subtitle
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    
    # Add dashboard title and description
    st.title("FINANCIAL DASHBOARD by Lorena Vargas")
    st.subheader("Final Project FINANCIAL PROGRAMMING ")
    col1, col2 = st.columns([1,5])
    col1.write("Data source:")
    col2.image('./Desktop/yahoo_finance.jpg', width=100)
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    @st.cache_data
    def load_sp500_stocks():
        ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
        ticker = st.selectbox("Select a stock ticker", ticker_list)


    @st.cache_data
    def fetch_stock_data(ticker):
        stock_data=yf.download(ticker, period="1y")
        return stock_data

    if st.sidebar.button("Update"):
        st.session_state['stock_data']=fetch_stock_data
    
    if 'stock_data' in st.session_state:
        updated_stock_data=st.session_state['stock_data']
        

    

    
#==============================================================================
# Tab 1
#==============================================================================
def render_tab1():
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """
       
    # Show to stock image
    col1, col2, col3 = st.columns([1, 3, 1])
    col2.image('./Desktop/finalproyect.jpg', use_column_width=True,
               caption='Company Stock Information')
    
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """        
        stock=yf.Ticker(ticker)
        return YFinance(ticker).info 

    @st.cache_data
    def get_stock_history(ticker, period="1mo"):
        stock = yf.Ticker(ticker)
        return stock.history(period=period, interval="1d")   
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    ticker = st.selectbox("Select a stock ticker", ticker_list)
    if ticker:
        st.session_state.ticker = ticker
    
    # If the ticker is already selected ticker 
    if ticker!='':
        # Get the company information in list format
        info= GetCompanyInfo(ticker)
        
        # Show the company description using markdown + HTML
        st.write('**1. Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        
        # Show some statistics as a DataFrame
        st.write('**2. Key Statistics:**')
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume'}
        company_stats = {}  # Dictionary
        for key in info_keys:
            company_stats.update({info_keys[key]:info[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        st.dataframe(company_stats)

    st.write('**3. Stock Price Chart:**')

    period_options = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
        "Year to Date": "ytd", "1 Year": "1y", "3 Years": "3y", "5 Years": "5y", "Max": "max"
    }
    selected_period = st.selectbox("Select time period", list(period_options.keys()))

    stock_data = get_stock_history(ticker, period=period_options[selected_period])
    fig = go.Figure(
        go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price')
    )
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Close Price",
    )
    st.plotly_chart(fig)

#==============================================================================
# Tab 2
#==============================================================================
def render_tab2():
    @st.cache_data
    def GetStockData(ticker, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df

    if 'ticker' in st.session_state:
        ticker = st.session_state.ticker
    else:
        ticker = None
    start_date = st.date_input("Start date", pd.to_datetime("2023-01-01").date())  # Default start date
    end_date = st.date_input("End date", pd.to_datetime("today").date())  # Default end date (today)
    
    # If the ticker name is selected and the check box is checked, show data
    show_data = st.checkbox("Show data table")
    if ticker!= '':
        stock_price = GetStockData(ticker,start_date, end_date)
        if show_data:
            st.write('**Stock price data**')
            st.dataframe(stock_price, hide_index=True, use_container_width=True)  
    # Initiate the plot figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

# Stock price area plot
        area_plot = go.Scatter(x=stock_price['Date'], y=stock_price['Close'],
                       fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
        fig.add_trace(area_plot, secondary_y=True)

# Stock volume bar plot
        bar_plot = go.Bar(x=stock_price['Date'], y=stock_price['Volume'], marker_color=np.where(stock_price['Close'].pct_change() < 0, 'red', 'green'),
                  showlegend=False)
        fig.add_trace(bar_plot, secondary_y=False)

#Moving Average(50-DAY)
        stock_price['SMA50'] = stock_price['Close'].rolling(window=50).mean()
        sma_plot = go.Scatter(
        x=stock_price['Date'],
        y=stock_price['SMA50'],
        mode='lines',
        name='50-Day SMA',
        line=dict(color='orange', dash='dash')
        )
        fig.add_trace(sma_plot, secondary_y=True)

# Add range selector buttons
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
            ])
        )
    )

# Customize the layout
        fig.update_layout(template='plotly_white')
        fig.update_yaxes(range=[0, stock_price['Volume'].max()*1.1], secondary_y=False)
        st.plotly_chart(fig)

#==============================================================================
# Tap 3
#==============================================================================
def render_tab3():
    if 'ticker' in st.session_state:
        ticker = st.session_state.ticker
    else:
        ticker = None

    # User input: Financial statement (select box to choose between Income Statement, Balance Sheet, Cash Flow)
    statement = st.selectbox("Select Financial Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])

    # User input: Period (select box to choose between Annual or Quarterly)
    period = st.selectbox("Select Period", ["Annual", "Quarterly"])
    
    @st.cache_data
    def get_financial_data(ticker, statement, period):
        stock= yf.Ticker(ticker)
        if statement == "Income Statement":
            if period == "Annual":
                return stock.financials
            else:
                return stock.quarterly_financials
        elif statement == "Balance Sheet":
            if period == "Annual":
                return stock.balance_sheet
            else:
                return stock.quarterly_balance_sheet
        elif statement == "Cash Flow":
            if period == "Annual":
                return stock.cashflow
            else:
                return stock.quarterly_cashflow
    
    
# Fetch and display financial data
    financial_data = get_financial_data(ticker, statement, period)

# Display selected financial data
    st.subheader(f"{statement} - {period}")
    if financial_data is not None:
        st.write(financial_data)
    else:
        st.write("Financial data not available for the selected options.")

#==============================================================================
# Tap 4
#==============================================================================
def render_tab4():
    if 'ticker' in st.session_state:
        ticker = st.session_state.ticker
    else:
        ticker = None

    simulations = st.selectbox("Select number of simulations:", options=[200,500,1000], index=2)

    # User input: Period (select box to choose between Annual or Quarterly)
    time_horizon = st.selectbox("Select time horizon(days)", options=[30,60,90],index=0)
    
    @st.cache_data
    def get_stock_data(ticker, period="1y"):
        stock = yf.Ticker(ticker)
        return stock.history(period=period)

    stock_data = get_stock_data(ticker, period="1y")
#Calculate daily volatility 
    close_price=stock_data["Close"]
    daily_volatility=np.std(close_price.pct_change())

    last_price=close_price.iloc[-1]
#Montecarlo simulation 
    np.random.seed(123)
    simulation_results=np.zeros((time_horizon, simulations))

    # Run the simulation
    simulation_df = pd.DataFrame(simulation_results)

    for n in range(simulations):
        #last_price=close_price.iloc[-1,:]
        prices=[last_price]
        for day in range(time_horizon):
    # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)
    # Generate the random future price
            future_price = prices[-1] * (1 + future_return)
    # Save the price and go next
            prices.append(future_price)
            
        simulation_results[:, n ]=prices[1:]
        

#VaR at 95% confidence level
    final_prices=simulation_df.iloc[-1,:]
    var_95=np.percentile(final_prices,5)

#display VaR
    st.subheader(f"Value at Risk (VaR) at 95% confidence level: ${var_95:,.2f}")
       
# Plot the simulation stock price in the future
    fig, ax = plt.subplots(figsize=(15, 10))
# Plot the prices
    ax.plot(simulation_df)
    ax.axhline(y=last_price, color='red')
# Customize the plot
    ax.set_title('Monte Carlo simulation')
    ax.set_xlabel('Day')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

#==============================================================================
# Tap 5
#==============================================================================
def render_tab5():
    if 'ticker' in st.session_state:
        ticker = st.session_state.ticker
    else:
        ticker = None
    # Function to fetch analyst ratings and price target data
    @st.cache_data
    def fetch_analyst_ratings(ticker):
        try:
        # Get stock information using yfinance
            stock = yf.Ticker(ticker)
            recommendations=stock.recommendations
            if recommendations is None:
                return pd.DataFrame()
            
        # Filter and format recent recommendations
            latest_recommendations = recommendations.tail(10)  # Last 10 recommendations
            latest_recommendations.reset_index(inplace=True)
            latest_recommendations.rename(columns={"Date": "Date", "Firm": "Analyst Firm", "To Grade": "New Rating", "From Grade": "Previous Rating", "Action": "Action"}, inplace=True)
        
        # Fetch target price summary
            price_target_data = stock.info
            price_target = {
                "Current Price": price_target_data.get("currentPrice", "N/A"),
                "Target High": price_target_data.get("targetHighPrice", "N/A"),
                "Target Low": price_target_data.get("targetLowPrice", "N/A"),
                "Target Mean": price_target_data.get("targetMeanPrice", "N/A"),
                "Target Median": price_target_data.get("targetMedianPrice", "N/A")
        }
            price_target_summary = pd.DataFrame([price_target])

        # Return the data
            return latest_recommendations, price_target_summary
        except Exception as e:
            st.error(f"Error for ticker {ticker}")
            return pd.DataFrame()

    latest_recommendations, price_targets=fetch_analyst_ratings(ticker)

    st.subheader(f"Analyst Ratings for {ticker}")
    st.write("Recent Analyst Recommendations:")
    st.dataframe(latest_recommendations)
    
    # Render price target summary
    st.subheader("Price Target Summary")
    st.write(price_targets)
    

#==============================================================================
# Main body
#==============================================================================
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5= st.tabs(["Company profile", "Chart", "Financials", "Monte Carlo simulation", "Recommendations"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4: 
    render_tab4()
with tab5:
    render_tab5()

st.markdown(
    """
    <style>
        .stApp {
            background: #F3F6F9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)