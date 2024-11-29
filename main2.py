import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime as dt
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import newsapi
import api
import json
from streamlit_lottie import st_lottie
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import date
import base64

# Set page config
st.set_page_config(page_title="Cryptocurrency Dashboard", layout="wide")


# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #0e1117, #1a1a2e);
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
            
    h2, h3 {
    color: #3498db;
    }

    h1 {
    color:#3498db;
    }
            
    h1 {    
        margin-top: -50px;
        font-size: 4rem;  /* Increase the font size */
    }
    .option-menu-container {
        margin-top: 50px;
    }
    .stSelectbox label {
        color: #3498db;
    }
    .stPlotlyChart {
        background-color: #1a1a2e;
        border-radius: 5px;
        padding: 10px;
    }
    .crypto-card {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .crypto-image {
        width: 50px;
        height: 50px;
        margin-right: 10px;
    }
    .sidebar-header {

        font-size: 54px;
        color: #3498db;
    }
    .lottie-animation {
        width: 100%;
        height: auto;
        max-width: 400px;
        margin: 0 auto;
    }  
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    .typing-title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px; /* Adjust this value as needed */
    }
    
    .typing-title {
        overflow: hidden;
        border-right: .15em solid #3498db;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: 0; /* Remove letter spacing */
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
        display: inline-block; /* This will make the element only as wide as its content */
        max-width: 24ch; /* Adjust this value based on the length of your text */
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #3498db; }
    }
</style>
""", unsafe_allow_html=True)      
model = load_model('Stock Predictions Model.keras')  

# Header
# Load and resize the image
image = Image.open("crypto_coins.png")
image = image.resize((200, 200))  # Adjust size as needed

# Convert the image to base64
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Create the HTML for the image and heading
html_string = f"""
<div style="display: flex; align-items: center; justify-content: center;">
    <img src="data:image/png;base64,{img_str}" style="width:100px; height:100px; margin-right:8px; margin-top:-60px;">
    <h1 style="color: #3498db;">Cryptocurrency Dashboard</h1>
</div>
"""

# Display the image and heading
st.markdown(html_string, unsafe_allow_html=True)
# Sidebar Heading
st.sidebar.markdown("<h1 class='sidebar-header'>Crypto Ticker</h1>", unsafe_allow_html=True)

cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "DOGE-USD"]
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", cryptos)

# Navigation with custom CSS class
st.markdown('<div class="option-menu-container">', unsafe_allow_html=True)
options = option_menu(None, ["Home", "Visual", "Data", "News", "Prediction"], 
    icons=['house', 'graph-up', 'table', 'newspaper', 'lightbulb'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1a1a2e"},
        "icon": {"color": "#3498db", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#3498db"},
        "nav-link-selected": {"background-color": "#3498db"},
    }
)
st.markdown('</div>', unsafe_allow_html=True)
crypto_lottie_urls = {
    "BTC-USD": "https://lottie.host/dc1c00b5-8a68-45b4-b46f-98095cc68415/D3Fyx1Ebvq.json",
    "ETH-USD": "https://lottie.host/a968f28d-ae6f-4e49-b080-c5e61f69e10c/1rv9p9Stgj.json",
    "BNB-USD": "https://cryptologos.cc/logos/bnb-bnb-logo.png",
    "ADA-USD": "https://lottie.host/77277acd-2b16-43fd-a2cc-75f3d0957ce1/fA8R4smCAc.json",
    "DOGE-USD": "https://s2.coinmarketcap.com/static/img/coins/200x200/74.png"
}

def is_lottie_url(url: str) -> bool:
    return url.endswith('.json')

def load_lottieurl(url: str):
    if is_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    else:
        return url  # Return the URL itself for images
def get_data(start_date, end_date):
    data = yf.download(selected_crypto, start=start_date, end=end_date)
    return data

if options == "Home":
    st.subheader(f"{selected_crypto.split('-')[0]} Daily Prices")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        lottie_url = crypto_lottie_urls.get(selected_crypto, "https://lottie.host/dc1c00b5-8a68-45b4-b46f-98095cc68415/D3Fyx1Ebvq.json")
        if is_lottie_url(lottie_url):
            lottie_animation = load_lottieurl(lottie_url)
            st_lottie(lottie_animation, key="lottie", height=300, quality="high")
        else:
            st.image(lottie_url, width=200)
    
    with col2:
        start_date = st.date_input("Start Date", dt(2022, 1, 1))
        end_date = st.date_input("End Date", dt.now())
        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
        else:
            data = get_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            st.dataframe(data.style.highlight_max(axis=0))
    
    with col3:
        st.metric(label="Current Price", value=f"${data['Close'].iloc[-1]:.2f}", delta=f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100:.2f}%")
        st.metric(label="24h Volume", value=f"${data['Volume'].iloc[-1]:,.0f}")
        st.metric(label="Market Cap", value=f"${data['Close'].iloc[-1] * data['Volume'].iloc[-1]:,.0f}")
elif options == "News":
    st.header("Cryptocurrency News")
    
    col1, col2 = st.columns([1, 2])
    # with col1:
    #     lottie_url = crypto_lottie_urls.get(selected_crypto, "https://lottie.host/dc1c00b5-8a68-45b4-b46f-98095cc68415/D3Fyx1Ebvq.json")
    #     if is_lottie_url(lottie_url):
    #         lottie_animation = load_lottieurl(lottie_url)
    #         st_lottie(lottie_animation, key="lottie", height=300, quality="high")
    #     else:
    #         st.image(lottie_url, width=200)
    
    @st.cache_data
    def load_news():
        try:
            khabra = newsapi.cryptonews()  
            for i in khabra['data']:
                try:
                    st.image(i['thumbnail'])
                    st.subheader(i['title'])
                    st.write(i['createdAt'])
                    st.write(i['description'])
                    st.write("Read Full news...")
                    st.write(i['url'])
                    st.markdown("---")
                except Exception as e:
                    st.write(f"An error occurred while displaying a news item: {e}")
        except Exception as e:
            st.error(f"An error occurred while fetching news: {e}")
    
    load_news()
     

elif options == "Visual":
    st.header("Visualizing Cryptocurrency Data")

    col1, col2 = st.columns([1, 2])
    with col1:
        lottie_url = crypto_lottie_urls.get(selected_crypto, "https://lottie.host/dc1c00b5-8a68-45b4-b46f-98095cc68415/D3Fyx1Ebvq.json")
        if is_lottie_url(lottie_url):
            lottie_animation = load_lottieurl(lottie_url)
            st_lottie(lottie_animation, key="lottie", height=300, quality="high")
        else:
            st.image(lottie_url, width=200)
    
    # Date range selection
    start_date = st.sidebar.date_input("Start Date", dt(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", dt.now())
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
    else:
        crypto_data = get_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        with st.container():
            # Candlestick chart
            st.header('Price Movements Over Time ')
            fig_candlestick = go.Figure()
            fig_candlestick.add_trace(go.Candlestick(
                x=crypto_data.index,
                open=crypto_data['Open'],
                high=crypto_data['High'],
                low=crypto_data['Low'],
                close=crypto_data['Close'],
                name='Candlestick',
                increasing_line_color='blue',
                decreasing_line_color='grey'
            ))
            fig_candlestick.update_layout(
                title=f'{selected_crypto} Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600,
                width=1200,
                xaxis_rangeslider_visible=False
            )
            fig_candlestick.update_yaxes(tickprefix='$')
            st.plotly_chart(fig_candlestick)

           # Closing Prices Line Chart
            st.header('Closing Prices Over Time')
            fig_close = go.Figure()
            fig_close.add_trace(go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Close'],
                mode='lines',
                name='Close Prices',
                line=dict(color='blue', width=1)
            ))
            fig_close.update_layout(
                title=f'{selected_crypto} Closing Prices',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600,  # Height of the chart
                width=1200,  # Width of the chart
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_close)

            # 30-day moving average
            st.header('30-Day Moving Average of Closing Prices')
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=crypto_data.index,
                y=crypto_data['Close'].rolling(window=30).mean(),
                mode='lines',
                name='30-day Moving Average',
                line={'color': '#CD00FF'}
            ))
            fig_ma.update_layout(
                title=f'{selected_crypto} 30-day Moving Average',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600,  # Height of the chart
                width=1200,  # Width of the chart
                xaxis_rangeslider_visible=False
            )
            fig_ma.update_yaxes(tickprefix='$')
            st.plotly_chart(fig_ma)

            # Volume traded bar chart
            st.header('Volume Traded Over Time')
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=crypto_data.index,
                y=crypto_data['Volume'],
                name='Volume Traded',
                marker={'color': 'rgba(50, 171, 96, 0.7)'}
            ))
            fig_volume.update_layout(
                title=f'{selected_crypto} Volume Traded',
                xaxis_title='Date',
                yaxis_title='Volume',
                height=600,  # Height of the chart
                width=1200,  # Width of the chart
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_volume)

            
            # Plotting Box Plot
            st.header('Box Plot of High, Close, Low, and Adj Close Prices')
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=crypto_data['High'], name='High'))
            fig_box.add_trace(go.Box(y=crypto_data['Close'], name='Close'))
            fig_box.add_trace(go.Box(y=crypto_data['Low'], name='Low'))
            fig_box.add_trace(go.Box(y=crypto_data['Adj Close'], name='Adj Close'))
            fig_box.update_layout(
                title=f'{selected_crypto} Box Plot',
                xaxis_title='Price Type',
                yaxis_title='Price',
                height=600,  # Height of the chart
                width=1200,  # Width of the chart
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_box)


            # Relative Strength Index (RSI)
            st.header('Relative Strength Index (RSI) Over Time')
            delta = crypto_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=crypto_data.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=1)
            ))
            fig_rsi.update_layout(
                title=f'{selected_crypto} RSI',
                xaxis_title='Date',
                yaxis_title='RSI',
                height=600,  # Height of the chart
                width=1200,  # Width of the chart
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_rsi)

        # Heatmap
            st.header('Correlation Heatmap')
            fig_heatmap, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(crypto_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig_heatmap)

            # Pie chart
            st.header('Volume Proportion Pie Chart')
            crypto_data['Month'] = crypto_data.index.to_period('M')
            volume_by_month = crypto_data.groupby('Month')['Volume'].sum()
            fig_pie = go.Figure(data=[go.Pie(labels=volume_by_month.index.strftime('%Y-%m'), values=volume_by_month)])
            fig_pie.update_layout(title='Proportion of Volume Traded by Month', height=600, width=1200)
            st.plotly_chart(fig_pie)
elif options == "Data":
    st.markdown("<h2 style='text-align: center;'>Cryptocurrency Market Data</h2>", unsafe_allow_html=True)

    # Replace with your API call function to fetch all data
    all_data = api.fetch_data()

    # Create headers with custom styling
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<h3 style='color: #F1C40F ;'>Coin</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='color: #F1C40F;'>Logo</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h3 style='color:#F1C40F;'>Price</h3>", unsafe_allow_html=True)
    with col4:
        st.markdown("<h3 style='color: #F1C40F;'>Volume</h3>", unsafe_allow_html=True)
    with col5:
        st.markdown("<h3 style='color: #F1C40F;'>Market Cap</h3>", unsafe_allow_html=True)

    # Display data
    for i in all_data:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"<div class='crypto-card'><strong>{i['name']}</strong></div>", unsafe_allow_html=True)
            with col2:
                st.image(i['png32'], width=50, output_format='PNG')
            with col3:
                st.markdown(f"<div class='crypto-card'>${i['rate']:.2f}</div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='crypto-card'>${i['volume']:,.0f}</div>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"<div class='crypto-card'>${i['cap']:,.0f}</div>", unsafe_allow_html=True)

elif options == "Prediction":
    today = date.today()
    # st.header("Stock Price Prediction")

    st.markdown("<h2><div class='typing-title'>Stock Market Predictor 📈</div></h2>", unsafe_allow_html=True)
    # st.set_page_config(page_title='your_title',  layout = 'wide', initial_sidebar_state = 'auto')

    stock =st.text_input('Enter Stock Symbol', 'GOOG')
    start = '2020-01-01'
    end = today

    data = yf.download(stock, start ,end)
    st.subheader(stock+' Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)



    # Assuming `model` is your trained model and `time_step` is the number of time steps in your input data
    x_input = data_test_scale[len(data_test_scale) - 100:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()


    lst_output = []
    n_steps = 100  # Assuming 100 time steps
    i = 0
    # Predict tomorrow's price
    pred_days = 2

    # pred_days = 30  # Number of days to predict
    # Loop structure for predicting the next day
    while i < pred_days:

        if len(temp_input) > 100:

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i += 1

        else:

            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(lst_output)

    # Assuming `pred_days` is the number of days to predict
    future_dates = pd.date_range(start=data.index[-1], periods=pred_days + 1)[1:]

    # Create a DataFrame for the predicted prices and dates
    predicted_data = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predicted_prices.reshape(-1)
    })

    # Display the predicted data
    st.subheader('Predicted Stock Prices of Today and Tomorrow')
    st.write(predicted_data)


    # Predict the next 30 days
    # pred_days = 30
    # # Loop structure for predicting the next 30 days
    # while i < pred_days:
    #     ...
    lst_output = []
    n_steps = 100  # Assuming 100 time steps
    i = 0
    pred_days = 30  # Number of days to predict
    while i < pred_days:

        if len(temp_input) > 100:

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i += 1

        else:

            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(lst_output)

    # Assuming `pred_days` is the number of days to predict
    future_dates = pd.date_range(start=data.index[-1], periods=pred_days + 1)[1:]

    # Create a DataFrame for the predicted prices and dates
    predicted_data = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predicted_prices.reshape(-1)
    })

    # Display the predicted data
    st.subheader('Predicted Stock Prices of Next 30 days')
    st.write(predicted_data)



    lst_output = []
    n_steps = 100  # Assuming 100 time steps
    i = 0
    pred_days = 90  # Number of days to predict
    while i < pred_days:

        if len(temp_input) > 100:

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i += 1

        else:

            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(lst_output)

    # Assuming `pred_days` is the number of days to predict
    future_dates = pd.date_range(start=data.index[-1], periods=pred_days + 1)[1:]

    # Create a DataFrame for the predicted prices and dates
    predicted_data = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predicted_prices.reshape(-1)
    })

    # Display the predicted data
    st.subheader('Predicted Stock Prices of Next 90 days')
    st.write(predicted_data)


    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)



    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)


    st.subheader("Last 15 Days and Today/Tomorrow Prediction")
    fig5 = plt.figure(figsize=(8,6))
    sns.lineplot(x=np.arange(1, 16), y=data_test_scale[-15:].flatten(), label='Last 15 Days')
    sns.lineplot(x=np.arange(16, 18), y=predicted_prices[:2].flatten(), label='Today and Tomorrow Prediction')
    plt.xlabel('Day')
    plt.ylabel('Normalized Close Price')
    # plt.title('Last 15 Days and Today/Tomorrow Prediction')
    # plt.legend()
    plt.show()
    st.pyplot(fig5)

    # Plotting next 30 days price prediction
    st.subheader("Last 15 Days and Next 30 Days Prediction")
    fig6 = plt.figure(figsize=(8, 6))
    sns.lineplot(x=np.arange(1, 16), y=data_test_scale[-15:].flatten(), label='Last 15 Days')
    sns.lineplot(x=np.arange(16, 46), y=predicted_prices[:30].flatten(), label='Next 30 Days Prediction')
    plt.xlabel('Day')
    plt.ylabel('Normalized Close Price')
    plt.legend()
    plt.show()
    st.pyplot(fig6)

    # Plotting next 90 days price prediction
    st.subheader("Last 15 Days and Next 90 Days Prediction")
    fig7 = plt.figure(figsize=(8, 6))
    sns.lineplot(x=np.arange(1, 16), y=data_test_scale[-15:].flatten(), label='Last 15 Days')
    sns.lineplot(x=np.arange(16, 106), y=predicted_prices.flatten(), label='Next 90 Days Prediction')
    plt.xlabel('Day')
    plt.ylabel('Normalized Close Price')
    plt.legend()
    plt.show()
    st.pyplot(fig7)
