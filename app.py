import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

stock_files = {
    'Reliance': 'Reliance.csv',
    'HDFC': 'HDFC.csv',
    'Infosys': 'Infosys.csv',
    'SBI': 'SBI.csv',
    'M&M': 'M&M.csv',
    'TataMotors': 'Tatamotors.csv',
    'Wipro': 'Wipro.csv',
    'Titan': 'Titan.csv'
}

with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

def load_data(stock_name):
    file_path = stock_files.get(stock_name)
    if file_path:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        return df
    raise ValueError(f"Stock data for {stock_name} not found.")

def calculate_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df = df.bfill()
    return df

def generate_price_indicators_plot(df, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.plot(df.index, df['SMA20'], label='SMA20', color='red')
    plt.plot(df.index, df['SMA50'], label='SMA50', color='green')
    plt.title(f'{stock_name} Price with Indicators')
    plt.legend()
    plot_path = f'static/{stock_name}_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def generate_candlestick_chart(df, stock_name):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='green')),
        decreasing=dict(line=dict(color='red'))
    )])
    fig.update_layout(
        title=f'{stock_name} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        width=1000,
        height=600
    )
    plot_path = f'static/{stock_name}_candlestick.html'
    fig.write_html(plot_path)
    return plot_path

def generate_forecast(model, steps=7):
    forecast = model.forecast(steps=steps)
    return forecast

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        try:
            df = load_data(stock_name)
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

            if df_filtered.empty:
                return render_template('index.html', error_message="No data found. Dates must exclude holidays.")

            df_filtered = calculate_indicators(df_filtered)

            # Generate 7-day forecast using ARIMA model
            forecast = generate_forecast(arima_model, steps=7)
            predicted_prices = [round(price, 2) for price in forecast]

            # Generate plots
            price_indicators_plot = generate_price_indicators_plot(df_filtered, stock_name)
            candlestick_chart = generate_candlestick_chart(df_filtered, stock_name)

            return render_template('index.html',
                                   stock=stock_name,
                                   start_date=start_date,
                                   end_date=end_date,
                                   predicted_prices=predicted_prices,
                                   price_plot=price_indicators_plot,
                                   candlestick_chart=candlestick_chart)

        except Exception as e:
            return render_template('index.html', error_message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)