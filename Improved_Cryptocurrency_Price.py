import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from art import text2art
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tkinter import Tk, Label, Button, Text, Scrollbar, END, RIGHT, Y, LEFT, TOP, BOTTOM, VERTICAL, Frame, StringVar, Radiobutton, Toplevel, Canvas, NW
from PIL import Image, ImageTk

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal,
        'Histogram': histogram
    })

def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def create_features(df):
    df['SMA7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['SMA30'] = df['Close'].rolling(window=30, min_periods=1).mean()
    df['EMA7'] = df['Close'].ewm(span=7, adjust=False, min_periods=1).mean()
    df['EMA30'] = df['Close'].ewm(span=30, adjust=False, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
    
    macd_data = calculate_macd(df['Close'])
    df['MACD'] = macd_data['MACD']
    df['MACD_Signal'] = macd_data['Signal']
    df['MACD_Histogram'] = macd_data['Histogram']
    
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
    df['Price_Momentum'] = df['Close'].pct_change(periods=14)
    
    return df

def format_price(price):
    return f"${price:,.2f}" if price < 1000 else f"${price/1000:,.2f}K"

def backtest_model(model, X, y, test_size=0.2):
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    percentage_error = np.abs((y_test - predictions) / y_test) * 100
    mean_percentage_error = np.mean(percentage_error)
    accuracy = 100 - mean_percentage_error

    return accuracy, predictions, y_test

def process_and_predict(ticker, text_widget, image_label, forecast_label):
    crypto_data = yf.download(ticker, start='2010-07-17')
    text_widget.insert(END, f"Downloaded {ticker.split('-')[0]}.\n")
    
    last_date = crypto_data.index[-1].date()
    current_price = crypto_data['Close'].iloc[-1]
    text_widget.insert(END, f"Last available {ticker.split('-')[0]} price ({last_date.strftime('%d/%m/%Y')}): {format_price(current_price)}\n")
    
    crypto_data = create_features(crypto_data)
    crypto_data.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA30', 'EMA7', 'EMA30', 'RSI', 'Volume_Change', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Bollinger_Upper', 'Bollinger_Lower', 'Price_Momentum']
    target = 'Close'
    
    X = crypto_data[features]
    y = crypto_data[target].shift(-1)
    
    X = X[:-1]
    y = y[:-1]
    
    # Eliminar valores infinitos o demasiado grandes
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    mse_scores = []
    r2_scores = []
    
    text_widget.insert(END, "Training and evaluating model...\n")
    
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
    
    text_widget.insert(END, f"Average MSE: {np.mean(mse_scores)}\n")
    text_widget.insert(END, f"Average R2 Score: {np.mean(r2_scores)}\n")
    
    text_widget.insert(END, "Training final model...\n")
    model.fit(X_scaled, y)
    
    last_data = X_scaled.iloc[-1].to_frame().T
    prediction = model.predict(last_data)
    
    next_day = last_date + datetime.timedelta(days=1)
    predicted_price = prediction[0]
    
    # Set the forecast label
    if predicted_price > current_price:
        forecast_label.config(text=f"Predicted price for {next_day.strftime('%d/%m/%Y')}: {format_price(predicted_price)}", fg='green')
    else:
        forecast_label.config(text=f"Predicted price for {next_day.strftime('%d/%m/%Y')}: {format_price(predicted_price)}", fg='red')
    
    text_widget.insert(END, "Realizando backtesting...\n")
    accuracy, predictions, y_test = backtest_model(model, X_scaled, y)
    text_widget.insert(END, f"Precisión del modelo en datos históricos: {accuracy:.2f}%\n")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test.values, label='Precio real', color='blue')
    ax.plot(y_test.index, predictions, label='Predicción', color='red', alpha=0.7)
    ax.set_title(f'Backtesting: Predicciones vs Precios reales para {ticker.split("-")[0]}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio (USD)')
    
    # Formatear el eje y en miles de dólares
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1000:.2f}K'))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Configurar el formato de fecha en el eje x
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.autofmt_xdate()
    
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('backtesting_plot.png')
    plt.close()
    
    # Mostrar el gráfico en la interfaz gráfica
    img = Image.open('backtesting_plot.png')
    img = img.resize((600, 400), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    
    text_widget.insert(END, f"Predicciones y visualización completadas para {ticker.split('-')[0]}.\n")

def zoom(event):
    canvas = event.widget
    scale = 1.0
    if event.delta > 0:
        scale *= 1.1
    else:
        scale /= 1.1
    canvas.scale("all", event.x, event.y, scale, scale)
    canvas.config(scrollregion=canvas.bbox('all'))

def open_full_image(image_path):
    full_image_window = Toplevel()
    full_image_window.title("Full Image View")
    
    img = Image.open(image_path)
    img_tk = ImageTk.PhotoImage(img)
    
    canvas = Canvas(full_image_window, width=img.width, height=img.height)
    canvas.pack(side=LEFT, fill='both', expand=True)
    canvas_img = canvas.create_image(0, 0, anchor=NW, image=img_tk)
    canvas.config(scrollregion=canvas.bbox('all'))
    canvas.image = img_tk
    
    # Adding scrollbars
    x_scrollbar = Scrollbar(full_image_window, orient='horizontal', command=canvas.xview)
    x_scrollbar.pack(side=BOTTOM, fill='x')
    y_scrollbar = Scrollbar(full_image_window, orient='vertical', command=canvas.yview)
    y_scrollbar.pack(side=RIGHT, fill='y')
    canvas.config(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
    
    # Bind the mouse wheel event for zooming
    canvas.bind('<MouseWheel>', zoom)

def run_analysis():
    ticker = ticker_var.get()
    text_widget.delete(1.0, END)
    process_and_predict(ticker, text_widget, image_label, forecast_label)

def main():
    global ticker_var, text_widget, image_label, forecast_label
    
    root = Tk()
    root.title("Cripto Predict AI")
    
    ascii_art = text2art("NaMasDev")
    
    # Create a frame for the top section
    top_frame = Frame(root)
    top_frame.pack(side=TOP, fill='x')
    
    Label(top_frame, text="============================================================").pack()
    Label(top_frame, text=ascii_art).pack()
    Label(top_frame, text="Improved Cryptocurrency Price Prediction AI with Backtesting").pack()
    Label(top_frame, text="============================================================").pack()
    Label(top_frame, text="Created by: NaMasDev").pack()
    Label(top_frame, text="Github: https://github.com/djskom/").pack()
    Label(top_frame, text="Licence : MIT License").pack()
    Label(top_frame, text="Support my work with a like").pack()
    Label(top_frame, text="============================================================").pack()
    
    forecast_frame = Frame(root)
    forecast_frame.pack(side=TOP, fill='x', pady=5)
    
    forecast_label = Label(forecast_frame, text="", font=("Helvetica", 16))
    forecast_label.pack()
    
    main_frame = Frame(root)
    main_frame.pack(side=TOP, fill='x')
    
    Label(main_frame, text="Select a cryptocurrency to predict the price for:").pack()
    
    cryptocurrencies = {
        "1": "BTC-USD",
        "2": "ETH-USD",
        "3": "BNB-USD",
        "4": "SOL-USD",
        "5": "XRP-USD",
        "6": "DOGE-USD",
        "7": "AVAX-USD",
        "8": "ADA-USD",
        "9": "SHIB-USD",
        "10": "DOT-USD",
        "11": "ALL"
    }
    
    ticker_var = StringVar(value="BTC-USD")
    
    def select_crypto(value):
        ticker_var.set(value)
        for key, rb in radiobuttons.items():
            rb.config(bg='SystemButtonFace')  # Reset background color
        radiobuttons[value].config(bg='lightblue')  # Highlight selected button
    
    radiobuttons = {}
    for key, value in cryptocurrencies.items():
        rb = Radiobutton(main_frame, text=value.split('-')[0], variable=ticker_var, value=value, command=lambda v=value: select_crypto(v))
        rb.pack(side=LEFT)
        radiobuttons[value] = rb
    
    Button(main_frame, text="Run Analysis", command=run_analysis).pack(side=BOTTOM, pady=10)
    
    # Create a frame for the text output
    text_frame = Frame(root)
    text_frame.pack(side=LEFT, fill='both', expand=True)
    
    scroll_y = Scrollbar(text_frame, orient=VERTICAL)
    scroll_y.pack(side=RIGHT, fill=Y)
    
    # Set font size for the Text widget
    text_widget = Text(text_frame, wrap='word', yscrollcommand=scroll_y.set, font=("Helvetica", 14))
    text_widget.pack(side=LEFT, fill='both', expand=True)
    scroll_y.config(command=text_widget.yview)
    
    # Create a frame for the image output
    image_frame = Frame(root)
    image_frame.pack(side=RIGHT, fill='both', expand=True)
    
    image_label = Label(image_frame)
    image_label.pack()
    image_label.bind("<Button-1>", lambda e: open_full_image('backtesting_plot.png'))  # Bind click event

    root.mainloop()

if __name__ == "__main__":
    main()  
    
