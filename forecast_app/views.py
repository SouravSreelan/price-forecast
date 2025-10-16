from django.conf import settings
import matplotlib
matplotlib.use('Agg')  
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from django.shortcuts import render
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

def forecast_plot_view(request):
    csv_path = os.path.join(settings.BASE_DIR, "forecast_app", "data", "price_data.csv")
    model_path = os.path.join(settings.BASE_DIR, "xgb_price_model.pkl")
    
    df = pd.read_csv(csv_path)
    df.rename(columns={"date": "Date", "avg_monthly_price": "Price"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    df["lag_1"] = df["Price"].shift(1)
    df["lag_2"] = df["Price"].shift(2)
    df["lag_3"] = df["Price"].shift(3)
    df["month"] = df.index.month
    df = df.dropna()
    X = df[["lag_1", "lag_2", "lag_3", "month"]]
    y = df["Price"]

    model = joblib.load(model_path)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    last_row = df.iloc[-1]
    last_date = df.index[-1]
    future_preds = []
    current_lags = [last_row["lag_1"], last_row["lag_2"], last_row["lag_3"]]
    for i in range(12):
        month_feature = (last_date.month + i) % 12 or 12
        features = np.array([[current_lags[0], current_lags[1], current_lags[2], month_feature]])
        yhat = model.predict(features)[0]
        next_date = last_date + pd.DateOffset(months=i+1)
        future_preds.append((next_date, yhat))
        current_lags = [yhat, current_lags[0], current_lags[1]]
    forecast_df = pd.DataFrame(future_preds, columns=["Date", "Forecast"])
    forecast_df.set_index("Date", inplace=True)
    forecast_df.index = forecast_df.index.strftime("%d-%m-%Y")
    forecast_table = forecast_df.to_html(classes="table table-striped", float_format="%.2f")
    df.index = pd.to_datetime(df.index)
    forecast_df.index = pd.to_datetime(forecast_df.index)

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Price"], label="Actual")
    plt.plot(df.index, y_pred, label="Predicted")
    plt.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast", linestyle="--")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("Price Forecast")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    return render(request, "forecast_app/forecast_plot.html", {
        "plot_image": image_base64,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 2), 
        "forecast_table": forecast_table  
    })
