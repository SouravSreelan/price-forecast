import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib 

def train_and_save_model(csv_path, model_path="xgb_price_model.pkl"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['avg_monthly_price'] = df['avg_monthly_price'].interpolate()
    Q1 = df['avg_monthly_price'].quantile(0.25)
    Q3 = df['avg_monthly_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df['avg_monthly_price'] = df['avg_monthly_price'].clip(lower, upper)
    df['lag_1'] = df['avg_monthly_price'].shift(1)
    df['lag_2'] = df['avg_monthly_price'].shift(2)
    df['lag_3'] = df['avg_monthly_price'].shift(3)
    df['month'] = df['date'].dt.month
    df.dropna(inplace=True)

    X = df[['lag_1','lag_2','lag_3','month']]
    y = df['avg_monthly_price']
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, model_path)
    return model

def forecast_next_12_months(csv_path, model_path="xgb_price_model.pkl"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['avg_monthly_price'] = df['avg_monthly_price'].interpolate()
    Q1 = df['avg_monthly_price'].quantile(0.25)
    Q3 = df['avg_monthly_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df['avg_monthly_price'] = df['avg_monthly_price'].clip(lower, upper)
    model = joblib.load(model_path)
    last_row = df.iloc[-3:].copy()
    future_preds = []

    for i in range(12):
        month = (last_row['date'].iloc[-1].month % 12) + 1
        X_new = np.array([[last_row['avg_monthly_price'].iloc[-1],
                           last_row['avg_monthly_price'].iloc[-2],
                           last_row['avg_monthly_price'].iloc[-3],
                           month]])
        y_pred = model.predict(X_new)[0]
        future_preds.append(y_pred)

        new_row = pd.DataFrame({
            'date': [last_row['date'].iloc[-1] + pd.DateOffset(months=1)],
            'avg_monthly_price': [y_pred]
        })
        last_row = pd.concat([last_row, new_row], ignore_index=True)

    future_dates = pd.date_range(start=df['date'].iloc[-1] + pd.DateOffset(months=1),
                                 periods=12, freq='MS')
    future_forecast = pd.Series(future_preds, index=future_dates)
    return future_forecast
