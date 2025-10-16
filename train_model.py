import os
from forecast_app.utils.forecasting import train_and_save_model

csv_path = r"C:\Users\admin\Downloads\price_data.csv"
model_dir = r"D:\My Trials\task\price_forecast"  
model_path = os.path.join(model_dir, "xgb_price_model.pkl")

os.makedirs(model_dir, exist_ok=True)

train_and_save_model(csv_path, model_path)
print("Model trained and saved successfully!")
