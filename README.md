# Price Forecasting Web Application

This Django project forecasts the next 12 months of average monthly prices using an XGBoost model.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/SouravSreelan/price-forecast.git
cd price_forecast
```
### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```
### 3. Train the model
Before running the server, train the model using your historical data:
```bash
python train_model.py
```
### 4. Run the Django server
```bash
python manage.py runserver
```
### 5. Access the forecast
```bash
http://127.0.0.1:8000/api/forecast_plot/
```




