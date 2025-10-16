# Price Prediction Model Report

---

## 1. Price Prediction Model Overview

**Model Used:** **XGBoost Regressor**  

**Objective:** Forecast monthly prices based on historical trends and seasonal patterns.  

**Input Features:**  
- **Lagged prices:** Prices from the previous 1, 2, and 3 months  
- **Month of the year:** Captures seasonal fluctuations  

**Why XGBoost:**  
- High accuracy for non-linear patterns in time-series data  
- Handles missing data and outliers effectively  
- Fast and scalable for large datasets  

**How it works:**  

Historical Prices → Feature Extraction → XGBoost Model → Predicted Price

- The model “learns” patterns from past prices.  
- Uses previous months' prices and seasonality to predict future prices.  
- Produces accurate forecasts even with irregular fluctuations.

**Example:**  

| Month    | Actual Price | Predicted Price |
|---------|--------------|----------------|
| Jan-2025 | 11313       | 11400          |
| Feb-2025 | 14188       | 14250          |
| Mar-2025 | 14263       | 14400          |

---

## 2. Suggested Actions Based on Predicted Price Changes

**Price Increase Predicted:**  
- Buy raw materials early to avoid higher costs  
- Adjust product prices to maintain profit  
- Increase production or stock to meet demand  

**Price Decrease Predicted:**  
- Delay purchases to benefit from lower costs  
- Offer discounts or promotions to maintain sales  
- Reduce inventory to avoid holding excess stock  

**Visual Flow:**

Price Prediction → Identify Trend (Up/Down) → Recommended Action

**Example:**  
- Predicted prices rise in May → Buy raw materials in April → Maintain profit  
- Predicted prices fall in June → Delay bulk purchases → Save cost  

---

## 3. Measuring Effectiveness of Actions

**Methods:**  

1. **Compare predicted vs actual prices**  
   - Did the company avoid paying higher prices?  
   - Did they capitalize on price drops?  

2. **Track key business metrics:**  
   - Profit margins  
   - Inventory levels  
   - Sales revenue  

3. **Post-implementation analysis:**  
   - Analyze if actions taken (early procurement, promotions) achieved expected benefits  
   - Adjust future actions based on results  

**Visual Example:**  

Action Implemented → Measure Outcomes → Compare to Forecast → Feedback Loop

---

## 4. Deploying the Model Using Django

**Steps:**  

1. Train the XGBoost model in Python → Save as `xgb_price_model.pkl`  
2. Load model in Django backend  
3. Schedule automatic updates for new forecasts (e.g., monthly using Celery)  
4. Store forecasts in the database → Can be retrieved anytime  

**Visual Flow:**

Train Model → Save → Django Backend → Schedule Forecast → Store in DB → Use in Reports

**Outcome:**  
- Decision-makers always have up-to-date forecasts  
- Model runs automatically without manual intervention  

---

## 5. Integrating the Model into a Django Web Application

**Real-Time Predictions:**  

1. **User Interface:**  
   - Users select a date or provide latest price data  
   - Simple web form or dashboard  

2. **Backend:**  
   - Django loads the trained model  
   - Computes predicted prices instantly  

3. **Visualization:**  
   - Charts show historical vs predicted vs forecasted prices  
   - Tools like `matplotlib`, `Plotly`, or `Seaborn` can be used  

4. **API Endpoint (Optional):**  
   - `/api/forecast/` provides predictions for other systems  
   - Supports automated reporting or integration with ERP systems  

**Visual Flow:**

User Input → Django Backend → Load Model → Predict → Visualize/Return API Response

---

## 6. Monitoring the Model in Production

**Why Monitoring is Important:**  
- Market trends change → model predictions may drift  
- Early detection ensures reliable forecasts

**Monitoring Steps:**  

1. **Track Performance Metrics:**  
   - MAE (Mean Absolute Error)  
   - RMSE (Root Mean Squared Error)  
   - R² score  

2. **Detect Model Drift:**  
   - Compare actual prices to predicted prices over time  
   - Significant deviations may indicate drift  

3. **Update Model:**  
   - Retrain with latest data  
   - Replace the old model in Django backend  
   - Notify stakeholders about updated forecasts  

**Visual Flow:**

Forecast → Compare with Actual → Check Metrics → Model Drift? → Retrain/Update → Repeat

---

## 7. Summary

- XGBoost provides accurate, automated monthly price forecasts  
- Forecasts enable proactive decision-making to optimize profits  
- Integration with Django allows real-time predictions and dashboards  
- Continuous monitoring ensures model reliability and adaptability  

