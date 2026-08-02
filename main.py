import pandas as pd
import joblib
import csv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Initial training on dataset
# ----------------------------
df = pd.read_csv("synthetic_dataset.csv")

# Temporal features
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

def time_of_day(hour):
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Day"
    else:
        return "Evening"

df["time_of_day"] = df["hour"].apply(time_of_day)

# Categorical features
categorical_cols = ["building_type", "operational_schedule",
                    "electricity_tariff", "appliance_category", "time_of_day"]

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Features and target
X = df_encoded.drop(columns=["demand_kWh", "datetime", "energy_consumption"])
y = df_encoded["demand_kWh"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model_xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)
model_xgb.fit(X_train, y_train)
joblib.dump(model_xgb, "model_xgb.pkl")

# Train RandomForest
model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
model_rf.fit(X_train, y_train)
joblib.dump(model_rf, "model_rf.pkl")

# ----------------------------
# 2. FastAPI backend
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class PredictionRequest(BaseModel):
    timestamp: int

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # Load both models
        model_xgb = joblib.load("model_xgb.pkl")
        model_rf = joblib.load("model_rf.pkl")

        # Build features from timestamp only
        ts = pd.to_datetime(req.timestamp, unit="s")
        hour = ts.hour
        dayofweek = ts.dayofweek
        is_weekend = 1 if dayofweek in [5, 6] else 0
        tod = time_of_day(hour)

        input_dict = {
            "voltage": 0.0,
            "current": 0.0,
            "active_power": 0.0,
            "energy_consumption": 0.0,
            "power_factor": 0.0,
            "temperature": 0.0,
            "humidity": 0.0,
            "light_intensity": 0,
            "building_type": "default",
            "occupancy_level": 0,
            "operational_schedule": "default",
            "electricity_tariff": "default",
            "appliance_category": "default",
            "time_of_day": tod,
            "is_weekend": is_weekend
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        pred_xgb = model_xgb.predict(input_encoded)[0]
        pred_rf = model_rf.predict(input_encoded)[0]

        return {
            "predicted_demand_kWh_xgb": round(float(pred_xgb), 2),
            "predicted_demand_kWh_rf": round(float(pred_rf), 2)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
