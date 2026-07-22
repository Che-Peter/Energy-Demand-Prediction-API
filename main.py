import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load and preprocess data
# ----------------------------
df = pd.read_csv("synthetic_dataset.csv")

# Derive temporal features from timestamp
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

# Encode categorical features
categorical_cols = ["building_type", "operational_schedule",
                    "electricity_tariff", "appliance_category", "time_of_day"]

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Features and target
X = df_encoded.drop(columns=["demand_kWh", "datetime"])
y = df_encoded["demand_kWh"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")

# ----------------------------
# 3. FastAPI backend
# ----------------------------
app = FastAPI()

# Request schema
class PredictionRequest(BaseModel):
    timestamp: int
    voltage: float = 230.0
    current: float = 10.0
    active_power: float = 2000.0
    energy_consumption: float = 5.0
    power_factor: float = 0.95
    temperature: float = 28.0
    humidity: float = 70.0
    light_intensity: int = 500
    building_type: str = "Residential"
    occupancy_level: int = 15
    operational_schedule: str = "Daytime"
    electricity_tariff: str = "Peak"
    appliance_category: str = "HVAC"

@app.post("/predict")
def predict(req: PredictionRequest):
    model = joblib.load("model.pkl")

    input_dict = req.dict()

    # Derive temporal features from timestamp
    ts = pd.to_datetime(input_dict["timestamp"], unit="s")
    hour = ts.hour
    dayofweek = ts.dayofweek
    is_weekend = 1 if dayofweek in [5, 6] else 0

    if 0 <= hour < 6:
        tod = "Night"
    elif 6 <= hour < 12:
        tod = "Morning"
    elif 12 <= hour < 18:
        tod = "Day"
    else:
        tod = "Evening"

    input_dict["time_of_day"] = tod
    input_dict["is_weekend"] = is_weekend

    # Drop raw timestamp
    del input_dict["timestamp"]

    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    return {"predicted_demand_kWh": round(float(prediction), 2)}

@app.post("/train")
def retrain():
    df = pd.read_csv("synthetic_dataset.csv")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["time_of_day"] = df["hour"].apply(time_of_day)

    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    X = df_encoded.drop(columns=["demand_kWh", "datetime"])
    y = df_encoded["demand_kWh"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return {"status": "Model retrained and saved"}
