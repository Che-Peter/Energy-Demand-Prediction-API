import pandas as pd
import joblib
import csv
from fastapi import FastAPI
from pydantic import BaseModel
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
X = df_encoded.drop(columns=["demand_kWh", "datetime"])
y = df_encoded["demand_kWh"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# ----------------------------
# 2. FastAPI backend
# ----------------------------
app = FastAPI()

# Request schema for prediction
class PredictionRequest(BaseModel):
    timestamp: int
    voltage: float
    current: float
    active_power: float
    energy_consumption: float
    power_factor: float
    temperature: float
    humidity: float
    light_intensity: int
    building_type: str
    occupancy_level: int
    operational_schedule: str
    electricity_tariff: str
    appliance_category: str

# Request schema for dataset update (respecting CSV column order)
class UpdateRequest(BaseModel):
    timestamp: int
    voltage: float
    current: float
    active_power: float
    energy_consumption: float
    power_factor: float
    temperature: float
    humidity: float
    light_intensity: int
    building_type: str
    occupancy_level: int
    operational_schedule: str
    electricity_tariff: str
    appliance_category: str
    demand_kWh: float

@app.post("/predict")
def predict(req: PredictionRequest):
    model = joblib.load("model.pkl")
    input_dict = req.dict()

    # Temporal features
    ts = pd.to_datetime(input_dict["timestamp"], unit="s")
    hour = ts.hour
    dayofweek = ts.dayofweek
    is_weekend = 1 if dayofweek in [5, 6] else 0
    tod = time_of_day(hour)

    input_dict["time_of_day"] = tod
    input_dict["is_weekend"] = is_weekend
    del input_dict["timestamp"]

    input_df = pd.DataFrame([input_dict])
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

@app.post("/update")
def update_table(req: UpdateRequest):
    new_row = [
        req.timestamp,
        req.voltage,
        req.current,
        req.active_power,
        req.energy_consumption,
        req.power_factor,
        req.temperature,
        req.humidity,
        req.light_intensity,
        req.building_type,
        req.occupancy_level,
        req.operational_schedule,
        req.electricity_tariff,
        req.appliance_category,
        req.demand_kWh
    ]

    with open("synthetic_dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_row)

    return {"status": "Row appended to dataset"}
