"""
Energy demand forecasting API.

ThingSpeak is the source of truth. Nothing is pulled automatically: when the app
displays data it calls POST /sync, which fetches only the readings newer than the
last stored one (everything when the table is empty), retrains on the complete data
set and regenerates the next 24h of forecasts.

For training, readings are cleaned and averaged into 15-minute buckets. A forecast for
time t, made at "origin" (the latest data), is:

    baseline = hour-of-day profile(t) + w(horizon) * anomaly(origin)
    model    = baseline + shrink * residual_model(features)

- profile: mean demand for each hour of the day over the recent past
- anomaly: how far the last hour was above/below its normal profile; w() is fitted
  from data and fades to 0 for long horizons
- residual models (XGBoost and RandomForest) learn what the baseline misses
  (weekday effects, level shifts...). Each retrain validates them on the most recent
  unseen data and `shrink` scales their correction down when it does not help.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------
# Storage
# ----------------------------
# On Railway, mount a volume and set DATA_DIR to it, otherwise the data,
# models and forecasts are wiped on every redeploy.
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "dataset.csv")
MODEL_FILE = os.path.join(DATA_DIR, "models.pkl")
FORECAST_FILE = os.path.join(DATA_DIR, "forecasts.csv")

COLUMNS = [
    "timestamp", "voltage", "current", "active_power", "energy_consumption", "power_factor",
    "temperature", "humidity", "light_intensity",
    "building_type", "occupancy_level", "operational_schedule",
    "electricity_tariff", "appliance_category", "demand_kWh"
]
FORECAST_COLUMNS = ["timestamp", "pred_xgb", "pred_rf", "pred_baseline", "made_at"]

# ----------------------------
# ThingSpeak
# ----------------------------
THINGSPEAK_CHANNEL = os.environ.get("THINGSPEAK_CHANNEL", "3412539")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY", "X4N1SN9XUD1IAQM9")
THINGSPEAK_PAGE = 8000   # ThingSpeak's maximum results per request

# ----------------------------
# Model settings
# ----------------------------
BUCKET = pd.Timedelta(minutes=15)
HORIZONS_MIN = [15, 60, 240, 720, 1440]   # horizons the models are trained on
FORECAST_HOURS = 24                       # automatic forecast window after each update
MIN_VOLTAGE = 100.0                       # readings below this are outages / sensor errors
MIN_TRAIN_BUCKETS = 96                    # one day of 15-minute buckets
MIN_VALIDATION_BUCKETS = 96               # below this, residual models are not trusted
PROFILE_DAYS = 28                         # profile uses the most recent weeks only
MAX_TRAIN_DAYS = 180                      # cap training cost as data grows

FEATURES = ["hsin", "hcos", "dsin", "dcos", "is_weekend", "profile",
            "anomaly", "level_ratio", "horizon"]

# Serialises every read/write of the data, model and forecast files
lock = threading.Lock()
# Only one training run at a time; extra requests are skipped, the running one is fresh enough
train_lock = threading.Lock()
# Only one ThingSpeak sync at a time
sync_lock = threading.Lock()
last_sync = {"at": None, "added": 0, "error": None}


# ----------------------------
# Data helpers
# ----------------------------
def load_csv(path, columns) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def upsert(path, columns, new_rows: pd.DataFrame):
    """Insert rows, replacing any existing row with the same timestamp."""
    df = load_csv(path, columns)
    df = pd.concat([df, new_rows[columns]], ignore_index=True) if len(df) else new_rows[columns]
    df = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    df.to_csv(path, index=False)


def demand_series(df: pd.DataFrame) -> pd.Series:
    """Clean readings and average them into 15-minute buckets (UTC index)."""
    if df.empty:
        return pd.Series(dtype=float)
    df = df[pd.to_numeric(df["voltage"], errors="coerce") >= MIN_VOLTAGE]
    s = pd.Series(
        pd.to_numeric(df["demand_kWh"], errors="coerce").values,
        index=pd.to_datetime(df["timestamp"], unit="s")
    ).dropna().sort_index()
    if s.empty:
        return s
    return s.resample(BUCKET).mean().dropna()


def asof(series: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
    """Value of `series` at or before each time in `when` (NaN if none)."""
    if series.empty:
        return np.full(len(when), np.nan)
    pos = series.index.searchsorted(when, side="right") - 1
    out = np.where(pos >= 0, series.values[np.clip(pos, 0, None)], np.nan)
    return out.astype(float)


def hourly_profile(series: pd.Series) -> pd.Series:
    recent = series[series.index >= series.index.max() - pd.Timedelta(days=PROFILE_DAYS)]
    return recent.groupby(recent.index.hour).mean()


def build_features(targets, history: pd.Series, profile: pd.Series, mean_level: float,
                   origins) -> pd.DataFrame:
    """
    Features for forecasting `targets` from what was known at `origins`.
    Only data in `history` up to each origin is used, so nothing leaks from the future.
    """
    targets = pd.DatetimeIndex(targets)
    origins = pd.DatetimeIndex(origins)
    f = pd.DataFrame(index=targets)

    hour_frac = targets.hour + targets.minute / 60
    f["hsin"] = np.sin(2 * np.pi * hour_frac / 24)
    f["hcos"] = np.cos(2 * np.pi * hour_frac / 24)
    f["dsin"] = np.sin(2 * np.pi * targets.dayofweek / 7)
    f["dcos"] = np.cos(2 * np.pi * targets.dayofweek / 7)
    f["is_weekend"] = (targets.dayofweek >= 5).astype(int)

    f["profile"] = profile.reindex(targets.hour).fillna(mean_level).values

    recent_1h = asof(history.rolling("1h").mean(), origins)
    profile_at_origin = profile.reindex(origins.hour).fillna(mean_level).values
    f["anomaly"] = np.nan_to_num(recent_1h - profile_at_origin)

    level_24h = asof(history.rolling("24h").mean(), origins)
    f["level_ratio"] = np.nan_to_num(level_24h / mean_level, nan=1.0) if mean_level else 1.0

    f["horizon"] = ((targets - origins) / pd.Timedelta(minutes=1)).values
    return f


def anomaly_weight(weights: dict, horizon_min) -> np.ndarray:
    keys = sorted(weights)
    return np.interp(horizon_min, keys, [weights[k] for k in keys], right=0.0)


def training_rows(series: pd.Series, profile: pd.Series, mean_level: float):
    """Every bucket is a target once per training horizon."""
    frames, ys = [], []
    for h in HORIZONS_MIN:
        origins = series.index - pd.Timedelta(minutes=h)
        frames.append(build_features(series.index, series, profile, mean_level, origins))
        ys.append(series.values)
    return pd.concat(frames), np.concatenate(ys)


def fit_anomaly_weights(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Least-squares weight of the current anomaly for each horizon, clipped to [0, 1]."""
    weights = {}
    for h in HORIZONS_MIN:
        m = (X["horizon"] == h).values
        a = X["anomaly"].values[m]
        r = y[m] - X["profile"].values[m]
        denom = float((a * a).sum())
        weights[h] = float(np.clip((a * r).sum() / denom, 0, 1)) if denom > 0 else 0.0
    return weights


def fit_models(series: pd.Series):
    profile = hourly_profile(series)
    mean_level = float(series.mean())
    X, y = training_rows(series, profile, mean_level)
    weights = fit_anomaly_weights(X, y)
    baseline = X["profile"].values + X["anomaly"].values * anomaly_weight(weights, X["horizon"].values)

    xgb = XGBRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=10,
        random_state=42, objective="reg:squarederror"
    )
    xgb.fit(X[FEATURES], y - baseline)

    rf = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=30, max_features=0.6, random_state=42, n_jobs=-1
    )
    rf.fit(X[FEATURES], y - baseline)

    return {"xgb": xgb, "rf": rf, "profile": profile, "mean_level": mean_level, "weights": weights}


def predict_parts(bundle, history: pd.Series, targets, origin):
    """Returns (baseline, xgb, rf) arrays for the targets."""
    targets = pd.DatetimeIndex(targets)
    origins = pd.DatetimeIndex([origin] * len(targets))
    X = build_features(targets, history, bundle["profile"], bundle["mean_level"], origins)
    baseline = X["profile"].values + X["anomaly"].values * anomaly_weight(bundle["weights"], X["horizon"].values)
    xgb = baseline + bundle["shrink_xgb"] * bundle["xgb"].predict(X[FEATURES])
    rf = baseline + bundle["shrink_rf"] * bundle["rf"].predict(X[FEATURES])
    # Demand can't be negative
    return np.maximum(baseline, 0), np.maximum(xgb, 0), np.maximum(rf, 0)


def mae(a, b):
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 2)


def best_shrink(residual_pred, residual_true) -> float:
    """How much of a residual model's correction to keep, fitted on unseen data."""
    denom = float((residual_pred * residual_pred).sum())
    if denom <= 0:
        return 0.0
    return float(np.clip((residual_pred * residual_true).sum() / denom, 0, 1))


def validate(series: pd.Series):
    """
    Train on the older 80% of buckets, test on the newest 20% (as real forecasts from
    each horizon). Returns error metrics and how much to trust each residual model.
    """
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    if len(test) < MIN_VALIDATION_BUCKETS or len(train) < MIN_TRAIN_BUCKETS:
        return None, 0.0, 0.0

    fitted = fit_models(train)
    profile, mean_level = fitted["profile"], fitted["mean_level"]

    preds = {"profile": [], "baseline": [], "xgb_raw": [], "rf_raw": []}
    truth = []
    for h in HORIZONS_MIN:
        origins = test.index - pd.Timedelta(minutes=h)
        X = build_features(test.index, series, profile, mean_level, origins)
        base = X["profile"].values + X["anomaly"].values * anomaly_weight(fitted["weights"], X["horizon"].values)
        preds["profile"].append(X["profile"].values)
        preds["baseline"].append(base)
        preds["xgb_raw"].append(fitted["xgb"].predict(X[FEATURES]))
        preds["rf_raw"].append(fitted["rf"].predict(X[FEATURES]))
        truth.append(test.values)

    truth = np.concatenate(truth)
    base = np.concatenate(preds["baseline"])
    xgb_raw = np.concatenate(preds["xgb_raw"])
    rf_raw = np.concatenate(preds["rf_raw"])

    shrink_xgb = best_shrink(xgb_raw, truth - base)
    shrink_rf = best_shrink(rf_raw, truth - base)

    metrics = {
        "validation_buckets": int(len(test)),
        "mean_demand": round(float(test.mean()), 2),
        "mae_profile_only": mae(np.concatenate(preds["profile"]), truth),
        "mae_baseline": mae(base, truth),
        "mae_xgb": mae(base + shrink_xgb * xgb_raw, truth),
        "mae_rf": mae(base + shrink_rf * rf_raw, truth),
        "shrink_xgb": round(shrink_xgb, 2),
        "shrink_rf": round(shrink_rf, 2),
    }
    return metrics, shrink_xgb, shrink_rf


def generate_forecasts(bundle, history: pd.Series, hours=FORECAST_HOURS) -> pd.DataFrame:
    """Forecast every 15-minute slot for the next `hours`, from the latest data."""
    origin = history.index.max()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = max(origin, now).floor(BUCKET) + BUCKET
    targets = pd.date_range(start, start + pd.Timedelta(hours=hours) - BUCKET, freq=BUCKET)
    base, xgb, rf = predict_parts(bundle, history, targets, origin)
    return pd.DataFrame({
        # Works whatever the datetime resolution (pandas 3 is not always nanoseconds)
        "timestamp": ((targets - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).astype(int),
        "pred_xgb": np.round(xgb, 2),
        "pred_rf": np.round(rf, 2),
        "pred_baseline": np.round(base, 2),
        "made_at": int(time.time()),
    })


def refresh_forecasts() -> int:
    """Regenerate upcoming forecasts with the current models and latest data."""
    with lock:
        if not os.path.exists(MODEL_FILE):
            return 0
        bundle = joblib.load(MODEL_FILE)
        history = demand_series(load_csv(DATA_FILE, COLUMNS))
    if history.empty:
        return 0
    fc = generate_forecasts(bundle, history)
    with lock:
        upsert(FORECAST_FILE, FORECAST_COLUMNS, fc)
    return len(fc)


class TrainingSkipped(Exception):
    pass


def train_models() -> dict:
    """Retrain on the complete data set, then regenerate the upcoming forecasts."""
    if not train_lock.acquire(blocking=False):
        raise TrainingSkipped("Training already in progress")
    try:
        with lock:
            series = demand_series(load_csv(DATA_FILE, COLUMNS))
        if not series.empty:
            series = series[series.index >= series.index.max() - pd.Timedelta(days=MAX_TRAIN_DAYS)]
        if len(series) < MIN_TRAIN_BUCKETS:
            raise ValueError(f"Need at least {MIN_TRAIN_BUCKETS} 15-minute periods of data, have {len(series)}")

        metrics, shrink_xgb, shrink_rf = validate(series)
        bundle = fit_models(series)
        bundle.update({
            "shrink_xgb": shrink_xgb,
            "shrink_rf": shrink_rf,
            "metrics": metrics,
            "trained_at": int(time.time()),
            "buckets": int(len(series)),
        })
        with lock:
            joblib.dump(bundle, MODEL_FILE)

        forecasts = refresh_forecasts()
        return {"trained": True, "buckets": len(series), "metrics": metrics, "forecasts": forecasts}
    finally:
        train_lock.release()


# ----------------------------
# ThingSpeak sync
# ----------------------------
def thingspeak_get(params: dict) -> dict:
    query = urllib.parse.urlencode({"api_key": THINGSPEAK_READ_KEY, **params})
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL}/feeds.json?{query}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def ts_time(t: pd.Timestamp) -> str:
    # ThingSpeak expects "YYYY-MM-DD HH:MM:SS" in UTC
    return t.strftime("%Y-%m-%d %H:%M:%S")


def fetch_thingspeak(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """All feed entries in [start, end]. ThingSpeak returns the newest 8000, so page backwards."""
    entries = {}
    window_end = end
    while window_end >= start:
        feeds = thingspeak_get({
            "results": THINGSPEAK_PAGE, "start": ts_time(start), "end": ts_time(window_end)
        }).get("feeds", [])
        for f in feeds:
            entries[f["created_at"]] = f
        if len(feeds) < THINGSPEAK_PAGE:
            break
        oldest = pd.Timestamp(min(f["created_at"] for f in feeds)).tz_localize(None)
        window_end = oldest - pd.Timedelta(seconds=1)
    return list(entries.values())


def feed_to_row(feed: dict) -> dict:
    def num(key):
        try:
            return float(feed.get(key) or 0)
        except ValueError:
            return 0.0

    voltage, current, pf = num("field1"), num("field2"), num("field5")
    created = pd.Timestamp(feed["created_at"]).tz_localize(None)
    return {
        "timestamp": int((created - pd.Timestamp(0)) // pd.Timedelta(seconds=1)),
        "voltage": voltage,
        "current": current,
        "active_power": voltage * current * pf,
        "energy_consumption": 0.0,
        "power_factor": pf,
        "temperature": num("field6"),
        "humidity": num("field7"),
        "light_intensity": 0.0,
        "building_type": "",
        "occupancy_level": 0,
        "operational_schedule": "",
        "electricity_tariff": "",
        "appliance_category": "",
        "demand_kWh": voltage * current,
    }


def sync_from_thingspeak() -> dict:
    """
    Pull only the readings newer than the latest stored one (everything when the table
    is empty), then retrain if anything arrived, or just roll the forecasts forward if not.
    """
    if not sync_lock.acquire(blocking=False):
        return {"skipped": "Sync already in progress"}
    try:
        with lock:
            df = load_csv(DATA_FILE, COLUMNS)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if len(df):
            start = pd.Timestamp(int(df["timestamp"].max()) + 1, unit="s")
        else:
            created = thingspeak_get({"results": 0})["channel"]["created_at"]
            start = pd.Timestamp(created).tz_localize(None)

        rows = [feed_to_row(f) for f in fetch_thingspeak(start, now)]
        if rows:
            with lock:
                upsert(DATA_FILE, COLUMNS, pd.DataFrame(rows, columns=COLUMNS))

        result = {"added": len(rows), "from": int((start - pd.Timestamp(0)) // pd.Timedelta(seconds=1))}
        if rows or not os.path.exists(MODEL_FILE):
            try:
                result["train"] = train_models()
            except (TrainingSkipped, ValueError) as e:
                result["train"] = {"skipped": str(e)}
        else:
            result["forecasts"] = refresh_forecasts()

        last_sync.update({"at": int(time.time()), "added": len(rows), "error": None})
        return result
    except Exception as e:
        last_sync.update({"at": int(time.time()), "added": 0, "error": str(e)})
        raise
    finally:
        sync_lock.release()


# ----------------------------
# FastAPI backend
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    # Other fields the app sends are accepted and ignored: the forecast is time based
    timestamp: int


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/status")
def status():
    with lock:
        df = load_csv(DATA_FILE, COLUMNS)
        bundle = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    return {
        "rows": len(df),
        "first_timestamp": int(df["timestamp"].min()) if len(df) else None,
        "last_timestamp": int(df["timestamp"].max()) if len(df) else None,
        "model_trained": bundle is not None,
        "trained_at": bundle["trained_at"] if bundle else None,
        "metrics": bundle["metrics"] if bundle else None,
        "last_sync": last_sync,
    }


@app.post("/sync")
def sync():
    """Pull new readings from ThingSpeak now, retrain and refresh the forecasts."""
    try:
        return sync_from_thingspeak()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.api_route("/reset", methods=["POST", "DELETE"])
def reset():
    """Delete all readings, the trained models and stored forecasts. The next sync reloads everything."""
    with sync_lock, train_lock, lock:
        for path in (DATA_FILE, MODEL_FILE, FORECAST_FILE):
            if os.path.exists(path):
                os.remove(path)
    return {"reset": True}


@app.post("/train")
def train():
    """Retrain on the complete data set, then regenerate the upcoming forecasts."""
    try:
        return train_models()
    except TrainingSkipped as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/forecast")
def forecast():
    """Regenerate the upcoming forecasts from the latest data, without retraining."""
    try:
        return {"forecasts": refresh_forecasts()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/forecasts")
def forecasts(start: int = 0, end: int = 2**31 - 1):
    """Stored forecasts with start <= timestamp <= end (epoch seconds)."""
    with lock:
        df = load_csv(FORECAST_FILE, FORECAST_COLUMNS)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    return {"forecasts": df.to_dict(orient="records")}


@app.post("/predict")
def predict(req: PredictionRequest):
    """Forecast one moment from the latest data, and store it."""
    try:
        with lock:
            if not os.path.exists(MODEL_FILE):
                return JSONResponse(status_code=503, content={"error": "Model not trained yet"})
            bundle = joblib.load(MODEL_FILE)
            history = demand_series(load_csv(DATA_FILE, COLUMNS))
        if history.empty:
            return JSONResponse(status_code=503, content={"error": "No data to forecast from"})

        target = pd.Timestamp(req.timestamp, unit="s")
        # Forecast from the latest data before the target, never using the target's own data
        known = history[history.index < target]
        if known.empty:
            known = history
        origin = known.index.max()
        base, xgb, rf = predict_parts(bundle, known, [target], origin)

        row = pd.DataFrame([{
            "timestamp": req.timestamp,
            "pred_xgb": round(float(xgb[0]), 2),
            "pred_rf": round(float(rf[0]), 2),
            "pred_baseline": round(float(base[0]), 2),
            "made_at": int(time.time()),
        }])
        with lock:
            upsert(FORECAST_FILE, FORECAST_COLUMNS, row)

        return {
            "predicted_demand_kWh_xgb": float(row["pred_xgb"][0]),
            "predicted_demand_kWh_rf": float(row["pred_rf"][0]),
            "predicted_demand_baseline": float(row["pred_baseline"][0]),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
