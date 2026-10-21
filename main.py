"""
Energy demand forecasting API.

ThingSpeak is the source of truth. Nothing is pulled automatically: when the app
displays data it calls POST /sync, which fetches only the readings newer than the
last stored one (everything when the table is empty), retrains on the complete data
set and regenerates the next 24h of forecasts.

For training, readings are cleaned and averaged into 15-minute buckets. A forecast for
time t, made at "origin" (the latest data), is:

    baseline = hour-of-day profile(t) * level(origin)
    model    = baseline * (1 + shrink * residual_model(features))

- profile: recency-weighted mean demand for each hour of the day over the recent weeks
- level: demand over the last 24h relative to what the profile expected for those hours,
  so the forecast follows level shifts (e.g. a building using half its usual power)
- residual models (XGBoost and RandomForest) learn, as a fraction of the baseline, what
  it misses (hour-specific changes, the last hour's deviation fading with horizon...).
  Recent days weigh most. Each retrain backtests them the way the app uses them (train up
  to a past moment, forecast the next 24h, repeat) and `shrink` scales their correction
  down when it does not help.
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
# DATA_DIR sets where data, models and forecasts are stored; otherwise the current folder.
# On hosts with a wiped disk (e.g. Render's free plan) the next sync reloads everything
# from ThingSpeak and retrains; only past forecasts are lost.
DATA_DIR = (os.environ.get("DATA_DIR")
            or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
            or ".")
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
MIN_VALIDATION_BUCKETS = 48               # below this, residual models are not trusted
PROFILE_DAYS = 28                         # profile uses the most recent weeks only
PROFILE_HALFLIFE_DAYS = 7                 # ...each week counting half as much as the next
SAMPLE_HALFLIFE_DAYS = 3                  # residual models weight recent behaviour most
LEVEL_WINDOW = "24h"                      # recent level the profile is scaled to
BACKTEST_DAYS = 4                         # validation: 24h forecasts from cutoffs over these days
BACKTEST_STEP = pd.Timedelta(hours=12)
MAX_TRAIN_DAYS = 180                      # cap training cost as data grows
MODEL_VERSION = 2                         # bump when the bundle format changes; forces a retrain

FEATURES = ["hsin", "hcos", "level", "anomaly", "horizon"]

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


def load_bundle():
    """The trained models, or None if missing or saved by an older version of this file."""
    if not os.path.exists(MODEL_FILE):
        return None
    bundle = joblib.load(MODEL_FILE)
    return bundle if bundle.get("version") == MODEL_VERSION else None


def hourly_profile(series: pd.Series) -> pd.Series:
    """
    Daily shape: recency-weighted mean demand per hour of the day. Hours with no data
    are interpolated from their neighbours and the shape is lightly smoothed.
    """
    recent = series[series.index >= series.index.max() - pd.Timedelta(days=PROFILE_DAYS)]
    age = np.asarray((recent.index.max() - recent.index) / pd.Timedelta(days=1))
    w = 0.5 ** (age / PROFILE_HALFLIFE_DAYS)
    g = pd.DataFrame({"v": recent.values * w, "w": w, "h": recent.index.hour}).groupby("h").sum()
    p = (g["v"] / g["w"]).reindex(range(24))
    # Three copies so interpolation and smoothing wrap around midnight
    p3 = pd.concat([p, p, p], ignore_index=True).interpolate(limit_direction="both")
    p3 = p3.rolling(3, center=True, min_periods=1).mean()
    return pd.Series(p3.values[24:48], index=range(24))


def recent_level(history: pd.Series, profile: pd.Series, origins) -> np.ndarray:
    """
    Recent demand relative to the profile: mean of the last LEVEL_WINDOW of readings
    divided by the profile's mean over the same timestamps (1.0 when unknown).
    """
    if history.empty:
        return np.ones(len(origins))
    expected = pd.Series(profile.reindex(history.index.hour).values, index=history.index)
    num = asof(history.rolling(LEVEL_WINDOW).mean(), origins)
    den = asof(expected.rolling(LEVEL_WINDOW).mean(), origins)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    return np.where(np.isfinite(ratio) & (den > 0), ratio, 1.0)


def build_features(targets, history: pd.Series, profile: pd.Series, origins) -> pd.DataFrame:
    """
    Features for forecasting `targets` from what was known at `origins`.
    Only data in `history` up to each origin is used, so nothing leaks from the future.
    Also holds "baseline": the profile scaled to the recent level.
    """
    targets = pd.DatetimeIndex(targets)
    origins = pd.DatetimeIndex(origins)
    f = pd.DataFrame(index=targets)

    hour_frac = targets.hour + targets.minute / 60
    f["hsin"] = np.sin(2 * np.pi * hour_frac / 24)
    f["hcos"] = np.cos(2 * np.pi * hour_frac / 24)

    f["level"] = recent_level(history, profile, origins)
    f["baseline"] = profile.reindex(targets.hour).values * f["level"].values

    # How far the last hour sits above/below the baseline, as a fraction
    recent_1h = np.nan_to_num(asof(history.rolling("1h").mean(), origins), nan=float(history.mean()))
    with np.errstate(divide="ignore", invalid="ignore"):
        f["anomaly"] = np.clip(np.nan_to_num(recent_1h / f["baseline"].values - 1), -1, 3)

    f["horizon"] = ((targets - origins) / pd.Timedelta(minutes=1)).values
    return f


def training_rows(series: pd.Series, profile: pd.Series):
    """Every bucket is a target once per training horizon."""
    frames, ys = [], []
    for h in HORIZONS_MIN:
        origins = series.index - pd.Timedelta(minutes=h)
        frames.append(build_features(series.index, series, profile, origins))
        ys.append(series.values)
    return pd.concat(frames), np.concatenate(ys)


def relative_residual(y, baseline) -> np.ndarray:
    """What the residual models learn: actual / baseline - 1, clipped against outliers."""
    return np.clip(np.asarray(y) / np.maximum(baseline, 1e-6), 0, 4) - 1


def fit_models(series: pd.Series):
    profile = hourly_profile(series)
    X, y = training_rows(series, profile)
    target = relative_residual(y, X["baseline"].values)
    age = np.asarray((series.index.max() - X.index) / pd.Timedelta(days=1))
    weight = 0.5 ** (age / SAMPLE_HALFLIFE_DAYS)

    xgb = XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_child_weight=20, reg_lambda=10,
        random_state=42, objective="reg:squarederror"
    )
    xgb.fit(X[FEATURES], target, sample_weight=weight)

    rf = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=50, max_features=0.5, max_depth=6,
        random_state=42, n_jobs=-1
    )
    rf.fit(X[FEATURES], target, sample_weight=weight)

    return {"xgb": xgb, "rf": rf, "profile": profile, "version": MODEL_VERSION}


def raw_parts(bundle, history: pd.Series, targets, origin):
    """(profile only, baseline, xgb correction, rf correction), before shrinking."""
    targets = pd.DatetimeIndex(targets)
    X = build_features(targets, history, bundle["profile"], [origin] * len(targets))
    return (bundle["profile"].reindex(targets.hour).values, X["baseline"].values,
            bundle["xgb"].predict(X[FEATURES]), bundle["rf"].predict(X[FEATURES]))


def predict_parts(bundle, history: pd.Series, targets, origin):
    """Returns (baseline, xgb, rf) arrays for the targets."""
    _, baseline, xgb_raw, rf_raw = raw_parts(bundle, history, targets, origin)
    xgb = baseline * (1 + bundle["shrink_xgb"] * xgb_raw)
    rf = baseline * (1 + bundle["shrink_rf"] * rf_raw)
    # Demand can't be negative
    return np.maximum(baseline, 0), np.maximum(xgb, 0), np.maximum(rf, 0)


def mae(a, b):
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 2)


def best_shrink(baseline, correction, truth) -> float:
    """How much of a residual model's correction to keep, fitted on unseen data."""
    step = baseline * correction
    denom = float((step * step).sum())
    if denom <= 0:
        return 0.0
    return float(np.clip((step * (truth - baseline)).sum() / denom, 0, 1))


def validate(series: pd.Series):
    """
    Backtest exactly what the app shows: every BACKTEST_STEP over the last BACKTEST_DAYS,
    train only on the data up to that moment and forecast the next 24h. Returns error
    metrics and how much to trust each residual model.
    """
    end = series.index.max()
    cutoffs = pd.date_range(end - pd.Timedelta(days=BACKTEST_DAYS), end - BACKTEST_STEP, freq=BACKTEST_STEP)
    parts = {"profile": [], "baseline": [], "xgb": [], "rf": [], "truth": []}
    for cutoff in cutoffs:
        train = series[series.index <= cutoff]
        test = series[(series.index > cutoff) & (series.index <= cutoff + pd.Timedelta(hours=FORECAST_HOURS))]
        if len(train) < MIN_TRAIN_BUCKETS or test.empty:
            continue
        fitted = fit_models(train)
        for key, values in zip(["profile", "baseline", "xgb", "rf"],
                               raw_parts(fitted, train, test.index, train.index.max())):
            parts[key].append(values)
        parts["truth"].append(test.values)

    if sum(len(t) for t in parts["truth"]) < MIN_VALIDATION_BUCKETS:
        return None, 0.0, 0.0
    profile, base, xgb_raw, rf_raw, truth = (
        np.concatenate(parts[k]) for k in ["profile", "baseline", "xgb", "rf", "truth"])

    shrink_xgb = best_shrink(base, xgb_raw, truth)
    shrink_rf = best_shrink(base, rf_raw, truth)

    metrics = {
        "validation_buckets": int(len(truth)),
        "mean_demand": round(float(truth.mean()), 2),
        "mae_profile_only": mae(profile, truth),
        "mae_baseline": mae(np.maximum(base, 0), truth),
        "mae_xgb": mae(np.maximum(base * (1 + shrink_xgb * xgb_raw), 0), truth),
        "mae_rf": mae(np.maximum(base * (1 + shrink_rf * rf_raw), 0), truth),
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
        bundle = load_bundle()
        history = demand_series(load_csv(DATA_FILE, COLUMNS))
    if bundle is None or history.empty:
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
        if rows or load_bundle() is None:
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
        bundle = load_bundle()
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
            bundle = load_bundle()
            if bundle is None:
                return JSONResponse(status_code=503, content={"error": "Model not trained yet"})
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
