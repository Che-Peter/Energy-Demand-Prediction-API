# Energy-Demand-Prediction-API

FastAPI service that forecasts energy demand from the ThingSpeak channel's readings
with XGBoost and RandomForest, and stores the forecasts.

ThingSpeak is the source of truth. Nothing is pulled automatically: when the app
displays data it calls `POST /sync`, which fetches only the readings newer than the
last stored one (all of them the first time), retrains on the complete data set and
regenerates the forecasts for the next 24 hours in 15-minute steps.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/sync` | Pull new ThingSpeak readings, retrain, refresh forecasts |
| GET | `/forecasts?start=&end=` | Stored forecasts between two epoch-second timestamps |
| POST | `/predict` | Forecast one moment `{"timestamp": ...}` and store it |
| GET | `/status` | Stored readings, training time and validation accuracy |
| POST | `/train` | Retrain without pulling |
| POST | `/forecast` | Refresh the next 24h of forecasts without retraining |
| POST/DELETE | `/reset` | Delete all readings, models and forecasts |

## Deployment (Render)

Live at https://energy-demand-prediction-api-m2kj.onrender.com, deployed from this repo
by `render.yaml` (free plan). Every push to `main` redeploys automatically.

- The free plan sleeps after 15 minutes without requests (the first request then takes
  about a minute) and its disk is wiped on every restart or redeploy: the next sync
  reloads everything from ThingSpeak and retrains, only past forecasts are lost.
  A free pinger (e.g. cron-job.org) calling `/status` every 10 minutes keeps it awake.
- `DATA_DIR` sets where data is stored (current folder by default).
- Run a single worker (the start command does).
- Optional: `THINGSPEAK_CHANNEL`, `THINGSPEAK_READ_KEY`.

## Testing

```
run_local.bat                      # backend on http://localhost:8000
python test_api.py                 # checks the local backend
python test_api.py https://energy-demand-prediction-api-m2kj.onrender.com   # safe on the live server; add --reset to wipe first
```

Open the web app with `?api=http://localhost:8000` to use the local backend.
