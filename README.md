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

## Deployment (Railway)

- Attach a volume. Data is stored at its mount path automatically
  (`RAILWAY_VOLUME_MOUNT_PATH`); without a volume every redeploy wipes the data and
  the next sync reloads everything from ThingSpeak. `DATA_DIR` overrides the location.
- Run a single worker (the Procfile does).
- Optional: `THINGSPEAK_CHANNEL`, `THINGSPEAK_READ_KEY`.

## Testing

```
run_local.bat                      # backend on http://localhost:8000
python test_api.py                 # checks the local backend
python test_api.py https://<railway-url>   # safe on the live server; add --reset to wipe first
```

Open the web app with `?api=http://localhost:8000` to use the local backend.
