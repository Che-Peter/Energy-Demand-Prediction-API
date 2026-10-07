"""
Checks a running backend end to end.

    python test_api.py                                   # local backend on port 8000
    python test_api.py https://energy-demand-prediction-api-production.up.railway.app
    python test_api.py http://localhost:8000 --reset     # wipe first, then full reload

Without --reset nothing is deleted, so it is safe to run against the live server.
"""
import json
import sys
import time
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = (args[0] if args else "http://localhost:8000").rstrip("/")
RESET = "--reset" in sys.argv

failures = 0


def call(method, path):
    req = urllib.request.Request(BASE + path, method=method)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.load(resp)
            code = resp.status
    except urllib.error.HTTPError as e:
        code, body = e.code, json.loads(e.read() or b"{}")
    return code, body, time.time() - started


def check(name, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def when(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)) if ts else "-"


print(f"Testing {BASE}\n")

if RESET:
    code, body, _ = call("POST", "/reset")
    check("reset deletes all data", code == 200 and body.get("reset"), body)

code, before, _ = call("GET", "/status")
check("status responds", code == 200, f"{before.get('rows')} readings stored, last {when(before.get('last_timestamp'))}")

# First sync: everything if the table is empty, otherwise only newer readings
code, s1, secs = call("POST", "/sync")
check("sync pulls from ThingSpeak", code == 200 and "added" in s1,
      f"added {s1.get('added')} readings starting {when(s1.get('from'))} in {secs:.1f}s")
if before.get("last_timestamp"):
    check("sync started after the last stored reading (incremental)",
          s1.get("from") == before["last_timestamp"] + 1,
          f"last stored {before['last_timestamp']}, pulled from {s1.get('from')}")

code, mid, _ = call("GET", "/status")
check("model is trained", mid.get("model_trained"), f"metrics {mid.get('metrics')}")

# Second sync right away: nothing old may be pulled again
code, s2, secs = call("POST", "/sync")
check("second sync pulls only new readings",
      code == 200 and s2.get("from") == mid["last_timestamp"] + 1 and s2.get("added", 99) <= 5,
      f"added {s2.get('added')} in {secs:.1f}s")

code, after, _ = call("GET", "/status")
check("no duplicates stored", after["rows"] == mid["rows"] + s2.get("added", 0),
      f"{mid['rows']} -> {after['rows']}")

now = int(time.time())
code, fc, _ = call("GET", f"/forecasts?start={now}&end={now + 24 * 3600}")
rows = fc.get("forecasts", [])
check("next 24h of forecasts stored", code == 200 and len(rows) >= 90, f"{len(rows)} forecasts")
for r in rows[:4]:
    print(f"        {when(r['timestamp'])}  XGBoost {r['pred_xgb']}  RandomForest {r['pred_rf']}")

target = now + 2 * 3600
code, p, _ = call("POST", f"/predict")  # missing body must be rejected
check("predict rejects a request without a timestamp", code == 422)

req = urllib.request.Request(BASE + "/predict", method="POST",
                             data=json.dumps({"timestamp": target}).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as resp:
    p = json.load(resp)
check("predict a chosen time", "predicted_demand_kWh_xgb" in p, f"{when(target)}: {p}")

code, fc, _ = call("GET", f"/forecasts?start={target}&end={target}")
check("chosen-time forecast is stored", len(fc.get("forecasts", [])) == 1)

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)
