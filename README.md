# Adaptive Dispatch Engine

AI-assisted operational simulator for ultra-fast cloud-kitchen delivery.

- **Phase 1 (done):** SimPy discrete-event simulation core + synthetic data generation.
- **Phase 2 (done):** ML prep-time prediction — LightGBM/XGBoost/MLP vs rule baseline,
  quantile prediction intervals, persisted in-process `Predictor`.
- **Phase 3 (done):** Dispatch engine — adaptive vs immediate policies, dynamic risk
  buffer, decision logging (see §Phase 3).
- **Phase 4 (done):** FastAPI backend + WebSocket streaming — REST routers, background
  thread runner, live snapshots, sim control over HTTP (see §Phase 4).
- **Phase 5 (done):** Real-time dashboard — Vite + React 19 SPA served by the Phase 4
  API, live `/ws` snapshots, sim control, policy compare (see §Phase 5).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Phase 1 — Simulation

```bash
python run_sim.py --scenario normal --seed 42 --days 1
```

Outputs are written to `data/<scenario>_seed<seed>_day<days>/`:

- `event_log.csv` — raw simulation events
- `orders.csv` — order-level training data for the prep-time model

Scenarios in `config/scenarios/`: `normal`, `lunch_rush`, `rain`,
`low_staffing`, `traffic_spike`.

## Phase 2 — ML Prep-Time Prediction

Generate a pooled training set (scenarios × seeds → `data/train/`):

```bash
python generate_dataset.py --scenarios normal,rain,low_staffing --seeds 1,2,3 --days 2
```

Train all models, calibrate intervals, persist artifacts:

```bash
python train_models.py --data-dir data/train --out artifacts
```

Artifacts (`encoder.joblib`, `model.joblib`, `q_low.joblib`, `q_high.joblib`,
`meta.json`) are gitignored. Use the predictor:

```python
from models.predict import Predictor
p = Predictor.load("artifacts")
p.predict({
    "items_count": 2, "workload_at_placement": 5.0, "staff_level": 3,
    "hour_of_day": 19, "order_complexity": "standard",
    "weather_severity": "rain", "traffic_severity": "normal", "kitchen_id": 1,
})
# -> {'prep_mean': .., 'prep_low': .., 'prep_high': .., 'uncertainty': 'low'|'medium'|'high'}
```

Reference result (9 pooled runs, 701 rows): LightGBM MAE 2.45 min vs
rule-baseline 5.92 min; 80% interval coverage ≈ 0.72 on temporal test split.

## Phase 3 - Dispatch Engine

The dispatch engine decides *when* to send a rider for each order. Two policies
share one interface (`dispatch/`):

- `immediate` - anti-pattern baseline: dispatch at placement.
- `adaptive` - dispatch so the rider reaches the kitchen at the predicted
  ready time: `dispatch_at = placed + prep_estimate - travel_to_kitchen -
  risk_buffer`, using the calibrated lower quantile of the prep prediction plus
  a risk buffer that widens with prediction uncertainty and kitchen congestion.

```bash
# single policy
python run_dispatch.py --policy immediate --scenario normal --seed 42 --days 1
python run_dispatch.py --policy adaptive  --scenario normal --seed 42 --days 1 --predictor-dir artifacts

# side-by-side comparison on the same seed
python run_dispatch.py --compare --scenario normal --seed 42 --days 1 --predictor-dir artifacts
python run_dispatch.py --compare --scenario rain   --seed 42 --days 1 --predictor-dir artifacts
```

Reference result (seed 42, 1 day, normal): adaptive and immediate are tied on
customer-facing delivery time and on-time % (both hit the physical floor of the
10-min promise), but adaptive cuts rider kitchen-wait from 2.62 to 1.44 min -
lower cost score (14840 vs 15050). Under rain adaptive wins across the board
(cost 18318 vs 19639, lower order wait, higher on-time %).

Metrics: `on_time_rate`, `avg_delivery_min`, `avg_late_min`,
`avg_order_wait_min` (food ready, waiting on a rider), `avg_rider_wait_kitchen_min`
(rider waiting on food), `avg_rider_idle_min` (unassigned, for transparency), and
`cost_score = 1x wasted rider time + 5x order wait + 10x late`. Wasted rider time
counts only assigned-but-waiting time; unassigned availability is not dispatch waste.

Configuration: `config/dispatch.yaml` (risk buffers, congestion buffer, cost
weights, promise, hub distance range, pickup/penalty times, weather/traffic
factors). Additive Phase 3 columns land in `orders.csv`.

## Phase 4 - FastAPI Backend & Streaming

The simulator runs live behind a FastAPI server. A background thread steps the
SimPy engine on a configurable clock and pushes JSON snapshots over a WebSocket;
the same snapshots back the REST endpoints. Start it with:

```bash
python run_api.py --scenario normal --seed 42 --policy adaptive --speed 60
```

- `--speed N` = sim-minutes per wall-clock second (60 = one sim-day in 24s);
  omit for fast-as-possible (CI/tests).
- `--autostart` starts the run on boot; otherwise use `POST /sim/start`.

Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /sim/status` | latest snapshot (sim time, running/paused/finished, KPIs) |
| `GET /sim/config` / `POST /sim/config` | read / change scenario, seed, policy, speed |
| `POST /sim/start` / `POST /sim/pause` / `POST /sim/resume` | run control |
| `POST /sim/step {"minutes": 5}` | advance exactly N minutes while paused |
| `POST /sim/reset {"scenario": "rain", "seed": 7, "policy": "immediate"}` | re-run with new config |
| `GET /orders?status=completed&limit=100` / `GET /orders/{id}` | order lifecycle |
| `POST /prediction` | prep-time forecast + interval (no sim side effects) |
| `POST /dispatch` | what the active policy decides for a synthetic order |
| `GET /dispatch/decisions?limit=50` | recent dispatch decisions, newest first |
| `WS /ws?scenario=..&seed=..&policy=..&speed=..` | live snapshot stream |

The `/ws` message is the dashboard contract: `sim_time_min`, scenario/policy/
seed, weather/traffic, `kitchens` (queues + orders), `riders` (status, busy,
assigned order), `recent_decisions`, `metrics` (running KPIs via
`dispatch/metrics.py`), and the last 50 `events`. Live API run (seed 42,
normal, adaptive): placed=402, on-time ≈ 14.9%, cost=14840.2 — identical to the
Phase 3 CLI results.

Quick WebSocket check (PowerShell):

```powershell
$ws = [System.Net.WebSockets.ClientWebSocket]::new()
$ws.ConnectAsync([Uri]"ws://127.0.0.1:8000/ws?scenario=normal&seed=42&policy=adaptive&speed=60", [Threading.CancellationToken]::None).GetAwaiter().GetResult()
# read 2-3 snapshots, sim_time_min increases monotonically, then $ws.CloseAsync(...)
```

## Phase 5 - Real-Time Dashboard

A single-page dashboard that renders the Phase 4 snapshot contract live over
WebSocket, with REST-based run control and a side-by-side policy compare. It is
a pure client: it makes no API changes and ships no charting/map libraries
(bars are plain CSS).

Build once, then serve:

```bash
npm --prefix dashboard install
npm --prefix dashboard run build          # tsc --noEmit && vite build -> dashboard/dist
python run_api.py --scenario normal --seed 42 --policy adaptive
```

Open <http://127.0.0.1:8000/> — the API serves the built SPA at `/` whenever
`dashboard/dist/index.html` exists (`--no-dashboard` disables it). Routes
(`/ws`, `/sim/*`, `/dispatch/*`) still win because the static mount is
registered last.

During development, run the Vite dev server instead (hot reload, proxies `/ws`
and all REST paths to the API on 127.0.0.1:8000):

```bash
npm --prefix dashboard run dev            # http://127.0.0.1:5173
```

Dashboard features:

- Live header: scenario/seed/policy, weather + traffic badges, run status,
  connection state (with reconnect backoff), sim clock.
- 8 KPI cards fed by the streaming `metrics` (orders, on-time %, cost score,
  delivery/wait times).
- Kitchens panel (queues + in-flight orders), Riders panel (status/busy/
  assigned), dispatch-decisions table, and a live event log.
- Controls: scenario / policy / seed / days / speed, Start / Pause / Resume /
  Step +5 / Reset.
- Compare: runs `immediate` then `adaptive` to completion on the same seed via
  reset→start→poll and shows bar-for-bar KPI winners. Verified against the
  Phase 4/3 result: adaptive cost 14840 vs immediate 15050, rider kitchen-wait
  1.44 vs 2.62; immediate edges on-time % (15.4 vs 14.9).

## Tests

```bash
pytest -q
```

