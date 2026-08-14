# Adaptive Dispatch Engine

AI-assisted operational simulator for ultra-fast cloud-kitchen delivery.

- **Phase 1 (done):** SimPy discrete-event simulation core + synthetic data generation.
- **Phase 2 (done):** ML prep-time prediction — LightGBM/XGBoost/MLP vs rule baseline,
  conformal prediction intervals, persisted in-process `Predictor`.
- **Phase 3 (done):** Dispatch engine — adaptive vs immediate policies, dynamic risk
  buffer, decision logging (see §Phase 3).
- **Phase 4 (done):** FastAPI backend + WebSocket streaming — REST routers, background
  thread runner, live snapshots, sim control over HTTP (see §Phase 4).
- **Phase 5 (done):** Real-time dashboard — Vite + React 19 SPA served by the Phase 4
  API, live `/ws` snapshots, sim control, policy compare (see §Phase 5).
- **Phase 6 (done):** Root-cause analysis for late orders — structured per-order and
  aggregate attribution of lateness to kitchen prep, queue, dispatch, rider travel,
  rider wait, and customer travel.
- **Phase 7 (done):** 10,000+ paired randomized experiments — run Immediate vs Adaptive
  on identical conditions across thousands of seeds to measure statistical significance
  and condition-dependent effects.
- **Phase 8 (done):** Enhanced operations dashboard — root-cause visualization,
  experiment results, and order investigation drill-down.

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

Artifacts (`encoder.joblib`, `model.joblib`, `meta.json`) are gitignored. The
`meta.json` holds the split-conformal `calibration_quantile` and nominal
coverage used to build the prediction interval. Use the predictor:

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

Reference result (9 pooled runs, 6383 trainable rows; train/calibration/test =
4468/957/958): best model (selected on the calibration split, not the test set)
is XGBoost — MAE 1.42 min vs rule-baseline 2.13 min on the untouched temporal
test split. Prediction intervals are calibrated with split conformal prediction
on a held-out calibration set: **80% nominal** interval, **≈ 72% empirical test
coverage**. The nominal 80% is an exact match on the calibration window
(≈ 0.80) but not on the test window, because the pooled-runs "temporal" test
split is effectively a held-out time-of-day window (hours 19–23 of every run),
where prep-time errors are genuinely larger. The README therefore reports
"80% nominal with ≈ 72% empirical test coverage" and does **not** claim a
measured 80% interval.

## Phase 3 - Dispatch Engine

The dispatch engine decides *when* to send a rider for each order. Two policies
share one interface (`dispatch/`):

- `immediate` - anti-pattern baseline: dispatch at placement.
- `adaptive` - dispatch so the rider reaches the kitchen at the predicted
  ready time: `dispatch_at = placed + prep_estimate - travel_to_kitchen -
  risk_buffer`, using the lower bound of the calibrated (conformal) prep-time
  prediction interval plus a risk buffer that widens with prediction uncertainty
  and kitchen congestion.
  Dispatch is also promise-aware: if food-readiness timing would make the order
  miss its delivery SLA, the rider is dispatched earlier (toward the latest
  safe time), so orders at risk are never scheduled late by the policy.

```bash
# single policy
python run_dispatch.py --policy immediate --scenario normal --seed 42 --days 1
python run_dispatch.py --policy adaptive  --scenario normal --seed 42 --days 1 --predictor-dir artifacts

# side-by-side comparison on the same seed
python run_dispatch.py --compare --scenario normal --seed 42 --days 1 --predictor-dir artifacts
python run_dispatch.py --compare --scenario rain   --seed 42 --days 1 --predictor-dir artifacts
```

Reference result (seed 42, 1 day, 10 riders, 15-min promise, on the corrected
training data): adaptive and immediate are tied on customer-facing on-time %
(72.8%) — both sit near the physical floor of the 15-min promise — but adaptive
cuts rider kitchen-wait (2.44 vs 2.55 min) at lower cost score (5053 vs 5109).
Under rain adaptive again edges the cost score (6261 vs 6268) at 61.6% on-time.

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
normal, adaptive, 10 riders): placed=402, on-time ≈ 72.8%, cost ≈ 5053 —
identical to the Phase 3 CLI results.

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
  Phase 4/3 result: adaptive cost 5053 vs immediate 5109, rider kitchen-wait
  2.44 vs 2.55; on-time is tied at ~72.8%.

## Phase 6 - Root-Cause Analysis

Post-simulation analysis that answers: **"Why was this order late?"**

For every completed order, the analysis reconstructs the timeline:

```
placed → kitchen_queue → kitchen_prep → dispatch → rider_to_kitchen → rider_wait → pickup → customer_travel → delivered
```

Some stages overlap (e.g., kitchen prep runs while rider travels). The analysis
distinguishes **contributing delays** from the **bottleneck/limiting factor** that
actually caused the SLA breach.

### Root-cause categories

- `KITCHEN_PREP` — actual prep duration exceeded budget
- `KITCHEN_QUEUE` — order waited for kitchen capacity
- `DISPATCH_DELAY` — dispatch decision made late
- `RIDER_TRAVEL_TO_KITCHEN` — hub-to-kitchen transit time
- `RIDER_WAIT_AT_KITCHEN` — rider arrived before food ready
- `CUSTOMER_TRAVEL` — kitchen-to-customer transit time
- `MULTIPLE_FACTORS` — no single dominant cause

### API endpoints

```
GET /analysis/root-causes          # aggregate + top late orders
GET /analysis/root-causes/orders/{id}  # per-order detail
```

### CLI usage

```bash
# Run simulation first
python run_dispatch.py --policy adaptive --scenario normal --seed 42 --days 1 --predictor-dir artifacts

# Then query via API (while server running):
curl http://127.0.0.1:8000/analysis/root-causes
```

### Example output

```json
{
  "aggregate": {
    "total_orders": 311,
    "late_orders": 116,
    "on_time_rate": 0.627,
    "primary_cause_percentages": {
      "KITCHEN_PREP": 38.4,
      "CUSTOMER_TRAVEL": 27.1,
      "RIDER_WAIT_AT_KITCHEN": 19.7
    }
  },
  "late_orders": [...]
}
```

---

## Phase 7 - 10,000+ Paired Randomized Experiments

Determines whether Adaptive dispatch actually outperforms Immediate across
realistic operating conditions.

### Methodology

**Paired experiments**: for each random seed/environment, run BOTH policies on
identical conditions:

```
Same environment (seed)
      |
      +---- Immediate
      |
      +---- Adaptive
```

This controls for order arrival patterns, kitchen workload, traffic, weather,
and rider hub distances — isolating the policy effect.

**Win criterion — end-to-end delivery time.** A paired run is scored by its
average end-to-end delivery time (the customer-facing SLA). Adaptive wins when
it is faster than Immediate by more than 0.1 min on average; Immediate wins when
the reverse holds; otherwise the run is a tie. The full delivery-time picture is
reported alongside the wins: P50/P90/P95/P99/max delivery-time differences,
on-time rate, and late-order counts. Cost score remains in the output for
transparency only and never decides a winner.

**Distribution capture.** During each run the per-order delivery times are
accumulated into pooled histograms (adaptive and immediate) and saved as
`delivery_distribution.json`, powering the distribution / CDF / percentile
visualizations on the dashboard.

### Randomized variables

- Order volume (via `demand_multiplier`)
- Weather severity (clear/rain/storm with transitions)
- Traffic spikes (stochastic)
- Kitchen staffing level
- Customer distance distribution

### CLI

```bash
# Single scenario, 1000 paired runs
python run_experiment.py --experiments 1000 --scenario normal --base-seed 42

# All scenarios, 500 per scenario (2500 total paired runs)
python run_experiment.py --multi-scenario --experiments-per-scenario 500

# Full 10k experiment
python run_experiment.py --experiments 10000 --scenario normal
```

Outputs written to `data/experiments/<scenario>/`:
- `experiments.csv` — per-seed paired metrics (both policies + differences)
- `experiment_summary.json` — aggregate statistics (wins, delivery-time diffs, percentiles)
- `delivery_distribution.json` — pooled delivery-time histograms / CDFs per policy

### API endpoints

```
POST /experiments/run      # start background experiment
GET  /experiments/status   # poll progress
GET  /experiments/results  # load latest results
GET  /experiments/results/{scenario}  # load scenario-specific results
```

### Example output

```
======================================================================
EXPERIMENT SUMMARY: normal (1 day(s), 1000 paired runs)
======================================================================

WIN CRITERION: average end-to-end delivery time (min)
  (tie unless |diff| > 0.10 min)

WIN COUNTS:
  Adaptive wins:  647
  Immediate wins: 289
  Ties:           64
  Adaptive win rate: 64.7%

METRIC DIFFERENCES (Adaptive - Immediate):
  Metric                                        Mean     Median        Std
  ----------------------------------------------------------------------
  On-time % (pp)                                +1.24      +1.10      2.31
  Avg delivery (min)                            -0.18      -0.15      0.42
  P50 delivery (min)                            -0.11
  P90 delivery (min)                            -0.33
  P95 delivery (min)                            -0.41
  P99 delivery (min)                            -0.55
  Max delivery (min)                            -0.61
  Late orders (#)                               -2.30
  Avg lateness (min)                            -0.21      -0.18      0.35
  Avg order wait (min)                          -0.12
  Avg rider kitchen wait (min)                  -0.41      -0.38

NOTE: Negative values mean Adaptive is better (lower is better for all metrics except on-time %)
      Positive on-time % means Adaptive has higher on-time rate
      Cost score differences are reported for transparency only; wins use delivery time.
======================================================================
```

---

## Phase 8 - Enhanced Operations Dashboard

The dashboard now includes three new sections:

### 1. Root Cause Analysis Panel
- **Primary cause breakdown** — horizontal bar chart with percentages
- **Contributing factors** — secondary bar chart
- **Late orders table** — sortable, clickable for drill-down
- **Order detail modal** — stage durations with visual bars

### 2. Experiment Results Panel
- **Win counts** — Adaptive / Immediate / Ties with win rate (wins by average end-to-end delivery time, 0.1-min tie threshold)
- **Metric comparison table** — mean/median/std for delivery percentiles (P50/P90/P95/P99/max), on-time rate, late orders, waits; cost score shown for transparency only
- **Cross-scenario breakdown** — win rate and delivery-time diffs per scenario
- **Statistical notes** — methodology reminders
- **Run controls** — trigger 1k or multi-scenario experiments

### 3. Delivery-Time Distributions Panel
- **Histogram overlay** — pooled delivery-time distribution, Adaptive vs Immediate per scenario
- **CDF curves** — share of orders delivered by each elapsed-minute mark
- **Percentile bars** — P50/P90/P95/P99 absolute delivery times side by side per policy
- **Scenario × tail heatmap** — mean diff in minutes (green = Adaptive faster, red = Immediate faster) across Avg/P50/P90/P95/P99/Max/Late-count

### 3. Order Investigation
Click any late order in the Root Cause table to see:
```
ORDER #1842
Delivery Time: 18.7 min
Promise:       15.0 min
Lateness:      3.7 min
Primary Cause: KITCHEN_PREP
Contributing:  RIDER_TRAVEL_TO_KITCHEN

Stage Durations:
  Kitchen Queue:          1.2 min
  Kitchen Prep:           6.1 min
  Dispatch Delay:         0.8 min
  Rider -> Kitchen:       3.2 min
  Rider Wait at Kitchen:  0.4 min
  Kitchen -> Customer:    7.0 min
```

---

## Architecture

```
Simulation (SimPy)
    ↓
Dispatch Policy (Immediate | Adaptive)
    ↓
Order Lifecycle (timestamps at each stage)
    ↓
Metrics (dispatch/metrics.py)
    ├── Root Cause Analysis (dispatch/root_cause.py)
    │   ├── Per-order bottleneck identification
    │   └── Aggregate statistics
    └── Experiment Engine (dispatch/experiment.py)
        ├── 10,000+ paired runs
        ├── Statistical aggregation
        └── Results storage (CSV + JSON)
    ↓
Dashboard (React 19 + WebSocket)
    ├── Live KPIs
    ├── Root Cause Panel
    ├── Experiment Results Panel
    └── Order Investigation Modal
```

---

## Root-Cause Methodology

The analysis uses a **bottleneck heuristic** rather than "longest duration wins":

1. **Reconstruct timeline** from stored timestamps
2. **Identify overlapping stages** (prep || rider travel)
3. **Check rider wait** — if rider waits at kitchen, prep was the bottleneck
4. **Check customer travel** — if no wait but travel is high, that's the bottleneck
5. **Check dispatch delay** — if order sat waiting for dispatch
6. **Check queue** — if order waited for kitchen capacity
7. **Fallback** — largest non-trivial duration if no clear bottleneck

**Limitation**: With current simulator granularity, we cannot prove causality —
only identify the limiting factor in the observed timeline. The heuristic is
defensible but not ground truth.

---

## Experiment Methodology

- **Paired design**: Same seed → same arrivals, prep times, hub distances, weather/traffic
- **Win criterion**: Average end-to-end delivery time (tie unless |diff| > 0.1 min); cost score reported for transparency only
- **Metrics**: On-time rate, P50/P90/P95/P99/max delivery time, late-order count, lateness, order wait, rider wait — all as Adaptive − Immediate differences
- **Statistics**: Mean, median, std across runs; paired t-test applicable
- **Distribution capture**: Per-order delivery times pooled into histograms/CDFs per policy and scenario
- **No multiprocessing**: Sequential for determinism and simplicity

---

## How to Run

### 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Train predictor (required for Adaptive)

```bash
python generate_dataset.py --scenarios normal,rain,low_staffing --seeds 1,2,3 --days 2
python train_models.py --data-dir data/train --out artifacts
```

### 3. Start API + Dashboard

```bash
npm --prefix dashboard install
npm --prefix dashboard run build
python run_api.py --scenario normal --seed 42 --policy adaptive
```
Open http://127.0.0.1:8000/

### 4. Run single simulation (CLI)

```bash
python run_dispatch.py --policy adaptive --scenario normal --seed 42 --days 1 --predictor-dir artifacts
python run_dispatch.py --compare --scenario normal --seed 42 --days 1 --predictor-dir artifacts
```

### 5. Run experiments (CLI)

```bash
# Quick test (100 paired runs)
python run_experiment.py --experiments 100 --scenario normal

# Full 10k on a single scenario (large runs take hours)
python run_experiment.py --experiments 10000 --scenario normal

# Multi-scenario: 5 scenarios x 2000 = 10,000 paired runs
python run_experiment.py --multi-scenario --experiments-per-scenario 2000

# Multi-scenario quick check
python run_experiment.py --multi-scenario --experiments-per-scenario 100
```

### 6. Run tests

```bash
pytest -q
```

### 7. Query root-cause via API

```bash
# After running a simulation via API
curl http://127.0.0.1:8000/analysis/root-causes
curl http://127.0.0.1:8000/analysis/root-causes/orders/42
```

---

## Tests

```bash
pytest -q
```

