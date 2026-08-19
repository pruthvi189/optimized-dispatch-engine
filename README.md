# Adaptive Dispatch Engine

A simulation prototype for evaluating food-delivery dispatch strategies. It models order arrivals, kitchen operations, rider logistics, and delivery outcomes to compare how different dispatch policies perform under identical conditions.

The central question: **can a dispatch system make better kitchen+rider decisions by considering more than simple proximity?**

---

## Overview

A food-delivery order requires coordination between the customer, kitchen, and rider. A simple dispatch strategy assigns the nearest kitchen and nearest available rider. An adaptive strategy evaluates multiple kitchen+rider combinations and estimates end-to-end delivery time for each.

The Adaptive Dispatch Engine simulates this operational environment and compares the two approaches side by side. The simulator models:

- Order arrivals following a realistic hourly demand curve
- Kitchen capacity and per-order staffing levels
- Kitchen queue buildup and wait times
- Order-specific preparation time estimation
- Rider availability and dispatch timing
- Rider-to-kitchen travel
- Kitchen-to-customer travel with traffic and weather effects
- Dispatch decisions and their delivery outcomes

---

## Core Idea

The dashboard compares two dispatch policies:

### Baseline Dispatch (`nearest_heuristic`)

Selects the kitchen nearest to the customer, then assigns the nearest available rider to that kitchen. This is a straightforward heuristic that does not consider queue depth, rider travel time, or estimated delivery windows.

### Optimized Dispatch (`joint_optimizer`)

Evaluates available kitchen+rider combinations and selects the pair with the lowest estimated end-to-end delivery time. For each candidate pair, the estimated delivery time is:

```
max(rider → kitchen time, kitchen wait + prep time)
  + pickup time
  + kitchen → customer travel time
```

The `max()` accounts for parallelism: rider travel and kitchen preparation can happen simultaneously. If the rider arrives before the food is ready, the rider waits; if the food is ready before the rider arrives, the order waits. The optimizer finds the pair that minimizes the overall delivery window.

---

## Architecture

```
Demand / Orders
      ↓
Simulation Engine
      ↓
Kitchen + Rider State
      ↓
Dispatch Policy
 ┌────┴─────────────┐
 │                  │
Baseline        Joint Optimizer
 │                  │
 └────┬─────────────┘
      ↓
Dispatch Decision
      ↓
Simulation
      ↓
Metrics / Events
      ↓
Dashboard + Experiments
```

**Major components:**

| Directory | Purpose |
|-----------|---------|
| `simulation/` | SimPy discrete-event engine: order generation, kitchen process, rider pool, delivery process, spatial model, environment (weather, traffic) |
| `dispatch/` | Dispatch policies, ETA computation, metrics, root-cause analysis, experiment runner |
| `api/` | FastAPI backend with WebSocket streaming for real-time dashboard updates |
| `dashboard/` | React + TypeScript frontend with Baseline vs Optimized comparison, "Why This Dispatch?" visualization, kitchen/rider state |
| `config/` | Dispatch configuration, scenario presets (normal, rain, low_staffing, lunch_rush, traffic_spike) |
| `scripts/` | Calibration, validation, and analysis utilities |
| `models/` | Prediction artifacts (when applicable) |

---

## Simulation

The default configuration represents a simulated food-delivery operation:

| Parameter | Value |
|-----------|-------|
| Orders per day | ~422 |
| Kitchens | 4 (fixed locations in 22km x 22km service area) |
| Riders | 15 |
| Rider speed | 22 km/h |
| Kitchen staffing | [3, 3, 2, 2] staff per kitchen |
| Workload factor | 0.027 per queued order |
| Customer distance range | 3.0–17.0 km (Bangalore-calibrated) |
| Rider-to-kitchen distance | 0.5–2.0 km (synthetic per-rider matrix) |
| Pickup time | 0.5 min |
| Delivery promise | 40 min |
| Late-night orders | Included (00:00–03:00 at low rates) |

Customer locations are uniformly distributed within the service area. Distances are Euclidean (straight-line). Kitchen positions are fixed and chosen for geographic diversity across the service area.

This is a simulated operating environment, not a production fleet configuration.

---

## Prep-Time Estimation

The simulation uses an order-specific deterministic prep-time estimator. It considers:

- **Order complexity** — classified as Simple, Standard, or Complex based on item count
- **Item count** — scales prep time within the complexity tier's base range

Base prep ranges by complexity:

| Complexity | Items | Prep Range |
|------------|-------|------------|
| Simple | 1–2 | 3–6 min |
| Standard | 3–5 | 5–9 min |
| Complex | 6+ | 8–15 min |

The estimator produces a mean, low, and high estimate along with an uncertainty tier (low/medium/high). This is a transparent rule-based calculation, not an ML model.

---

## Why This Dispatch?

The dashboard includes a decision visualization that makes the optimizer's choices interpretable. For each dispatch decision, it shows:

- **Selected kitchen and rider** with a clear "SELECTED" indicator
- **Kitchen → customer distance** — how far the chosen kitchen is from the customer
- **Rider → kitchen distance** — how far the assigned rider is from the kitchen
- **Prep time** — estimated preparation time for this order
- **Kitchen wait** — current queue depth and staffing at the chosen kitchen
- **Estimated delivery time** — the optimizer's total delivery estimate

The optimizer can evaluate up to 4 kitchens × 15 riders = 60 candidate combinations per order, depending on rider availability. The visualization presents the factors that drove the decision rather than treating the optimizer as a black box.

---

## Results

**Seed 42 · 1-day simulation**

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Average delivery | 24.1 min | 22.6 min |
| P50 delivery | 21.7 min | 20.9 min |
| P95 delivery | 45.6 min | 37.7 min |
| P99 delivery | 63.9 min | 49.2 min |
| On-time rate | 93.3% | 96.4% |

In this run:

- Average delivery improves by 1.5 minutes.
- P95 delivery improves by 8.0 minutes.
- On-time rate improves by 3.1 percentage points.

These numbers are from a specific seed and scenario. They should not be interpreted as guaranteed production performance.

---

## Experiments & Diagnostics

Beyond the main Baseline vs Optimized comparison, the project includes additional policies and analysis tools:

- **Policy comparisons** — run multiple policies against the same seed/scenario
- **Delivery distributions** — histograms and CDFs of delivery times per policy
- **Kitchen/rider utilization** — queue depths, staff utilization, rider idle time
- **Root-cause analysis** — why specific orders were late, stage-by-stage breakdown
- **Dispatch decisions** — full evaluation log of candidate kitchen+rider pairs
- **Operational bottlenecks** — queue buildup, rider availability gaps, traffic effects

The main dashboard focuses on Baseline Dispatch and Optimized Dispatch. Other policies (immediate, adaptive, nearest_kitchen, optimized_kitchen) are available through the Experiments & Diagnostics interface and CLI.

---

## Metrics

| Metric | Definition |
|--------|------------|
| **Average Delivery Time** | Mean time from order placement to delivery (minutes). |
| **P95 Delivery Time** | 95th percentile delivery time — captures tail performance. |
| **On-Time Rate** | Percentage of orders delivered within the 40-minute promise. |
| **Order Wait** | Time food sits ready at the kitchen waiting for rider pickup (`pickup_at - prep_finished_at`). Measures handoff efficiency, not customer-facing delay. |

---

## Technology

**Backend / Simulation**
- Python, SimPy (discrete-event simulation), NumPy, Pandas

**Dispatch & Analysis**
- scikit-learn, SciPy (statistical analysis)

**API**
- FastAPI, Uvicorn, WebSockets

**Frontend**
- React 19, TypeScript, Vite, TanStack React Query

**Data**
- PyYAML (configuration), joblib (artifact persistence)

---

## Running Locally

### 1. Backend Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd dashboard
npm install
npm run build
cd ..
```

### 3. Start the API + Dashboard

```bash
python run_api.py --autostart --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

The dashboard connects via WebSocket and streams simulation state in real time.

### 4. Run a Simulation (CLI)

```bash
python run_dispatch.py --policy nearest_heuristic --scenario normal --seed 42 --days 1
python run_dispatch.py --policy joint_optimizer --scenario normal --seed 42 --days 1
```

### 5. Run Experiments (CLI)

```bash
python run_experiment.py --experiments 2000 --scenario normal
python run_experiment.py --multi-scenario --experiments-per-scenario 2000
```

### 6. Run Tests

```bash
pytest -q
```

---

## Project Structure

```
adaptive-dispatch-engine/
├── api/                  # FastAPI backend, WebSocket streaming, routers
├── config/               # Dispatch config, scenario presets
├── dashboard/            # React + TypeScript frontend
├── dispatch/             # Policies, ETA, metrics, root-cause, experiments
├── models/               # Prediction artifacts (when applicable)
├── scripts/              # Calibration, validation, analysis utilities
├── simulation/           # SimPy engine, entities, riders, spatial model
├── tests/                # Test suite
├── run_api.py            # Start API + dashboard
├── run_dispatch.py       # Run single policy via CLI
├── run_experiment.py     # Run paired experiments via CLI
└── requirements.txt      # Python dependencies
```

---

## Limitations

- This is a simulation/prototype, not a production dispatch system.
- Simulation assumptions affect results: Euclidean distances, single-order-per-rider, deterministic prep estimator.
- Current benchmark results are from a specific seed (42) and scenario (normal, 1-day).
- Synthetic operational inputs are not equivalent to live production telemetry.
- Results should not be interpreted as guaranteed real-world performance.
- Optimizer effectiveness depends on the operating regime and available decision signals.

---

## Design Principles

**Explainability** — Dispatch decisions should be understandable. The "Why This Dispatch?" visualization shows exactly which factors drove each decision.

**Comparable experiments** — Baseline and Optimized are always evaluated under identical simulation conditions (same seed, same arrivals, same environment).

**Operational realism** — The simulator models kitchen queues, capacity constraints, rider availability, traffic, and weather rather than treating dispatch as a pure routing problem.

**Honest evaluation** — Metrics come from the simulation. Results are reported for specific seeds and scenarios without overclaiming generality.
