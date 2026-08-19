# Phase 2: Real-Data Integration Report

## Summary

Phase 2 integrates real Bangalore food-delivery data (Zomato/Courier, March 2022) into the adaptive-dispatch-engine's ML pipeline and produces simulator calibration parameters. All existing code, tests, and architecture are preserved.

---

## Data Source

- **Dataset**: HuggingFace `allenborochin/zomato_delivery_EDA`
- **Subset**: Bangalore (identified via `Delivery_person_ID` prefix `BANGRE`)
- **Rows**: 2,956 orders | 60 unique riders | 36 dates (March 2022)
- **File**: `data/external/bangalore_orders.csv`

### Target Distribution
| Stat | Value |
|------|-------|
| Mean | 26.3 min |
| Median | 26.0 min |
| Std | 9.2 min |
| Range | 10 – 54 min |
| P10 | 15.0 min |
| P90 | 39.0 min |

---

## ML Model (Real Data)

### Architecture Decision: Different Target Than Production

The existing production model predicts **prep time** (`actual_prep_duration_min`) using synthetic features (`items_count`, `workload_at_placement`, `staff_level`, `kitchen_id`) that do not exist in real data.

The real-data model predicts **delivery time** (`delivery_time_min` = total end-to-end) — a different but equally valid task. Both models coexist without conflict.

### Features

| Feature | Type | Source |
|---------|------|--------|
| `distance_km` | numeric | REAL-DATA-DERIVED (Haversine) |
| `hour_sin` | numeric | REAL-DATA-DERIVED (time-of-day) |
| `hour_cos` | numeric | REAL-DATA-DERIVED (time-of-day) |
| `order_complexity` | categorical | MAPPED from real `Type_of_order` |
| `weather_severity` | categorical | MAPPED from real `Weather_conditions` |
| `traffic_severity` | categorical | MAPPED from real `Road_traffic_density` |

### Synthetic-Only Features (Not Fabricatable)
- `items_count` — not in dataset
- `workload_at_placement` — no kitchen load data
- `staff_level` — no staffing data
- `kitchen_id` — no kitchen identification

### Split Strategy
Chronological temporal split (no leakage):
- Train: 2,020 rows (70%, dates 01-03 → 26-03)
- Calib: 432 rows (15%, dates 27-03 → 31-03)
- Test: 434 rows (15%, dates 01-04 → 06-04)

### Training Results

| Model | Calib MAE | Calib MAPE | Calib RMSE | Latency |
|-------|-----------|------------|------------|---------|
| LightGBM | 6.24 min | 27.7% | 7.92 min | 0.59 ms |
| XGBoost | 6.50 min | 28.8% | 8.27 min | 0.36 ms |
| **MLP** | **6.05 min** | **26.8%** | **7.62 min** | **0.18 ms** |

**Best model: MLP**

### Test Set Evaluation

| Metric | Value |
|--------|-------|
| MAE | 6.08 min |
| MAPE | 26.9% |
| RMSE | 7.66 min |
| Nominal coverage | 80% |
| Actual coverage | 80.2% |
| Mean interval width | 19.2 min |
| Conformal qhat | 9.58 min |

### Artifacts
Saved to `artifacts_real/` (production `artifacts/` untouched):
- `model.joblib` — trained MLP model (78 KB)
- `encoder.joblib` — fitted ordinal encoder (1.5 KB)
- `meta.json` — model metadata and calibration parameters

---

## Simulator Calibration

### Demand Curve (orders/hour, REAL-DATA-DERIVED)

```
08:00   3.8  ###
09:00   3.6  ###
10:00   4.1  ####
17:00   7.8  #######
18:00   7.8  #######
19:00   9.4  #########
20:00   7.5  #######
21:00   8.5  ########
22:00   8.8  ########
23:00   7.4  #######
```

Key insight: Real demand is **bimodal** — small morning peak (8-10 AM), large evening peak (5-11 PM). The simulator's default curve should be updated.

### Delivery Time by Weather (REAL-DATA-DERIVED)

| Weather | Mean (min) | vs Sunny |
|---------|-----------|----------|
| Sunny | 22.3 | baseline |
| Sandstorms | 26.3 | +18% |
| Stormy | 26.0 | +17% |
| Windy | 26.0 | +17% |
| Fog | 28.4 | +27% |
| Cloudy | 28.7 | +29% |

Note: Cloudy/Fog have higher delivery times than storms in real data — possibly due to visibility effects on navigation.

### Delivery Time by Traffic (REAL-DATA-DERIVED)

| Traffic | Mean (min) | vs Low |
|---------|-----------|--------|
| Low | 21.7 | baseline |
| Medium | 26.6 | +23% |
| High | 27.6 | +27% |
| Jam | 30.9 | +42% |

### Distance Distribution (REAL-DATA-DERIVED)

| Stat | Value |
|------|-------|
| Mean | 9.7 km |
| P10 | 3.1 km |
| P90 | 17.1 km |
| Range | 1.6 – 20.2 km |

Simulator default range: 1.0–3.5 km (too narrow).

### Fleet Mix (REAL-DATA-DERIVED)

| Vehicle | Share |
|---------|-------|
| motorcycle | 59.0% |
| scooter | 33.5% |
| electric_scooter | 7.4% |

Simulator: no fleet mix modeled.

### Festival Effect (REAL-DATA-DERIVED)

| Festival | Mean (min) | Count |
|----------|-----------|-------|
| No | 26.0 | 2,903 |
| Yes | 44.3 | 53 |

Festival causes **+70% delivery time increase**. Simulator: not modeled.

---

## Files Changed

| File | Status | Purpose |
|------|--------|---------|
| `models/real_features.py` | **NEW** | Real-data feature engineering module |
| `scripts/train_real_data.py` | **NEW** | Training script for real Bangalore data |
| `scripts/calibrate_simulator.py` | **NEW** | Simulator calibration from real data |
| `config/bangalore_calibration.json` | **NEW** | Calibration parameters and comparisons |
| `artifacts_real/model.joblib` | **NEW** | Trained MLP model |
| `artifacts_real/encoder.joblib` | **NEW** | Fitted ordinal encoder |
| `artifacts_real/meta.json` | **NEW** | Model metadata |
| `docs/PHASE2_INTEGRATION.md` | **NEW** | This report |

### Files NOT Changed (Preserved)
- `models/features.py` — production feature module (synthetic data)
- `models/train.py` — production training pipeline
- `models/predict.py` — production predictor
- `models/uncertainty.py` — conformal calibration
- `simulation/*` — all simulation code
- `api/*` — all API code
- `dashboard/*` — all dashboard code
- `dispatch/*` — all dispatch logic
- `tests/*` — all 122 tests pass with no changes

---

## Remaining Synthetic Assumptions

These parameters remain synthetic — no real data available:

1. **Prep time model** (`kitchen.py` BASE_PREP_RANGES) — no real prep time data
2. **Kitchen workload/staffing** — no real kitchen load data
3. **Kitchen count and staff levels** — synthetic parameter
4. **Rider count and speed** — synthetic (22 km/h assumed)
5. **Cancellation rates** — synthetic parameter
6. **Weather state transitions** — Markov chain probabilities synthetic
7. **Traffic spike probability** — synthetic parameter
8. **Rider hub-to-kitchen distance** — synthetic parameter
9. **Pickup time at kitchen** — synthetic (0.5 min assumed)
10. **Order items count distribution** — synthetic parameter

---

## Verification

- All 122 tests pass (`pytest tests/ -q`)
- Dashboard build clean (`npm run build`)
- Real-data model artifacts saved to `artifacts_real/`
- Calibration JSON saved to `config/bangalore_calibration.json`
- No production code modified
- No production artifacts overwritten

---

## Next Steps (Phase 3 — Awaiting User Approval)

1. **Apply calibration to simulator** — update `DEFAULT_CONFIG` demand curve, distance range, weather/traffic factors based on real calibration data
2. **Wire real-data model into API** — optional endpoint for real-data predictions alongside production model
3. **Run comparison experiments** — synthetic-data model vs real-data model on simulated scenarios
4. **Dashboard integration** — display real-data model metrics and calibration data
