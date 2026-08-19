# Phase 1: Data Understanding & Preprocessing — Real Food-Delivery Dataset

## 1. Dataset Source

- **Origin**: [HuggingFace allenborochin/zomato_delivery_EDA](https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA)
- **Raw file**: `data/external/zomato_cleaned.csv` (6.4 MB, 38,964 rows × 22 columns)
- **License**: MIT
- **Date range**: Feb–Apr 2022 (multi-city India)

## 2. City Identification & Extraction

**Problem**: The `City` column only contains `Metropolitian` / `Urban` / `Semi-Urban` — not actual city names.

**Solution**: The `Delivery_person_ID` field encodes the city in its first 6 characters:
```
BANGRES17DEL01 → stem "BANGRE" → Bangalore
MUMRES02DEL01  → stem "MUMRES" → Mumbai
```

All 22 city stems were mapped. **2,956 Bangalore rows extracted** (7.6% of total).

### City distribution (all data)

| City | Rows |
|------|------|
| Bangalore | 2,956 |
| Coimbatore | 2,961 |
| Indore | 2,932 |
| Pune | 2,920 |
| Chennai | 2,914 |
| Ranchi | 2,384 |
| + 15 more cities | ~24,897 |

## 3. Bangalore Data Summary

| Property | Value |
|----------|-------|
| Rows | 2,956 |
| Unique dates | 36 (March 2022) |
| Unique riders | 60 |
| Missing values | `Delivery_person_Age`: 80 (2.7%), `Time_Orderd`: 70 (2.4%) |

### Column inventory

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `Delivery_person_ID` | str | BANGRES17DEL01 | Rider ID, encodes city |
| `Order_Date` | str | 12-02-2022 | DD-MM-YYYY |
| `Time_Orderd` | str | 21:55 or 0.458 | HH:MM or decimal fraction |
| `Time_Order_picked` | str | 22:10 | When rider picked up |
| `Weather_conditions` | str | Fog, Stormy, etc. | 6 values |
| `Road_traffic_density` | str | Low/Medium/High/Jam | 4 values |
| `Type_of_order` | str | Snack/Meal/Drinks/Buffet | 4 values |
| `Type_of_vehicle` | str | motorcycle/scooter/electric_scooter | 3 values |
| `Vehicle_condition` | int | 0, 1, 2 | |
| `multiple_deliveries` | float | 0.0–3.0 | Batch deliveries |
| `Festival` | str | Yes/No | |
| `Restaurant_lat/lon` | float | 12.97, 77.59 | Restaurant location |
| `Delivery_location_lat/lon` | float | 13.02, 77.70 | Customer location |
| `distance_km` | float | 1.6–20.2 | Computed distance |
| `delivery_speed` | str | Fast/Average/Slow | Derived |
| `Time_taken (min)` | int | 10–54 | **Target variable** |

## 4. Feature Mapping: Existing Pipeline ↔ Real Dataset

### Current ML features (`models/features.py`)

| Current Feature | Real Dataset Column | Derivable? | Mapping Strategy |
|----------------|--------------------|----|----|
| `items_count` | `Type_of_order` | Partial | Map order type → item count range (Snack=1-3, Meal=3-5, etc.) |
| `workload_at_placement` | — | **No** | Not in dataset; would need kitchen-level queue data |
| `staff_level` | — | **No** | Not in dataset |
| `hour_sin`, `hour_cos` | `Time_Orderd` | **Yes** | Parse hour → cyclical encoding |
| `order_complexity` | `Type_of_order` | **Yes** | Snack/Drinks→simple, Meal→standard, Buffet→complex |
| `weather_severity` | `Weather_conditions` | **Yes** | Sunny/Cloudy/Windy→clear, Fog→rain, Stormy/Sandstorms→storm |
| `kitchen_id` | `Restaurant_lat/lon` | Partial | Cluster lat/lon → kitchen zones (requires clustering step) |

### Target variable

| Current Target | Real Dataset | Notes |
|---------------|-------------|-------|
| `actual_prep_duration_min` | `Time_taken (min)` | ⚠️ This is **total delivery time**, not prep time alone. The dataset has no prep-vs-delivery split. Can be used as a proxy or to calibrate delivery-time distributions. |

## 5. Simulator Parameters Calibratable from Real Data

| Simulator Parameter | Real Data Source | Calibration Method |
|--------------------|-----------------|-------------------|
| **Order arrival rates by hour** | `Time_Orderd` | Hour histogram → Poisson rates per hour bucket |
| **Delivery time distribution** | `Time_taken (min)` | Fit lognormal/empirical CDF |
| **Weather effects on delivery** | `Weather_conditions` × `Time_taken` | Mean delta: Sunny=22min, Fog=28min, Stormy=26min |
| **Traffic effects on delivery** | `Road_traffic_density` × `Time_taken` | Mean delta: Low=22min, Jam=31min |
| **Distance distribution** | `distance_km` | Fit empirical: mean=9.7km, range 1.6–20.2km |
| **Vehicle type mix** | `Type_of_vehicle` | 59% motorcycle, 34% scooter, 7% electric |
| **Festival impact** | `Festival` × `Time_taken` | Festival: 44min avg vs 26min normal (+69%) |
| **Order type distribution** | `Type_of_order` | 26% Meal, 26% Snack, 25% Drinks, 23% Buffet |
| **Batch delivery frequency** | `multiple_deliveries` | Distribution of concurrent deliveries |

## 6. Key Findings

### What the real data tells us about Bangalore delivery

1. **Delivery times are much higher than the 10-min Swish promise**: Mean=26.3 min, median=26 min, max=54 min. This aligns with the Swish review research (20-45 min actuals).

2. **Traffic is the biggest factor**: Jam deliveries take 31 min vs 22 min for Low traffic (+41%).

3. **Weather has moderate impact**: Fog/rain adds ~3 min to delivery time.

4. **Festivals cause massive disruption**: 44 min average vs 26 min normal (+69%).

5. **Motorcycle-dominated fleet**: 59% motorcycle, which matches Bangalore's two-wheeler culture.

6. **Evening peak**: Hour distribution peaks at 19:00–22:00 (7-10 PM), consistent with dinner rush.

### What the real data does NOT provide

- **No prep time vs delivery time split**: `Time_taken` is total end-to-end. Cannot isolate kitchen prep from transit.
- **No kitchen workload/staffing**: Cannot derive `workload_at_placement` or `staff_level`.
- **No order items count**: Only categorical order type (Snack/Meal/Drinks/Buffet).
- **No kitchen ID**: Only lat/lon, which could be clustered but adds complexity.

## 7. Files Created

| File | Purpose |
|------|---------|
| `data/external/zomato_cleaned.csv` | Raw multi-city dataset (38,964 rows) |
| `data/external/bangalore_orders.csv` | Bangalore-only extracted data (2,956 rows, 31 columns) |
| `scripts/prepare_real_data.py` | Reproducible preprocessing script |

## 8. Recommendations for Phase 2+

1. **Use real data for simulator calibration** — order arrival rates, delivery time distributions, weather/traffic effects, fleet mix.
2. **Accept `Time_taken` as a delivery-time proxy** — not prep time, but still valuable for validating the simulator's delivery component.
3. **Kitchen workload/staffing remain synthetic** — no real data source found; keep current synthetic model for these.
4. **Order complexity mapping is straightforward** — Snack/Drinks→simple, Meal→standard, Buffet→complex maps cleanly to existing `VALID_COMPLEXITY`.
