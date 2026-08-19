# Code Deletion Log

## [2026-08-15] Refactor Session — Dead Code Cleanup

### Unused Dependencies Removed
- react-window@2.3.0 (dashboard) — Last used by deleted `src/components/VirtualizedEventLog.tsx`, Size: ~14 kB
- @tanstack/react-query-devtools@5.101.4 (dashboard) — Never imported by any source file
- (lockfile `dashboard/package-lock.json` re-synced via `npm install`)

### Unused Files Deleted
- dashboard/src/components/VirtualizedEventLog.tsx — 0 imports anywhere; App.tsx uses `./components/EventLog` instead
- dashboard/src/components/VirtualizedEventLog.css — only imported by the deleted .tsx

### Unused Exports Removed
- dashboard/src/types.ts — `PairedResult`, `ExperimentMetricDiffs`, `ExperimentPolicyMetrics`
  - Reason: only referenced among themselves inside types.ts; no component imports them
- dashboard/src/api.ts — `api.config()` and `api.orderRootCause()`
  - Reason: zero call sites in `src` (rg-verified); backend endpoints `/sim/config` and
    `/analysis/root-causes/orders/{id}` remain (tested via test_api.py); only client wrappers removed
- dashboard/src/api.ts — `RootCauseAnalysis` type import (was only used by `orderRootCause`)

### Dead Python Code Removed
- dispatch/experiment.py — `run_single_simulation()` (public wrapper, 0 callers; `_run_single_simulation` retained)
- dispatch/root_cause.py — `format_root_cause_analysis()`, `format_aggregate_analysis()` (0 callers)
- Unused imports removed:
  - api/runner.py: `import tempfile`, `import logging`, `logger = ...`
  - api/broadcast.py: `import logging`, `logger = ...`
  - api/routers/dispatch.py: `OrderComplexity`
  - api/routers/experiment.py: `print_summary`, `BackgroundTasks` (import + `background_tasks` param),
    dead `global _experiment_running` declaration (name never assigned in that scope)
  - dispatch/experiment.py: `format_metrics`, `PairedDiffStats`, `analyze_orders`, `aggregate_root_causes`
  - simulation/cancellations.py, delivery.py, dispatcher.py, environment.py, kitchen.py,
    order_generator.py: `import simpy` (simpy only used in simulation/engine.py)
  - simulation/order_generator.py: `OrderComplexity`; simulation/event_log.py: `OrderStatus`
  - run_dispatch.py: `import tempfile`
- dispatch/experiment.py — 5 placeholder-less f-strings converted to plain strings (pyflakes F541)

### Verified and KEPT (suspicious candidates)
- Dashboard `DeliveryDistributionPanel` CDF-fallback in `percentileValues` — LIVE, not dead:
  3 of 5 checked-in `data/experiments/*/delivery_distribution.json` (normal, rain, traffic_spike)
  lack `percentiles`; removing the fallback would break P50/P90/P95/P99 rendering for those scenarios
- types.ts `StageDurations`, `RootCauseAggregate`, `KitchenOrder`, `ExperimentSummary`,
  `DistributionSeries` — flagged as "used in module" by knip/ts-prune but referenced by exported
  interfaces (RootCauseAnalysis, RootCausesResponse, KitchenState, ExperimentResultsResponse)
- dispatch/experiment.py `_run_single_simulation`, `compute_summary`, `save_results`, `load_results`,
  `print_summary`, `DistributionAccumulator`, `run_paired_experiment`, `run_experiment`,
  `run_multi_scenario_experiment` — used by tests, run_experiment.py, scripts/regen_distributions.py
- dispatch/root_cause.py `analyze_orders`, `aggregate_root_causes` — used internally by
  `aggregate_root_causes` and by api/routers/analysis.py
- `save_individual` param — used by tests and scripts
- simulation/dispatcher.py `DispatchState.from_env` — used by simulation
- api/routers/sim.py `_publish` — no duplicate publish logic (single wiring through
  start/pause/resume/step/reset)

### Not Scanned (out of scope)
- models/*.py — pyflakes shows unused imports there (pandas/numpy/lightgbm/xgboost) but models/
  was not in scope; left untouched

### Impact
- Files deleted: 2
- Dependencies removed: 2
- Unused imports/functions/types removed: ~25
- Bundle size reduction: dashboard JS 265.83 kB (was ~280+ kB); pyflakes clean across
  api/, dispatch/, simulation/, scripts/, root scripts

### Testing
- `python -m pytest -q` — 122 passed, 410 warnings (matches baseline)
- `npm run build` (dashboard) — tsc --noEmit + vite build clean
- `python -m pyflakes api dispatch simulation run_api.py run_sim.py run_dispatch.py
  run_experiment.py train_models.py generate_dataset.py scripts` — no output (clean)
- npx knip — only 5 "used in module" types remain (kept deliberately)
- npx ts-prune — same 5 "used in module" types (kept deliberately)
- Running API server (pid from data/api.pid) unaffected; WebSocket connections still accepted
