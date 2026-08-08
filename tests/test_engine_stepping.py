import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine, load_scenario  # noqa: E402


@pytest.mark.parametrize("steps", [(1, 3, 5), (7, 13, 100)])
def test_stepped_run_matches_one_shot(tmp_path, steps):
    out1 = tmp_path / "oneshot"
    out2 = tmp_path / "stepped"

    config = load_scenario("normal", seed=42)
    config["days"] = 1
    one = SimulationEngine(config, out_dir=str(out1), scenario_name="normal")
    one.run()

    config = load_scenario("normal", seed=42)
    config["days"] = 1
    eng = SimulationEngine(config, out_dir=str(out2), scenario_name="normal")
    eng._setup()
    total = eng.total_minutes
    t = 0
    for step in steps:
        t = min(t + step, total)
        eng.advance(t)
        if eng.is_finished:
            break
    if not eng.is_finished:
        eng.advance(total)
    eng.finalize()

    h1 = hashlib.sha256((out1 / "event_log.csv").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out2 / "event_log.csv").read_bytes()).hexdigest()
    assert h1 == h2
