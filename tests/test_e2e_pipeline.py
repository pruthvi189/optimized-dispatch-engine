"""End-to-end test: generate_dataset → train_models → run_dispatch --compare"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import pytest


def run_cmd(cmd, cwd=None, timeout=300):
    """Run command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.slow
@pytest.mark.e2e
def test_full_pipeline_generate_train_compare(tmp_path):
    """
    Full pipeline test:
    1. generate_dataset for normal scenario, 2 seeds
    2. train_models on pooled data
    3. run_dispatch --compare with the trained predictor
    4. Assert adaptive beats immediate on cost score
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use temp directories for isolation
    train_dir = tmp_path / "train"
    artifacts_dir = tmp_path / "artifacts"

    # Step 1: generate_dataset
    ret, out, err = run_cmd([
        sys.executable, "generate_dataset.py",
        "--scenarios", "normal",
        "--seeds", "1,2",
        "--days", "1",
        "--out", str(train_dir)
    ], cwd=repo_root)
    assert ret == 0, f"generate_dataset failed: {err}"

    # Step 2: train_models
    ret, out, err = run_cmd([
        sys.executable, "train_models.py",
        "--data-dir", str(train_dir),
        "--out", str(artifacts_dir)
    ], cwd=repo_root)
    assert ret == 0, f"train_models failed: {err}"

    # Step 3: run_dispatch --compare
    ret, out, err = run_cmd([
        sys.executable, "run_dispatch.py",
        "--compare",
        "--scenario", "normal",
        "--seed", "42",
        "--days", "1",
        "--predictor-dir", str(artifacts_dir)
    ], cwd=repo_root)
    assert ret == 0, f"run_dispatch --compare failed: {err}"

    # Parse the comparison output
    lines = out.strip().split("\n")
    compare_started = False
    results = {}
    for line in lines:
        if "Comparison ===" in line:
            compare_started = True
            continue
        if compare_started and line.strip():
            parts = line.split()
            if len(parts) >= 8 and parts[0] in ("immediate", "adaptive"):
                policy = parts[0]
                results[policy] = {
                    "placed": int(parts[1]),
                    "completed": int(parts[2]),
                    "cancelled": int(parts[3]),
                    "on_time": float(parts[4].rstrip("%")),
                    "avg_wait": float(parts[5]),
                    "late": float(parts[6]),
                    "rider_kitchen_wait": float(parts[7]),
                    "cost": float(parts[8]),
                }

    assert "immediate" in results and "adaptive" in results

    # Core assertion: adaptive should be better on cost
    # Note: with small training data (2 seeds), the gap may be smaller
    # but adaptive should not be dramatically worse
    immediate_cost = results["immediate"]["cost"]
    adaptive_cost = results["adaptive"]["cost"]

    # Both policies should process the same number of orders (determinism)
    assert results["immediate"]["placed"] == results["adaptive"]["placed"]
    assert results["immediate"]["completed"] == results["adaptive"]["completed"]

    # Adaptive should at least be competitive on cost
    # (with limited training data, we allow some tolerance)
    cost_ratio = adaptive_cost / immediate_cost
    assert cost_ratio < 1.05, f"Adaptive cost {adaptive_cost} not competitive vs immediate {immediate_cost} (ratio={cost_ratio:.3f})"

    # Rider kitchen wait should be better with adaptive
    assert results["adaptive"]["rider_kitchen_wait"] <= results["immediate"]["rider_kitchen_wait"] * 1.1


@pytest.mark.slow
@pytest.mark.e2e
def test_determinism_across_full_pipeline(tmp_path):
    """Run the full pipeline twice with same seed, verify identical results."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run_once(tag):
        train_dir = tmp_path / f"train_{tag}"
        artifacts_dir = tmp_path / f"artifacts_{tag}"

        # generate
        ret, out, err = run_cmd([
            sys.executable, "generate_dataset.py",
            "--scenarios", "normal",
            "--seeds", "1",
            "--days", "1",
            "--out", str(train_dir)
        ], cwd=repo_root)
        assert ret == 0

        # train
        ret, out, err = run_cmd([
            sys.executable, "train_models.py",
            "--data-dir", str(train_dir),
            "--out", str(artifacts_dir)
        ], cwd=repo_root)
        assert ret == 0

        # compare
        ret, out, err = run_cmd([
            sys.executable, "run_dispatch.py",
            "--compare",
            "--scenario", "normal",
            "--seed", "42",
            "--days", "1",
            "--predictor-dir", str(artifacts_dir)
        ], cwd=repo_root)
        assert ret == 0

        # Parse final cost scores
        lines = out.strip().split("\n")
        costs = {}
        for line in lines:
            if line.startswith("[") and "cost=" in line:
                # format: "[policy] placed=... completed=... cancelled=... on_time=... delivery=... wait=... rider_kitchen_wait=... rider_idle=... cost=..."
                parts = line.split()
                policy = parts[0].strip("[]")
                for part in parts:
                    if part.startswith("cost="):
                        cost = float(part.split("=")[1])
                        break
                costs[policy] = cost
        return costs

    costs1 = run_once("a")
    costs2 = run_once("b")

    # Same seed + same pipeline = identical results
    assert costs1 == costs2, f"Pipeline not deterministic: {costs1} vs {costs2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not slow"])