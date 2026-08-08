import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.riders import travel_time_min, RiderPool  # noqa: E402
from simulation.entities import RiderStatus  # noqa: E402


def test_travel_time_basic():
    t = travel_time_min(distance_km=2.0, speed_kmh=30.0)
    assert abs(t - 4.0) < 1e-9


def test_travel_time_scaled():
    t = travel_time_min(distance_km=2.0, speed_kmh=30.0, traffic_factor=1.3, weather_factor=1.1)
    assert abs(t - 4.0 * 1.3 * 1.1) < 1e-9


def test_travel_time_zero_speed():
    with pytest.raises(ValueError):
        travel_time_min(distance_km=2.0, speed_kmh=0.0)


def test_rider_pool_assign_and_release():
    pool = RiderPool({"riders": {"count": 2, "speed_kmh": 25.0}})
    assert pool.idle_count() == 2
    rider = pool.assign_next_idle()
    assert rider is not None
    assert rider.status == RiderStatus.ASSIGNED
    assert pool.idle_count() == 1
    pool.release(rider.rider_id)
    assert pool.idle_count() == 2


def test_rider_pool_exhausted():
    pool = RiderPool({"riders": {"count": 1, "speed_kmh": 25.0}})
    pool.assign_next_idle()
    assert pool.assign_next_idle() is None


def test_rider_unavailable():
    pool = RiderPool({"riders": {"count": 1, "speed_kmh": 25.0}})
    pool.mark_unavailable(1)
    assert pool.riders[0].status == RiderStatus.UNAVAILABLE
    assert pool.assign_next_idle() is None
