import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.order_generator import hour_of_day, complexity_from_items  # noqa: E402
from simulation.entities import OrderComplexity  # noqa: E402


def test_hour_of_day():
    assert hour_of_day(0) == 0
    assert hour_of_day(60) == 1
    assert hour_of_day(1439) == 23
    assert hour_of_day(1440) == 0
    assert hour_of_day(2880) == 0


def test_complexity_buckets():
    assert complexity_from_items(1) == OrderComplexity.SIMPLE
    assert complexity_from_items(2) == OrderComplexity.SIMPLE
    assert complexity_from_items(3) == OrderComplexity.STANDARD
    assert complexity_from_items(5) == OrderComplexity.STANDARD
    assert complexity_from_items(6) == OrderComplexity.COMPLEX
