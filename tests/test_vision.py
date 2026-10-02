import numpy as np
from fastapi.testclient import TestClient
from main import app
from src.tracker import SimpleCentroidTracker
from src.analytics import TrafficAnalytics

def test_api_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

def test_tracker_registration():
    tracker = SimpleCentroidTracker()
    boxes = [[10, 10, 50, 50], [100, 100, 150, 150]]
    tracked = tracker.update(boxes)
    assert len(tracked) == 2
    assert 1 in tracked
    assert 2 in tracked

def test_analytics_crossing():
    analytics = TrafficAnalytics(line_y=50)
    # Object below threshold
    stats = analytics.process_positions({1: (20, 60)})
    assert stats == 1
    # Repeated count check (should not recount same ID)
    stats2 = analytics.process_positions({1: (20, 70)})
    assert stats2 == 1
