from typing import Dict, Tuple, Set

class TrafficAnalytics:
    """Calculates counting metrics and virtual line crossing events."""

    def __init__(self, line_y: int = 250):
        self.line_y = line_y
        self.counted_ids: Set[int] = set()
        self.crossed_count = 0

    def process_positions(self, tracked_objects: Dict[int, Tuple[int, int]]) -> int:
        for obj_id, centroid in tracked_objects.items():
            if obj_id not in self.counted_ids:
                if centroid[1] >= self.line_y:
                    self.counted_ids.add(obj_id)
                    self.crossed_count += 1
        return self.crossed_count

    def get_stats(self) -> dict:
        return {
            "total_crossed": self.crossed_count,
            "active_objects": len(self.counted_ids)
        }
