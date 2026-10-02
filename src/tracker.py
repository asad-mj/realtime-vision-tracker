import numpy as np
from typing import Dict, List, Tuple

class SimpleCentroidTracker:
    """Lightweight persistent ID tracker based on Euclidean centroid distance."""

    def __init__(self, max_disappeared: int = 15):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: Tuple[int, int]) -> int:
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[List[float]]) -> Dict[int, Tuple[int, int]]:
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register((input_centroids[i][0], input_centroids[i][1]))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Compute pairwise distance between existing object centroids and input centroids
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = (input_centroids[col][0], input_centroids[col][1])
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_cols = set(range(input_centroids.shape[0])).difference(used_cols)
        for col in unused_cols:
            self.register((input_centroids[col][0], input_centroids[col][1]))

        return self.objects
