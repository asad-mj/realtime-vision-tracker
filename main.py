import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from src.detector import ObjectDetector
from src.tracker import SimpleCentroidTracker
from src.analytics import TrafficAnalytics

app = FastAPI(
    title="Real-Time Vision & Analytics Service",
    description="High-performance object detection, tracking, and spatial analytics engine.",
    version="1.0.0"
)

detector = ObjectDetector()
tracker = SimpleCentroidTracker()
analytics = TrafficAnalytics(line_y=300)

class TrackedItem(BaseModel):
    object_id: int
    centroid: List[int]

class FrameAnalysisResponse(BaseModel):
    detections_count: int
    tracked_count: int
    cumulative_line_crossings: int
    tracked_objects: List[TrackedItem]

@app.get("/health", tags=["Monitoring"])
def health():
    return {"status": "healthy", "service": "vision-tracker"}

@app.post("/analyze-frame", response_model=FrameAnalysisResponse, tags=["Inference"])
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    boxes, _, _ = detector.detect(frame)
    tracked = tracker.update(boxes.tolist())
    crossings = analytics.process_positions(tracked)

    tracked_items = [
        TrackedItem(object_id=obj_id, centroid=list(pos))
        for obj_id, pos in tracked.items()
    ]

    return {
        "detections_count": len(boxes),
        "tracked_count": len(tracked),
        "cumulative_line_crossings": crossings,
        "tracked_objects": tracked_items
    }
