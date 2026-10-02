import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.detector import ObjectDetector
from src.tracker import SimpleCentroidTracker
from src.analytics import TrafficAnalytics

st.set_page_config(page_title="Vision Analytics Hub", page_icon="👁️", layout="wide")
st.title("👁️ Real-Time Vision Analytics & Object Tracking Engine")

@st.cache_resource
def load_models():
    return ObjectDetector(), SimpleCentroidTracker(), TrafficAnalytics(line_y=200)

detector, tracker, analytics = load_models()

uploaded_file = st.file_uploader("Upload an Image Frame for Analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    boxes, confs, classes = detector.detect(frame)
    tracked = tracker.update(boxes.tolist())
    crossings = analytics.process_positions(tracked)

    # Annotate frame
    annotated = frame.copy()
    h, w, _ = frame.shape
    cv2.line(annotated, (0, 200), (w, 200), (0, 255, 255), 2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for obj_id, (cx, cy) in tracked.items():
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"ID: {obj_id}", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.metric("Total Detections", len(boxes))
        st.metric("Active Tracked IDs", len(tracked))
        st.metric("Virtual Line Crossings", crossings)
