# 👁️ Real-Time Vision Analytics & Object Tracking Engine

[![CI Pipeline](https://github.com/asad-mj/realtime-vision-tracker/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/asad-mj/realtime-vision-tracker/actions/workflows/ci-cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00599C?style=flat-square)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv)](https://opencv.org)

A computer vision microservice architected for multi-object detection, persistent centroid tracking, trajectory calculation, and automated line-crossing event analytics.

---

## 🏗️ Architecture

```text
Video Frame ──► YOLO Inference ──► Centroid Tracker ──► Analytics Engine ──► Annotated Output
                     │                     │                    │
              Bounding Boxes         Persistent IDs       Crossings & Stats
