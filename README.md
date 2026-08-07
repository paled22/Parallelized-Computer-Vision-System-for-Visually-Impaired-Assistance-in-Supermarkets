# Edge-Optimized Computer Vision System for Visually Impaired Assistance
🌍 *Read this in [Spanish](README-es.md)*

## Overview
This repository contains a proof-of-concept computer vision pipeline designed to assist visually impaired individuals in locating specific products on dense supermarket shelves. The project explores the trade-off between model accuracy, inference speed, and model size by adapting a standard YOLOv8n architecture for potential Edge Computing deployment.

## System Architecture
The system utilizes a cascade pipeline:
* **Detection (Architectural Parallelism):** A modified YOLOv8 model that replaces standard convolutions with depthwise separable convolutions (YOLO-DWConv). This mathematical parallelism processes input channels independently, optimizing hardware resource usage.
* **Classification:** A YOLOv8-cls model that classifies the cropped regions of interest into 25 distinct supermarket categories.
* **Accessibility Module:** Integration with the `gTTS` library to transform spatial bounding box coordinates into directional audio instructions (e.g., "left", "center", "right").

## Key Results
* **Efficiency & Size:** The DWConv architectural modifications successfully reduced total parameters by 18.8% (from 3,005,843 to 2,440,211) and computational load from 8.1 to 7.2 GFLOPs.
* **Concurrency vs. Precision Trade-off:** Implemented software-level concurrency using Python's `ThreadPoolExecutor` for batch image processing. This approach yielded a 45.3% improvement in inference speed, though we documented the expected threading overhead (due to the GIL) and an overall mAP50 precision drop from 0.8173 to 0.6840.
* **Edge Deployment Prep:** Validated mobile deployment viability by performing an initial INT8 quantization via LiteRT/TFLite, achieving a ~3.6x reduction in model size.

## Dataset
This project was trained and evaluated using the **Freiburg Groceries Dataset**. 
*(Note: Datasets are not included in this repository to keep it lightweight. Please download the dataset independently and place it in the `/data` directory before running the notebook).*

## Tech Stack
Python, PyTorch, Ultralytics YOLO, OpenCV, `concurrent.futures`, Google Colab, TFLite.
