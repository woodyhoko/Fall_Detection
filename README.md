# Fall Detection for IoT Small Sensors

**Ho Ko · Yueqi Liu · Jiaxin Xie** — ECE, University of Waterloo

📄 **[Read the Paper](https://github.com/woodyhoko/Fall_Detection/blob/main/paper.pdf)**

---

## Abstract

> *We propose two system frameworks for elderly fall detection deployable on ESP32-cam devices under $9 USD — combining CNN and RNN models with distributed and centralized IoT architectures for cost-efficient, low-power fall detection.*

Falls are the leading cause of injury among the elderly. Existing solutions are either expensive (depth cameras, Kinect), require complex wearable setups, or demand too much compute for cheap IoT devices. This paper proposes ML models specifically designed to run on the **ESP32-cam** — a $9 microcontroller with an integrated camera — making the system accessible and scalable for home deployment.

---

## Key Contributions

### 1. Two IoT System Architectures

**Distributed Analysis**
```
[ESP32-cam] ──► [Local ML inference] ──► [Cloud alert]
[ESP32-cam] ──► [Local ML inference] ──► [Cloud alert]
```
Each sensor runs the full model locally. Simple to set up and expand. Higher compute requirement per device, but fully autonomous.

**Centralized Analysis**
```
[ESP32-cam] ──► feature vectors ──► [Raspberry Pi 4] ──► [Cloud alert]
[ESP32-cam] ──► feature vectors ──┘
```
Sensors handle only feature extraction; a central Raspberry Pi 4 performs fall classification. Lower per-sensor compute, more flexible task extension, but requires local network infrastructure.

### 2. Two Detection Models

The analysis pipeline has two stages:
1. **Feature Extraction** — CNN-based spatial feature extraction from video frames
2. **Temporal Classification** — RNN-based sequence classifier that detects the fall event pattern across time

Both models are designed to be lightweight enough to partially or fully execute on ESP32-class hardware.

---

## Why ESP32-cam?

| Solution Type | Cost | Power | Deployment |
|---|---|---|---|
| Kinect / depth camera | High | High | Complex |
| Wearable (accelerometer) | Medium | Medium | User compliance issues |
| Raspberry Pi + USB camera | ~$91 | Medium | Limited range per device |
| **ESP32-cam (ours)** | **<$9** | **Low** | **Simple, scalable** |

---

## Dataset

**[IMVIA Fall Detection Dataset](https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html)** — a public benchmark of annotated video sequences of fall events and daily-living activities (walking, sitting, lying for rest) recorded in a controlled environment.

Key challenge: distinguishing a fall from intentional floor activity (yoga, resting), which many existing systems fail at.

---

## Results

Model 2 (the centralized architecture with separate feature extraction and RNN classification) proved more accurate than Model 1 and is the recommended configuration for real deployments. Full quantitative results are in the paper (Section 4).

---

## Repository Contents

| File | Description |
|---|---|
| `dataset_generation.ipynb` | Frame extraction, annotation parsing, train/val split |
| `fall detection.ipynb` | Model training, evaluation, result charts |
| `fall_model.h5` | Saved Keras weights — main fall detection model |
| `flowmodel.h5` | Saved Keras weights — optical flow auxiliary model |
| `FallDataset_anno.zip` | IMVIA dataset annotations |
| `paper.pdf` | Full paper (9 pages) — IEEE format |

---

## Stack

- Python, TensorFlow / Keras
- OpenCV (frame extraction, optical flow)
- Jupyter Notebooks

---

## Run

```bash
pip install tensorflow opencv-python jupyter numpy
# Step 1: preprocess dataset
jupyter notebook dataset_generation.ipynb
# Step 2: train and evaluate
jupyter notebook "fall detection.ipynb"
```

> Pre-trained weights (`fall_model.h5`, `flowmodel.h5`) are included — skip Step 1 to evaluate directly.
