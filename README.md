# Fall Detection via Computer Vision

A deep learning pipeline for **automatic fall detection** from video, combining optical flow analysis with pose estimation on the IMVIA benchmark dataset.

📄 **[Read the Paper](https://github.com/woodyhoko/Fall_Detection/blob/main/paper.pdf)**

---

## Overview

Falls are a leading cause of injury, particularly among the elderly. This project develops an automated detection system that identifies fall events in video sequences without wearable sensors.

The pipeline:
1. **Optical flow** estimation to capture inter-frame motion magnitude
2. **Pose keypoint** extraction per frame
3. **Classification** of fall vs. non-fall sequences using a trained model

---

## Dataset

**[IMVIA Fall Detection Dataset](https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html)** — a public benchmark containing annotated video sequences of fall and daily-living activities recorded in a controlled environment.

The dataset annotations are processed in `dataset_generation.ipynb` before model training.

---

## Repository Contents

| File | Description |
|---|---|
| `dataset_generation.ipynb` | Dataset loading, frame extraction, annotation preprocessing |
| `fall detection.ipynb` | Model training, evaluation, and results |
| `fall_model.h5` | Saved Keras model weights |
| `flowmodel.h5` | Saved optical flow model weights |
| `FallDataset_anno.zip` | Dataset annotations |
| `paper.pdf` | Full research paper |

---

## Stack

- Python, TensorFlow / Keras
- OpenCV (optical flow)
- Jupyter Notebooks

---

## Usage

```bash
pip install tensorflow opencv-python jupyter
jupyter notebook "fall detection.ipynb"
```

