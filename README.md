# Fall Detection for IoT Small Sensors

*Two CNN+RNN detection frameworks for elderly fall detection deployable on sub-$9 ESP32-cam devices.*

**Ho Ko · Yueqi Liu · Jiaxin Xie** — ECE, University of Waterloo

📄 **[Read the Paper](https://github.com/woodyhoko/Fall_Detection/blob/main/paper.pdf)** | **[▶ Pipeline Demo](demo.html)**

---

## 1. Clinical and economic motivation

Falls are the leading cause of injury-related death among adults over 65 and the most common cause of traumatic brain injury in this population (WHO 2021). In Canada, fall-related hospitalizations cost the healthcare system approximately $2.8 billion annually. Rapid detection and alerting — ideally within 60 seconds — dramatically improves outcomes by reducing the duration of the "long lie" (time spent on the floor before assistance arrives).

Existing fall detection solutions fail on one or more axes:

| Solution type | Cost | Privacy | Deployment complexity |
|---|---|---|---|
| Wearable (accelerometer/gyro) | Medium | Good | User compliance required |
| Depth camera (Kinect) | High | Poor (3D body scan) | Complex, single room |
| Pressure mat | Medium | Good | Fixed location, trip hazard |
| Standard IP camera + cloud AI | Low-Medium | Poor (raw video to cloud) | Requires broadband |
| **ESP32-cam + on-device ML (ours)** | **< $9** | **Good (no raw video leaves)** | **Plug-and-play** |

---

## 2. Two IoT architectures

### 2.1 Distributed analysis

Each ESP32-cam sensor runs the full ML pipeline locally. Only an alert signal (not video) is sent to the cloud.

```
[ESP32-cam #1]  → CNN features → RNN classify → alert if fall → MQTT → Cloud
[ESP32-cam #2]  → CNN features → RNN classify → alert if fall → MQTT → Cloud
```

**Advantages:** no single point of failure; each sensor is self-contained; privacy-preserving by design (raw frames never leave the device).

**Disadvantage:** the full model must fit within the ESP32's 520 KB SRAM + 4 MB PSRAM; aggressive quantization is required.

### 2.2 Centralized analysis

Sensors perform only feature extraction; a local Raspberry Pi 4 runs the RNN classifier.

```
[ESP32-cam #1]  → CNN features (float32 vector) ──┐
[ESP32-cam #2]  → CNN features ──────────────────── Raspberry Pi 4 → RNN → Cloud
[ESP32-cam #N]  → CNN features ──────────────────┘
```

**Advantages:** heavier RNN model possible; multi-camera temporal fusion (correlate falls across cameras); easier model updates (update only the Pi).

**Disadvantage:** requires local network; Raspberry Pi is a centralized dependency.

---

## 3. Detection models

### 3.1 Stage 1 — CNN feature extraction

A lightweight **MobileNet-style depthwise separable CNN** processes each video frame:

```
Frame (128×128 grayscale) → Conv → DepthwiseSep × 3 → GlobalAvgPool → Feature vector (64-d)
```

Depthwise separable convolutions reduce parameter count and FLOPs by ~8–9× compared to standard convolutions, making them suitable for ESP32's limited compute.

The CNN is trained to discriminate *fall-indicative spatial patterns* (person horizontal, high-speed limb motion) from normal patterns (vertical posture, slow movement). Features are invariant to background and lighting by design.

### 3.2 Stage 2 — RNN temporal classification

A sliding window of *T* = 30 consecutive CNN feature vectors (≈ 1 second at 30 fps) is fed to a **GRU (Gated Recurrent Unit)** classifier:

```
[f_{t-29}, ..., f_t] → GRU (hidden=64) → Dense → sigmoid → P(fall)

GRU update equations:
    z_t = σ(W_z [h_{t-1}, x_t])          (update gate)
    r_t = σ(W_r [h_{t-1}, x_t])          (reset gate)
    h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t]) (candidate)
    h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

The GRU captures the **temporal signature of a fall** — a characteristic rapid transition from vertical to horizontal posture followed by stillness — that a single-frame CNN cannot see. This temporal modeling is critical for distinguishing falls from intentional floor activities (stretching, yoga), which look visually similar in a single frame.

---

## 4. Dataset

**IMVIA Fall Detection Dataset** — publicly available benchmark of annotated video sequences:

- Fall events (forward fall, backward fall, lateral fall, fall from chair)
- Activities of daily living (walking, sitting, standing, lying for rest)
- Recorded in a controlled lab environment; multiple camera angles

**Key challenge:** distinguishing an emergency fall from intentional lying-down (resting, yoga). Many prior systems have high false-positive rates on this case. The temporal RNN substantially reduces false positives by observing the *dynamics* of the transition, not just the terminal posture.

---

## 5. Results

Model 2 (centralized, with full-size RNN on Raspberry Pi) achieves higher sensitivity and specificity than Model 1 (distributed, quantized RNN on ESP32) — particularly on the hard floor-rest case. Full quantitative results (accuracy, sensitivity, specificity, F1, latency) are in the paper (Section 4).

---

## 6. Repository contents

| File | Description |
|---|---|
| `dataset_generation.ipynb` | Frame extraction, annotation parsing, train/val/test split |
| `fall detection.ipynb` | Model training, evaluation, confusion matrix, result charts |
| `fall_model.h5` | Saved Keras weights — main fall detection model |
| `flowmodel.h5` | Saved Keras weights — optical flow auxiliary model |
| `FallDataset_anno.zip` | IMVIA dataset annotations |
| `paper.pdf` | Full paper (9 pages, IEEE format) |
| `demo.html` | Interactive browser visualization of the detection pipeline |

---

## 7. Stack

Python · TensorFlow / Keras · OpenCV · Jupyter Notebooks

---

## 8. Run

```bash
pip install tensorflow opencv-python jupyter numpy
# Step 1: generate dataset
jupyter notebook dataset_generation.ipynb
# Step 2: train and evaluate
jupyter notebook "fall detection.ipynb"
```

Pre-trained weights (`fall_model.h5`, `flowmodel.h5`) are included — skip Step 1 to evaluate directly.

---

## 9. References

1. WHO. *Falls — Key Facts.* World Health Organization, 2021.
2. A. G. Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv:1704.04861*, 2017.
3. K. Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP 2014*. (GRU architecture)
4. A. Rougier et al. "Robust Video Surveillance for Fall Detection Based on Human Shape Deformation." *IEEE TCSVT*, 21(5):611–623, 2011.
