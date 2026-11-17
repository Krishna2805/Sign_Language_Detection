# ASL Landmark Recognition — Architectural Context & Technical Specification

## 1. System Overview
The **ASL Landmark Recognition System** is a lightweight, high-throughput gesture classification pipeline that translates 3D hand pose kinematics into American Sign Language (ASL) finger-spelling representations. The pipeline operates on 21 spatial hand landmark nodes extracted via Google MediaPipe Hands, yielding a 63-dimensional coordinate feature space $[x_0, y_0, z_0, \dots, x_{20}, y_{20}, z_{20}]$.

The downstream neural network is an ultra-fast Multi-Layer Perceptron (MLP) containing approximately 18,200 trainable parameters that trains in under 5 seconds on CPU and evaluates with under 1 ms inference latency per frame.

---

## 2. MediaPipe Hand Landmark Space

Each hand gesture is parameterized as 21 three-dimensional coordinate vectors normalized to the bounding box:

| Index | Landmark Name | Anatomical Region | Coordinate Representation |
|:---:|:---|:---|:---|
| **0** | `WRIST` | Carpal Base | $[x_0, y_0, z_0]$ |
| **1-4** | `THUMB_CMC`, `THUMB_MCP`, `THUMB_IP`, `THUMB_TIP` | Radial Digit 1 | $[x_1, y_1, z_1] \dots [x_4, y_4, z_4]$ |
| **5-8** | `INDEX_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Digit 2 | $[x_5, y_5, z_5] \dots [x_8, y_8, z_8]$ |
| **9-12** | `MIDDLE_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Digit 3 | $[x_9, y_9, z_9] \dots [x_{12}, y_{12}, z_{12}]$ |
| **13-16** | `RING_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Digit 4 | $[x_{13}, y_{13}, z_{13}] \dots [x_{16}, y_{16}, z_{16}]$ |
| **17-20** | `PINKY_MCP`, `PIP`, `DIP`, `TIP` | Ulnar Digit 5 | $[x_{17}, y_{17}, z_{17}] \dots [x_{20}, y_{20}, z_{20}]$ |

Total Feature Vector Dimensionality:
$$\mathbf{x} \in \mathbb{R}^{63} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$

---

## 3. Classifier Architecture

The neural network is built with native Keras/TensorFlow and consists of:
- **Input Layer**: `Dense(63)` flat landmark vector
- **Hidden Layer 1**: `Dense(128, activation='relu')` with `Dropout(0.3)`
- **Hidden Layer 2**: `Dense(64, activation='relu')`
- **Output Layer**: `Dense(28, activation='softmax')` corresponding to:
  - 26 Alphabet Signs: `A`, `B`, `C`, ..., `Z`
  - 2 Control Signs: `space` (word boundary), `fullstop` (sentence termination)

Model Footprint:
- Weight File: `landmark_model.keras` (~97 KB)
- Total Parameters: 18,268 (all trainable)
- Execution Speed: ~0.8 ms per sample on CPU

---

## 4. Evaluation & Demonstration Modes

The inference suite (`test.py`) supports three distinct operational modes:

### Mode A: Headless Mock Benchmark (`--mock`)
Evaluates the model across canonical synthetic hand vectors without requiring a camera or graphical display:
```bash
python test.py --mock
```
Outputs per-class prediction accuracy, confidence percentages, and total classification metrics.

### Mode B: Direct File Evaluation (`--file <path>`)
Classifies a pre-recorded `.npy` landmark vector or directory of vectors:
```bash
python test.py --file sample_dataset/A/sample_000.npy
```

### Mode C: Real-Time Webcam Inference (Default)
Captures live frames via OpenCV, computes MediaPipe hand landmarks in real-time, and renders overlay predictions on screen:
```bash
python test.py
```
If a physical camera is not attached or inaccessible, it gracefully reports camera unavailability and recommends running `--mock`.

---

## 5. Synthetic Dataset & Reproducibility
The project includes `generate_synthetic_data.py` to deterministically create canonical 3D landmark configurations for all 28 classes with parameterized Gaussian noise ($\sigma = 0.008$), populating `sample_dataset/` for instant training and evaluation.
