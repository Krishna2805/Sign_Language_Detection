# ASL Landmark Recognition (MediaPipe + TensorFlow)

Real-time American Sign Language (ASL) finger-spelling recognition prototype using MediaPipe hand landmarks and a TensorFlow/Keras neural network classifier, with sentence aggregation and MQTT messaging support.

---

## 📌 Architecture & Data Flow

```
Webcam Frame
    │
    ▼
MediaPipe Hands (21 Landmark Extraction: 63 coordinates)
    │
    ├──> data.py: Records .npy landmark vectors by class
    │
    ├──> test.py: Real-time single-letter inference
    │
    └──> test_sentence.py: Frame-smoothing buffer + Sentence builder
            │
            ▼
    MQTT Broker (sign_language/sentence)
            │
            ▼
    speak.py: Subscribes and outputs synthesized speech via TTS
```

---

## 📂 Project Structure

```
Sign_Language_Detection/
├── data.py              # Capture hand landmarks from webcam into .npy files
├── train.py             # Train multi-layer perceptron (MLP) on landmark dataset
├── test.py              # Real-time single-letter prediction overlay
├── test_sentence.py     # Sentence construction with temporal smoothing & MQTT publisher
├── speak.py             # MQTT subscriber that outputs received sentences via TTS
├── requirements.txt     # Python dependency specifications
├── .env.example         # Environment variable template
├── .gitignore           # Ignored binaries, models, datasets, and local configs
└── README.md            # Project documentation
```

---

## 🛠️ Prerequisites & Setup

### 1. Environment Setup

Python 3.8+ is supported. Create and activate a virtual environment:

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Adjust variables if needed:
- `MQTT_BROKER`: IP address or hostname of your MQTT broker (default: `localhost`).
- `MQTT_PORT`: Broker port (default: `1883`).
- `CAMERA_INDEX`: Camera device index (default: `0`).
- `DATASET_PATH`: Folder for captured landmarks (default: `landmark_dataset`).
- `MODEL_PATH`: Saved model path (default: `landmark_model/landmark_model.h5`).

---

## 🚀 Usage

### Step 1: Collect Landmark Data
```bash
python data.py
```
- Position your hand in the webcam view.
- Press letter keys (`a`–`z`, `space`, `fullstop`) to switch the active class label.
- Press `q` to quit data collection.

### Step 2: Train Classifier
```bash
python train.py
```
- Trains a dense neural network (128 → Dropout → 64 → Softmax) on the collected `.npy` samples.
- Saves the trained model to `landmark_model/landmark_model.h5` and class names to `landmark_model/labels.txt`.

### Step 3: Run Inference

**Single-Letter Display**:
```bash
python test.py
```

**Sentence Construction + MQTT Publishing**:
```bash
python test_sentence.py
```

**TTS Speaker Listener (Optional)**:
```bash
python speak.py
```

---

## ⚙️ Configuration Reference

| Key | Default | Description |
|---|---|---|
| `MQTT_BROKER` | `localhost` | MQTT broker host IP or domain |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `sign_language/sentence` | MQTT topic for completed sentences |
| `CAMERA_INDEX` | `0` | OpenCV VideoCapture camera index |
| `DATASET_PATH` | `landmark_dataset` | Directory for raw .npy vector storage |
| `MODEL_PATH` | `landmark_model/landmark_model.h5` | Path for model weights |
| `LABELS_PATH` | `landmark_model/labels.txt` | Path for ordered label index |
| `DEFAULT_LANGUAGE` | `en` | TTS synthesis language code |
