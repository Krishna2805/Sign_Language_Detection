"""
ASL Gesture Recognition Inference & Evaluation
==============================================
Supports real-time webcam inference as well as headless --mock and --file modes.
"""

import os
import sys
import argparse
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()


def resolve_paths(model_arg=None, labels_arg=None):
    if model_arg and os.path.exists(model_arg):
        model_path = model_arg
    elif os.path.exists("landmark_model.keras"):
        model_path = "landmark_model.keras"
    elif os.path.exists(os.getenv("MODEL_PATH", "landmark_model/landmark_model.h5")):
        model_path = os.getenv("MODEL_PATH", "landmark_model/landmark_model.h5")
    else:
        raise FileNotFoundError("Model file missing. Run train.py or generate_synthetic_data.py first.")

    if labels_arg and os.path.exists(labels_arg):
        labels_path = labels_arg
    elif os.path.exists("labels.txt"):
        labels_path = "labels.txt"
    elif os.path.exists(os.getenv("LABELS_PATH", "landmark_model/labels.txt")):
        labels_path = os.getenv("LABELS_PATH", "landmark_model/labels.txt")
    else:
        raise FileNotFoundError("Labels file missing. Run train.py first.")

    return model_path, labels_path


def load_model_and_labels(model_path, labels_path):
    model = load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return model, class_names


def run_file_mode(file_path, model, class_names):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    if os.path.isdir(file_path):
        files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".npy")]
        print(f"Evaluating {len(files)} landmark vectors in directory '{file_path}'...")
        for p in files[:10]:
            vec = np.load(p)
            pred = model.predict(np.array([vec]), verbose=0)
            class_id = int(np.argmax(pred))
            conf = float(pred[0][class_id])
            print(f"  {os.path.basename(p)} -> Predicted: {class_names[class_id]} (Confidence: {conf*100:.1f}%)")
        return

    vector = np.load(file_path)
    if vector.shape != (63,) and vector.size != 63:
        print(f"Error: Vector must have 63 float values (21 points x 3 coordinates). Got {vector.shape}.")
        sys.exit(1)

    prediction = model.predict(np.array([vector.flatten()]), verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[0][class_id])
    label = class_names[class_id]
    print(f"\n[FILE INFERENCE]")
    print(f"  Input File  : {file_path}")
    print(f"  Prediction  : {label}")
    print(f"  Confidence  : {confidence*100:.2f}%\n")


def run_mock_mode(model, class_names, dataset_dir="sample_dataset"):
    print("\n=======================================================")
    print("      HEADLESS ASL GESTURE RECOGNITION EVALUATION      ")
    print("=======================================================")

    if not os.path.exists(dataset_dir):
        # Auto-generate if missing
        print(f"'{dataset_dir}' not found. Generating canonical samples...")
        from generate_synthetic_data import generate_dataset
        generate_dataset(output_dir=dataset_dir, samples_per_class=10)

    correct = 0
    total = 0
    evaluated_classes = []

    for class_name in class_names:
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        sample_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(".npy")]
        if not sample_files:
            continue

        evaluated_classes.append(class_name)
        # Evaluate up to 5 samples per class
        for s_path in sample_files[:5]:
            vector = np.load(s_path)
            prediction = model.predict(np.array([vector]), verbose=0)
            pred_id = int(np.argmax(prediction))
            pred_label = class_names[pred_id]
            conf = float(prediction[0][pred_id])

            is_match = (pred_label == class_name)
            if is_match:
                correct += 1
            total += 1

            status = "OK" if is_match else "FAIL"
            print(f"  [{status:^4}] Expected: {class_name:<8} | Predicted: {pred_label:<8} | Confidence: {conf*100:5.1f}%")

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print("-------------------------------------------------------")
    print(f"Total Evaluated : {total} samples across {len(evaluated_classes)} classes")
    print(f"Accuracy        : {accuracy:.2f}% ({correct}/{total})")
    print("=======================================================\n")


def run_camera_mode(model, class_names, camera_index=0):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        print(f"Camera mode requires OpenCV and MediaPipe: {e}")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[NOTICE] Physical camera at index {camera_index} could not be opened or is unavailable.")
        print("Tip: Run in headless mode using: python test.py --mock")
        print(" Or evaluate a landmark file using: python test.py --file <path_to_landmark.npy>\n")
        return

    print("Webcam started. Press 'q' to exit.")
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture video frame. Exiting.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    prediction = model.predict(np.array([landmarks]), verbose=0)
                    class_id = int(np.argmax(prediction))
                    confidence = float(prediction[0][class_id])
                    label = f"{class_names[class_id]} ({confidence:.2f})"
                    cv2.putText(frame, label, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Sign Language Gesture Recognition Inference")
    parser.add_argument("--mock", action="store_true", help="Run headless evaluation using synthetic landmark dataset")
    parser.add_argument("--file", type=str, default=None, help="Path to .npy landmark file or folder for headless evaluation")
    parser.add_argument("--model", type=str, default=None, help="Custom path to trained model (.keras or .h5)")
    parser.add_argument("--labels", type=str, default=None, help="Custom path to labels.txt")
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="Camera device index")

    args = parser.parse_args()
    model_path, labels_path = resolve_paths(args.model, args.labels)
    model, class_names = load_model_and_labels(model_path, labels_path)

    if args.mock:
        run_mock_mode(model, class_names)
    elif args.file:
        run_file_mode(args.file, model, class_names)
    else:
        run_camera_mode(model, class_names, camera_index=args.camera)


if __name__ == "__main__":
    main()
