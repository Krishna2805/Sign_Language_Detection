import os
import cv2
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()

model_path = os.getenv("MODEL_PATH", "landmark_model/landmark_model.h5")
labels_path = os.getenv("LABELS_PATH", "landmark_model/labels.txt")
camera_index = int(os.getenv("CAMERA_INDEX", "0"))

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Model or labels file missing. Run train.py first.")

# Load trained model and class labels
model = load_model(model_path)
with open(labels_path, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(camera_index)

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmark vector
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:  # 21 points * 3 (x, y, z)
                prediction = model.predict(np.array([landmarks]), verbose=0)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                label = f"{class_names[class_id]} ({confidence:.2f})"

                # Show label on screen
                cv2.putText(frame, label, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
