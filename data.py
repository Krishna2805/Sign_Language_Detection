import os
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Folder where class subfolders will be stored
dataset_path = os.getenv("DATASET_PATH", "landmark_dataset")
os.makedirs(dataset_path, exist_ok=True)

# Class labels to collect
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "fullstop"]
current_label = "fullstop"

camera_index = int(os.getenv("CAMERA_INDEX", "0"))
cap = cv2.VideoCapture(camera_index)

print("Press 'a', 'b', 'c' etc. to switch class | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Save landmark vector to respective class folder
            class_folder = os.path.join(dataset_path, current_label)
            os.makedirs(class_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            np.save(os.path.join(class_folder, f"{timestamp}.npy"), np.array(landmarks))

            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Collecting Landmarks", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Exit condition
        break
    elif key != -1:  # If a valid key is pressed
        try:
            key_char = chr(key)  # Convert the key code to a character
            if key_char.upper() in [l.upper() for l in labels]:  # Check if it's in the labels list
                current_label = key_char.upper()
        except ValueError:
            pass  # Ignore non-printable key values

cap.release()
cv2.destroyAllWindows()
