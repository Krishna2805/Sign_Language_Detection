import os
from collections import deque, Counter
import cv2
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt

load_dotenv()

# Configuration from .env
broker_address = os.getenv("MQTT_BROKER", "localhost")
broker_port = int(os.getenv("MQTT_PORT", "1883"))
topic = os.getenv("MQTT_TOPIC", "sign_language/sentence")
model_path = os.getenv("MODEL_PATH", "landmark_model/landmark_model.h5")
labels_path = os.getenv("LABELS_PATH", "landmark_model/labels.txt")
camera_index = int(os.getenv("CAMERA_INDEX", "0"))

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Model or labels file missing. Run train.py first.")

# MQTT setup
mqtt_connected = False
client = mqtt.Client()

def on_connect(c, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        print(f"Connected to MQTT broker at {broker_address}:{broker_port}")
    else:
        print(f"MQTT connection failed with code {rc}")

client.on_connect = on_connect

try:
    client.connect(broker_address, broker_port, 60)
    client.loop_start()
except Exception as e:
    print(f"Warning: Could not connect to MQTT broker ({broker_address}:{broker_port}): {e}. Inference will continue without publishing.")

# Load model and labels
model = load_model(model_path)
with open(labels_path, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(camera_index)

# Variables for sentence forming
current_sentence = ""
final_sentence = ""
buffer = deque(maxlen=15)
min_consistent_count = 10
last_confirmed = ""
hand_visible = False
hand_missing_frames = 0
max_missing_frames = 15

confirmed_letter_timer = 0
confirm_delay_frames = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_visible = True
        hand_missing_frames = 0

        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                class_id = np.argmax(prediction)
                predicted_label = class_names[class_id]
                buffer.append(predicted_label)

                most_common, count = Counter(buffer).most_common(1)[0]
                if count >= min_consistent_count:
                    if confirmed_letter_timer == 0 or most_common != last_confirmed:
                        if most_common.lower() == "space":
                            current_sentence += " "
                        elif most_common.lower() in ["stop", "fullstop"]:
                            current_sentence += "."
                        else:
                            current_sentence += most_common.upper()

                        last_confirmed = most_common
                        confirmed_letter_timer = confirm_delay_frames

    else:
        if hand_visible:
            hand_missing_frames += 1
            if hand_missing_frames >= max_missing_frames:
                final_sentence = current_sentence
                print("Final Sentence:", final_sentence)
                if final_sentence.strip() != "" and mqtt_connected:
                    client.publish(topic, final_sentence)
                current_sentence = ""
                last_confirmed = ""
                buffer.clear()
                hand_visible = False
                hand_missing_frames = 0

    if confirmed_letter_timer > 0:
        confirmed_letter_timer -= 1

    cv2.putText(frame, f"Current: {current_sentence}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if final_sentence:
        cv2.putText(frame, f"Final: {final_sentence}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
try:
    client.loop_stop()
    client.disconnect()
except Exception:
    pass
