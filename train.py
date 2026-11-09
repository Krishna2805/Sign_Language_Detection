import os
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

load_dotenv()

# Dataset directory (with subfolders per class)
dataset_path = os.getenv("DATASET_PATH", "landmark_dataset")
model_path = os.getenv("MODEL_PATH", "landmark_model/landmark_model.h5")
labels_path = os.getenv("LABELS_PATH", "landmark_model/labels.txt")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' not found. Run data.py to collect samples first.")

# Load data
X, y = [], []
class_names = sorted(os.listdir(dataset_path))
class_indices = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue
    for file in os.listdir(class_folder):
        if file.endswith(".npy"):
            path = os.path.join(class_folder, file)
            vector = np.load(path)
            X.append(vector)
            y.append(class_indices[class_name])

if len(X) == 0:
    raise ValueError(f"No .npy landmark files found in '{dataset_path}'. Collect training data first.")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(class_names))

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Build model
model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

# Save model and class labels
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(os.path.dirname(labels_path), exist_ok=True)

model.save(model_path)
with open(labels_path, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"Model saved to {model_path} and labels to {labels_path}.")
