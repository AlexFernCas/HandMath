import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# MediaPipe Hands para 1 mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Cargar modelo y scaler
model = tf.keras.models.load_model('models/neuronal_network.h5')
scaler = joblib.load('scaler.save')  # Carga el scaler guardado

def normalize_landmarks(landmarks):
    origin = landmarks[0]
    landmarks = landmarks - origin
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    landmarks_norm = normalize_landmarks(landmarks).flatten().reshape(1, -1)
    input_data = scaler.transform(landmarks_norm)
    return input_data

def extract_hand_features(results):
    features = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
    # Rellenar ceros para 2 manos (126 features)
    while len(features) < 21*3*2:
        features.extend([0.0, 0.0, 0.0])
    return np.array(features).reshape(1, -1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    input_data = preprocess_frame(frame)
    
    if input_data is not None:
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        text = f'Número: {predicted_class}'
    else:
        text = "No se detecta mano"
    
    cv2.putText(frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if input_data is not None else (0, 0, 255), 2)
    
    cv2.imshow('Hand Number Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
