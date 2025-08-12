import cv2
import numpy as np
import mediapipe as mp

# Configuración
MAX_HANDS = 1 

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Utilidades
def normalize_landmarks(landmarks):
    """
    landmarks: np.array shape (21,3)
    centra en landmark 0 y escala por la distancia máxima
    """
    origin = landmarks[0]
    landmarks = landmarks - origin
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks

def extraer_landmarks(frame):
    """
    Devuelve:
      - array (21,3) si detecta 1 mano
      - None si no se detecta mano
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])  # (21,3)
    return lm

def detectar_dedos(frame, model, scaler):
    """
    Extrae landmarks, normaliza, aplana y escala con el scaler cargado,
    llama al modelo y devuelve la clase (nº de dedos) o None si no hay mano.
    """
    lm = extraer_landmarks(frame)
    if lm is None:
        return None

    # Normalizar igual que en entrenamiento
    lm_norm = normalize_landmarks(lm)

    # Aplanar a (1,63)
    features = lm_norm.flatten().reshape(1, -1)

    # Transformar con scaler
    features_scaled = scaler.transform(features)

    # Predicción
    probs = model.predict(features_scaled, verbose=0)
    clase = int(np.argmax(probs, axis=1)[0])
    return clase

def op_from_code(code):
    return {1: '+', 2: '-', 3: '/', 4: '*'}.get(code, '?')