# interaction.py
import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# --- Config ---
MODEL_PATH = 'models/neuronal_network.h5' 
SCALER_PATH = 'scaler.save'
MAX_HANDS = 1 

# --- Cargar modelo y scaler ---
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Inicializar MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# --- Utilidades ---
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

    # Aplanar a (1,63)Q
    features = lm_norm.flatten().reshape(1, -1)

    # Transformar con scaler
    features_scaled = scaler.transform(features)

    # Predicción
    probs = model.predict(features_scaled, verbose=0)
    clase = int(np.argmax(probs, axis=1)[0])
    return clase

def op_from_code(code):
    return {1: '+', 2: '-', 3: '/', 4: '*'}.get(code, '?')

# --- Main loop ---
cap = cv2.VideoCapture(0)
fase = 1
n1 = n2 = op = None
last_action_time = 0.0
DEBOUNCE_SECONDS = 0.5  # evita dobles triggers al pulsar enter

print("Iniciando. 'q' para salir. Pulsa ENTER para confirmar cada fase.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Leemos tecla una sola vez por bucle
    key = cv2.waitKey(1) & 0xFF

    # Detectar dedos (clase) con modelo
    dedos = detectar_dedos(frame, model, scaler)  # int o None

    # Texto de ayuda:
    if dedos is None:
        estado_txt = "No se detecta mano"
    else:
        estado_txt = f"Detectado: {dedos}"

    # Lógica por fases
    if fase == 1:
        cv2.putText(frame, "Fase 1 - Dime primer numero y ENTER para confirmar", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Numero 1: {dedos if dedos is not None else '-'}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 2)
        if key in (13,10) and (time.time() - last_action_time) > DEBOUNCE_SECONDS:
            if dedos is not None:
                n1 = dedos
                fase = 2
                last_action_time = time.time()

    elif fase == 2:
        cv2.putText(frame, "Fase 2 - Selecciona operacion y ENTER para confirmar", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, "1:+  2:-  3:/  4:*", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.putText(frame, f"Operacion seleccionada: {op_from_code(dedos) if dedos is not None else '-'}", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,200), 2)
        if key in (13,10) and (time.time() - last_action_time) > DEBOUNCE_SECONDS:
            if dedos in (1,2,3,4):
                op = dedos
                fase = 3
                last_action_time = time.time()

    elif fase == 3:
        cv2.putText(frame, "Fase 3 - Dime segundo numero y ENTER para confirmar", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
        cv2.putText(frame, f"Numero 2: {dedos if dedos is not None else '-'}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,150,0), 2)
        if key in (13,10) and (time.time() - last_action_time) > DEBOUNCE_SECONDS:
            if dedos is not None:
                n2 = dedos
                fase = 4
                last_action_time = time.time()

    elif fase == 4:
        # Calcular resultado
        resultado = "Error"
        try:
            if op == 1:
                resultado = n1 + n2
            elif op == 2:
                resultado = n1 - n2
            elif op == 3:
                resultado = n1 / n2 if n2 != 0 else "Div0"
            elif op == 4:
                resultado = n1 * n2
        except Exception as e:
            resultado = f"Err:{e}"

        cv2.putText(frame, f"Resultado: {resultado}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.putText(frame, f"{n1} {op_from_code(op)} {n2}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,200), 2)

        # ENTER para reiniciar a fase 1
        if key in (13,10) and (time.time() - last_action_time) > DEBOUNCE_SECONDS:
            fase = 1
            n1 = n2 = op = None
            last_action_time = time.time()

    # Mostrar estado en pantalla
    cv2.putText(frame, estado_txt, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imshow("Calculadora gestos", frame)

    # Salir con 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
