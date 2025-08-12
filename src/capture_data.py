import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Configuración
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "hand_gestures.csv")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Abrir cámara
cap = cv2.VideoCapture(0)

print("[INFO] Pulsa una tecla numérica (0-9) para etiquetar el gesto actual.")
print("[INFO] Pulsa 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear horizontalmente para que sea tipo espejo
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture Gestures", frame)

    key = cv2.waitKey(1) & 0xFF

    # Guardar datos
    if key in range(ord('0'), ord('9') + 1):
        label = chr(key)
        if result.multi_hand_landmarks:
            row = [label]  # primera columna: etiqueta
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])  # coordenadas normalizadas

            # Rellenar con ceros si solo hay una mano
            num_features = 21 * 3 * 2 
            while len(row) < num_features + 1:
                row.extend([0.0, 0.0, 0.0])

            # Guardar en CSV
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"[OK] Gesto '{label}' guardado.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
