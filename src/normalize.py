import numpy as np
import csv

def normalize_landmarks(landmarks):
    origin = landmarks[0]
    landmarks = landmarks - origin
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks

input_file = 'data/hand_gestures.csv'
output_file = 'data/hand_gestures_normalized.csv'

with open(input_file, newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
    reader = csv.reader(csvfile_in)
    writer = csv.writer(csvfile_out)

    for row in reader:
        label = row[0]  # etiqueta
        features = list(map(float, row[1:]))

        # Sólo tomamos los primeros 63 floats (21 puntos * 3 coords)
        landmarks = np.array(features[:63]).reshape(21),
        # Normalizamos
        landmarks_norm = normalize_landmarks(landmarks)

        # Aplanamos para guardar en fila
        landmarks_norm_flat = landmarks_norm.flatten()

        # Añadimos la etiqueta delante
        row_out = [label] + list(landmarks_norm_flat)

        writer.writerow(row_out)

print(f"Archivo normalizado creado: {output_file}")
