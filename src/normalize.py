import numpy as np
import csv
from utils import normalize_landmarks

# Configuración
INPUT_FILE_PATH = '../data/hand_gestures.csv'
OUTPUT_FILE_PATH = '../data/hand_gestures_normalized.csv'

input_file = INPUT_FILE_PATH
output_file = OUTPUT_FILE_PATH

with open(input_file, newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
    reader = csv.reader(csvfile_in)
    writer = csv.writer(csvfile_out)

    for row in reader:
        label = row[0]  # etiqueta
        features = list(map(float, row[1:]))

        # 63 floats (21 puntos * 3 coords)
        landmarks = np.array(features[:63]).reshape(21, 3)
        
        # Normalizar
        landmarks_norm = normalize_landmarks(landmarks)

        # Aplanar (1 dimensión)
        landmarks_norm_flat = landmarks_norm.flatten()

        # Añadir etiqueta delante
        row_out = [label] + list(landmarks_norm_flat)

        writer.writerow(row_out)

print(f"Archivo normalizado creado: {output_file}")
