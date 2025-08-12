import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configuración
DATA_PATH = '../data/hand_gestures_normalized.csv'
MODEL_PATH = '../models/neuronal_network.h5'
SCALER_PATH = '../artifacts/scaler.save'

# Carga de datos
data = pd.read_csv(DATA_PATH)

# Primera columna = etiqueta, resto = características
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values.astype(int)

# Normalizar datos con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler
joblib.dump(scaler, SCALER_PATH)

# One-hot encoding de etiquetas
y_cat = to_categorical(y)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Crear modelo
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# Evaluar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy en test: {acc:.4f}')
print(f'Loss en test: {loss:.4f}')

# Guardar modelo
model.save(MODEL_PATH)
