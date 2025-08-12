import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Para guardar el modelo

# Carga los datos
df = pd.read_csv('../data/hand_gestures_normalized.csv')

X = df.iloc[:, 1:]
y = df.iloc[:, 0]   

# Entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrena
clf.fit(X_train, y_train)

# Predicciones sobre test
y_pred = clf.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar modelo
joblib.dump(clf, 'models/random_forest.pkl')

print("Modelo entrenado y guardado en 'hand_gesture_model.pkl'")
