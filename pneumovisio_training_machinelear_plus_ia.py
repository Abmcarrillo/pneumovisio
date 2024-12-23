import cv2
import numpy as np
import joblib
import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def procesar_imagen(ruta, size=(128, 128)):
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError("Error en la imagen.")
    img = cv2.resize(img, size)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog.compute(gris).flatten()

def cargar_imagenes(ruta_normal, ruta_neumonia):
    normal_features = []
    pneumonia_features = []
    labels = []
    for img in os.listdir(ruta_normal):
        if img.endswith(".jpeg"):
            normal_features.append(procesar_imagen(os.path.join(ruta_normal, img)))
            labels.append(0)
    for img in os.listdir(ruta_neumonia):
        if img.endswith(".jpeg"):
            pneumonia_features.append(procesar_imagen(os.path.join(ruta_neumonia, img)))
            labels.append(1)
    return np.array(normal_features + pneumonia_features), np.array(labels)

def entrenar_modelo(ruta_normal, ruta_neumonia):
    print("Cargando y procesando imágenes...")
    X, y = cargar_imagenes(ruta_normal, ruta_neumonia)
    print("Normalizando características...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Entrenando modelo...")
    start_time = time.time()
    clasificador = SVC(kernel='linear', random_state=42)
    clasificador.fit(X_train, y_train)
    print("Modelo entrenado en %s segundos." % (time.time() - start_time))
    print("Evaluando modelo...")
    y_pred = clasificador.predict(X_test)
    print("Métricas de evaluación:")
    print(f"Precisión: {metrics.accuracy_score(y_test, y_pred):.2f}")
    print(f"Precisión (Precision): {metrics.precision_score(y_test, y_pred):.2f}")
    print(f"Exhaustividad (Recall): {metrics.recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {metrics.f1_score(y_test, y_pred):.2f}")
    print(f"Matriz de confusión:\n{metrics.confusion_matrix(y_test, y_pred)}")
    print("Guardando modelo entrenado...")
    joblib.dump(clasificador, 'svm_pneumonia_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

ruta_normal = r"C:\your\route\to\datasheet\chest_xray\train\NORMAL"
ruta_neumonia = r"C:\your\route\to\datasheet\\chest_xray\train\PNEUMONIA"
entrenar_modelo(ruta_normal, ruta_neumonia)
