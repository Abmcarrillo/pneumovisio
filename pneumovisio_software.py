import cv2
import numpy as np
import joblib
from tkinter import Tk, filedialog, Label, Button
from sklearn.metrics import classification_report

modelo = r"C:\route\to\modelsvm\svm_pneumonia_model.pkl"
scaler_path = r"C:\route\to\scaler\escalador.pkl"
clasificador = joblib.load(modelo)
scaler = joblib.load(scaler_path)

hog = cv2.HOGDescriptor(
    _winSize=(128, 128), 
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

precision_modelo = 0.97
f1_score_modelo = 0.98

def procesar_imagen(ruta, size=(128, 128)):
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError("Error en la imagen.")
    img = cv2.resize(img, size)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog.compute(gris).flatten()
    features = features / np.linalg.norm(features)
    return features

def predecir(ruta):
    caracteristicas = procesar_imagen(ruta).reshape(1, -1)
    caracteristicas = scaler.transform(caracteristicas)
    prediccion = clasificador.decision_function(caracteristicas)
    return "Sin Incidencias" if prediccion[0] < 3.2 else "Neumonía Detectada", prediccion[0]

def seleccionar_imagen():
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpeg;*.jpg;*.png")])
    if ruta:
        try:
            resultado, valor_prediccion = predecir(ruta)
            nombre_imagen = ruta.split("/")[-1] if "/" in ruta else ruta.split("\\")[-1]
            label_resultado.config(
                text=(
                    f"Imagen: {nombre_imagen}\n"
                    f"Resultado: {resultado}\n"
                    f"Valor de predicción: {valor_prediccion:.2f}\n"
                    f"Precisión del modelo: {precision_modelo:.2f}\n"
                    f"F1-Score del modelo: {f1_score_modelo:.2f}"
                )
            )
        except Exception as e:
            label_resultado.config(text=f"Error: {e}")

ventana = Tk()
ventana.title("Pneumovisio - made by abmcarrillo")
ventana.geometry("500x300")

label_instrucciones = Label(ventana, text="Seleccione una imagen para analizar", font=("Arial", 12))
label_instrucciones.pack(pady=10)

boton_seleccionar = Button(ventana, text="Buscar en este Ordenador...", command=seleccionar_imagen, font=("Arial", 10))
boton_seleccionar.pack(pady=10)

label_resultado = Label(ventana, text="", font=("Arial", 12), justify="left")
label_resultado.pack(pady=20)

ventana.mainloop()
