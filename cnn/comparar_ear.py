import os
import sys
import cv2
import numpy as np

# config.py está en la carpeta de arriba (raíz del proyecto)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CNN_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(CNN_DIR, "dataset")
ONNX = os.path.join(CNN_DIR, "ojos.onnx")


def matriz(real, pred, titulo):
    vn = int(((real == 0) & (pred == 0)).sum())   # abierto -> dijo abierto
    fp = int(((real == 0) & (pred == 1)).sum())   # abierto -> dijo cerrado (falsa alarma)
    fn = int(((real == 1) & (pred == 0)).sum())   # cerrado -> dijo abierto (¡dormido sin alarma!)
    vp = int(((real == 1) & (pred == 1)).sum())   # cerrado -> dijo cerrado
    recall = vp / (vp + fn) if vp + fn else float("nan")
    precision = vp / (vp + fp) if vp + fp else float("nan")
    print(f"\n=== {titulo}: {len(real)} fotos ===")
    print("                   dijo ABIERTO   dijo CERRADO")
    print(f"  era ABIERTO      {vn:>10}   {fp:>12}")
    print(f"  era CERRADO      {fn:>10}   {vp:>12}")
    print(f"  Accuracy:            {(vp + vn) / len(real):.3f}")
    print(f"  Recall cerrados:     {recall:.3f}")
    print(f"  Precision cerrados:  {precision:.3f}")


def main():
    sesion = sys.argv[1] if len(sys.argv) > 1 else "prueba"
    carpeta = os.path.join(DIR, sesion)
    csv = os.path.join(carpeta, "ear.csv")
    if not os.path.exists(csv):
        print("No existe", csv, "- captura con capturar_ojos.py (la versión que guarda el EAR)")
        return

    # 1. Leer las fotos y su EAR (cada línea del csv: abiertos/12.png,0.2873)
    X, real, ears = [], [], []
    for linea in open(csv, encoding="utf-8"):
        archivo, valor = linea.strip().split(",")
        img = cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE)
        if img is None:                         # foto borrada o movida: se saltea
            continue
        X.append(img.astype("float32") / 255.0)
        real.append(1 if archivo.startswith("cerrados") else 0)
        ears.append(float(valor))
    X = np.array(X).reshape(-1, 24, 24, 1)
    real, ears = np.array(real), np.array(ears)
    print(f"Sesión {sesion}: {len(real)} fotos (abiertos={(real == 0).sum()}, cerrados={(real == 1).sum()})")

    # 2. CNN: misma regla que sistema.py (probabilidad > 0.5 = cerrado)
    net = cv2.dnn.readNetFromONNX(ONNX)
    net.setInput(X)
    pred_cnn = (net.forward().ravel() > 0.5).astype(int)
    matriz(real, pred_cnn, "CNN")

    # 3. EAR: misma regla que sistema.py (EAR < EAR_UMBRAL = cerrado)
    pred_ear = (ears < config.EAR_UMBRAL).astype(int)
    matriz(real, pred_ear, f"EAR (umbral {config.EAR_UMBRAL})")

    # 4. ¿Cuánto se separan los EAR de cada clase?
    print(f"\nEAR abiertos: mín {ears[real == 0].min():.3f}  mediana {np.median(ears[real == 0]):.3f}")
    print(f"EAR cerrados: mediana {np.median(ears[real == 1]):.3f}  máx {ears[real == 1].max():.3f}")


if __name__ == "__main__":
    main()
