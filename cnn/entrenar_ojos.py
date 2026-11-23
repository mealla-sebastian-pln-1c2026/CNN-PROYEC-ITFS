import os
import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tf2onnx
import sys

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ojos.onnx")


def cargar(sesiones):
    X, Y = [], []
    for s in sesiones:
        for label, t in enumerate(("abiertos", "cerrados")):
            d = f"{DIR}/{s}/{t}"
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                img = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                X.append(cv2.resize(img, (24, 24)))
                Y.append(label)
    X = np.array(X, dtype="float32").reshape(-1, 24, 24, 1) / 255.0
    Y = np.array(Y, dtype="float32")
    return X, Y

def evaluar(m, X, Y, titulo="Validación"):
    p = m.predict(X, verbose=0).ravel()        # probabilidad de "cerrado" de cada foto
    pred = (p > 0.5).astype(int)               # 1 = cerrado, 0 = abierto
    real = Y.astype(int)

    vn = int(((real == 0) & (pred == 0)).sum())   # abierto  -> dijo abierto  (bien)
    fp = int(((real == 0) & (pred == 1)).sum())   # abierto  -> dijo cerrado  (falsa alarma)
    fn = int(((real == 1) & (pred == 0)).sum())   # cerrado  -> dijo abierto  (¡dormido sin alarma!)
    vp = int(((real == 1) & (pred == 1)).sum())   # cerrado  -> dijo cerrado  (bien)

    accuracy = (vp + vn) / len(real)
    recall = vp / (vp + fn) if vp + fn else float("nan")
    precision = vp / (vp + fp) if vp + fp else float("nan")

    print(f"\n=== {titulo}: {len(real)} fotos ===")
    print("                   dijo ABIERTO   dijo CERRADO")
    print(f"  era ABIERTO      {vn:>10}   {fp:>12}")
    print(f"  era CERRADO      {fn:>10}   {vp:>12}")
    print(f"  Accuracy:            {accuracy:.3f}")
    print(f"  Recall cerrados:     {recall:.3f}   <- de los ojos cerrados, cuántos detectó")
    print(f"  Precision cerrados:  {precision:.3f}   <- de las alarmas, cuántas eran reales")


def main():
    args = sys.argv[1:]
    val_sesion = None                           # sesión que se usa SOLO para validar
    if "--val" in args:
        i = args.index("--val")
        if i + 1 >= len(args):
            print("Falta el nombre de la sesión después de --val (ejemplo: --val s3)")
            return
        val_sesion = args[i + 1]
        args = args[:i] + args[i + 2:]          # sacar "--val s3" de la lista
    if args:
        sesiones = args                         # las que escribiste: s1 s2
    elif os.path.isdir(DIR):
        sesiones = sorted(os.listdir(DIR))      # si no escribiste nada: todas
    else:
        print("No existe", DIR, "- captura primero con capturar_ojos.py")
        return
    if val_sesion in sesiones:
        sesiones.remove(val_sesion)             # nunca entrenar con la de validación
    print("Entrenar con:", sesiones, "  Validar con:", val_sesion or "20 % al azar")
    X, Y = cargar(sesiones)
    print(f"Muestras: {len(X)}  (abiertos={(Y == 0).sum()}, cerrados={(Y == 1).sum()})")
    if len(X) < 20:
        print("Pocas muestras. Captura mas con capturar_ojos.py")
        return
    orden = np.random.permutation(len(X))   # orden al azar
    X, Y = X[orden], Y[orden]
    # 1. Separar validación
    if val_sesion:
        X_tr, Y_tr = X, Y                       # todo lo cargado es para entrenar
        X_val, Y_val = cargar([val_sesion])     # la validación es una sesión que nunca vio
        if len(X_val) == 0:
            print("La sesión", val_sesion, "no tiene fotos")
            return
    else:
        n_val = int(len(X) * 0.2)
        X_val, Y_val = X[:n_val], Y[:n_val]
        X_tr, Y_tr = X[n_val:], Y[n_val:] 
    # 2. Las transformaciones al azar (solo para entrenar)
    aumento = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),                  # espejado
        keras.layers.RandomRotation(0.03),                      # giro de ±11°
        keras.layers.RandomBrightness(0.2, value_range=(0, 1)), # ±20 % de luz
        keras.layers.RandomContrast(0.2),                       # ±20 % de contraste
    ])

    # 3. Armar los lotes de entrenamiento con augmentation
    entreno = (tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
               .shuffle(len(X_tr))
               .batch(32)
               .map(lambda x, y: (tf.clip_by_value(aumento(x, training=True), 0, 1), y)))

    # 4. El modelo: igual que antes
    m = keras.Sequential([
        keras.layers.Input((24, 24, 1)),
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    m.fit(entreno, validation_data=(X_val, Y_val), epochs=15)
    evaluar(m, X_val, Y_val)
    spec = (tf.TensorSpec((None, 24, 24, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=SALIDA)
    print("Modelo CNN exportado en", SALIDA)


if __name__ == "__main__":
    main()
