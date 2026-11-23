# CNN propia: clasificador de ojos abiertos/cerrados

Esta CNN se entrena con imágenes propias. Detecta si los ojos están abiertos o
cerrados y reemplaza al EAR en la detección de somnolencia (el EAR queda como
respaldo si no existe `cnn/ojos.onnx`).

Flujo: **capturar sesiones → validar con una sesión nueva → entrenar el modelo final → el sistema lo usa solo.**

Todos los comandos se corren desde la carpeta raíz del proyecto, en PowerShell.

## 0. Entornos

Hay dos entornos de Python, porque TensorFlow y MediaPipe no conviven (conflicto de `protobuf`):

| Entorno | Para qué | Instalar |
|---|---|---|
| `.venv` | capturar y correr el sistema (MediaPipe, OpenCV) | `pip install -r requirements.txt` |
| `.venv-rostros` | solo entrenar (TensorFlow, tf2onnx) | `pip install -r requirements-entrenamiento.txt` |

## 1. Capturar sesiones (entorno .venv)

Una **sesión** es una captura en un momento y con una luz determinados. Hacen falta
**al menos 3 sesiones con luces distintas** (por ejemplo: luz de día, lámpara, luz del techo).

```powershell
.\.venv\Scripts\python.exe cnn\capturar_ojos.py s1
```

Cambiar `s1` por `s2`, `s3`, ... en cada sesión. Teclas (con la ventana de la cámara en foco):

- **a** → guardar **ojos ABIERTOS**. Mover un poco la cabeza; parpadear lo menos posible.
- **c** → guardar **ojos CERRADOS**. **Cerrar los ojos antes de apretar `c`** y apretar `s`
  antes de abrirlos, para que no se cuelen fotos con los ojos abiertos.
- **s** → parar de guardar.
- **q** → salir.

Apuntar a **~500 de cada clase por sesión** (guarda los dos ojos por frame).
Las imágenes quedan en:

```
cnn/dataset/
  s1/abiertos/   s1/cerrados/
  s2/abiertos/   s2/cerrados/
  s3/abiertos/   s3/cerrados/
```

**Revisar las fotos antes de entrenar**: si en `cerrados` hay ojos abiertos (o al revés),
moverlas a `cnn/descartadas/`. Si se sacan fotos de una carpeta, renumerar las que quedan
desde 0; si no, al capturar de nuevo en esa sesión se pisan archivos.

> `cnn/dataset/` y `cnn/descartadas/` contienen datos biométricos y están en `.gitignore`.

## 2. Medir el modelo con una sesión que nunca vio (entorno .venv-rostros)

```powershell
.\.venv-rostros\Scripts\python.exe cnn\entrenar_ojos.py --val s3
```

Entrena con todas las sesiones menos `s3` y valida con `s3`. Al final imprime la
**matriz de confusión**, el **accuracy**, el **recall** y la **precision** de "cerrados":

- **Recall cerrados**: de los ojos cerrados, cuántos detectó (si es bajo: conductor dormido sin alarma).
- **Precision cerrados**: de las alarmas, cuántas eran reales (si es baja: falsas alarmas).

Este es el número honesto para el informe. Validar con fotos de la misma sesión
(sin `--val`) da resultados inflados, porque los frames de una sesión son casi iguales.

También se pueden elegir las sesiones de entrenamiento: `entrenar_ojos.py s1 s2 --val s3`.

## 3. Entrenar el modelo final (entorno .venv-rostros)

```powershell
.\.venv-rostros\Scripts\python.exe cnn\entrenar_ojos.py
```

Sin `--val` usa **todas** las sesiones (valida con un 20 % al azar) y exporta
**`cnn/ojos.onnx`**. Correrlo siempre **después** del paso 2, porque cada corrida
sobrescribe `ojos.onnx`.

## 4. Usarla

No hay que hacer nada más: al iniciar `sistema.py`, si existe `cnn/ojos.onnx` lo carga
con OpenCV (`cv2.dnn`) y detecta la somnolencia con la **CNN** en vez del EAR.

```powershell
.\sistema.bat
```

## Resultados obtenidos (un usuario, cámara Logitech C505)

| Entrenado con | Validado con (sesión nunca vista) | Ojos abiertos bien | Ojos cerrados bien |
|---|---|---|---|
| s1 | s2 | 2,6 % | 100 % |
| s1 | s3 | 82 % | 100 % |
| s1 + s2 | s3 | 100 % | 100 % |
| s1 + s3 | s2 | 100 % | 100 % |

Con una sola sesión el modelo aprende la luz de esa sesión y falla con otra;
con dos sesiones de luz distinta generaliza a una tercera.
**Limitación:** una sola persona y una sola cámara.

## Arquitectura de la CNN

Entrada 24×24 en escala de grises → `Conv(16) + MaxPool` → `Conv(32) + MaxPool`
→ `Flatten` → `Dense(64)` → `Dropout(0.4)` → `Dense(1, sigmoid)`.
Salida: probabilidad de "ojo cerrado" (umbral 0.5). 15 épocas, optimizador Adam.

El recorte del ojo es **cuadrado, según el ancho del ojo** (landmarks de MediaPipe Face Mesh),
igual en `capturar_ojos.py` y en `sistema.py`, para que el modelo vea en uso lo mismo que en el entrenamiento.

**Data augmentation** (solo al entrenar, no queda dentro del modelo exportado):
espejado horizontal, rotación de ±11°, brillo ±20 % y contraste ±20 %.

## Nota técnica (para la defensa)

- La CNN se **entrena en Keras/TensorFlow** (`tf_keras`) y se **exporta a ONNX** (opset 13).
- En ejecución se corre con **OpenCV (`cv2.dnn`)**, que no necesita TensorFlow,
  por eso convive con MediaPipe en un mismo programa (evita el conflicto de
  `protobuf` entre TensorFlow y MediaPipe).
- La separación entrenamiento/validación se hace a mano después de mezclar
  (`np.random.permutation`), porque `validation_split` de Keras toma siempre el último 20 %.
