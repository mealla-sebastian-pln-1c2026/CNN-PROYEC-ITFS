# Sistema de Identificación de Conductor y Detección de Somnolencia

Proyecto Integrador de Diego Sebastián Mealla y Horacio Barrabasqui.

Identifica al conductor con la cámara web, detecta somnolencia en tiempo real y da
alertas por voz, por Arduino (opcional) y por Telegram (opcional).

## Cómo funciona

1. **Identificación del conductor:** MediaPipe Face Detection encuentra la cara y un
   reconocedor **LBPH** (OpenCV) la compara con los conductores registrados en
   `conductores.db`. Se confirma cuando lo reconoce en 15 frames seguidos.
2. **Detección de somnolencia:** MediaPipe Face Mesh ubica los ojos y una **CNN propia**
   (`cnn/ojos.onnx`) decide en cada frame si están abiertos o cerrados. Si no existe el
   modelo, se usa el **EAR** (Eye Aspect Ratio) como respaldo.
3. **Alertas:**
   - Ojos cerrados más de 1,2 s → "¡Despierta!" por voz y buzzer del Arduino.
   - Cada 3 microsueños → mensaje por Telegram al contacto del conductor (como máximo uno cada 2 minutos).
   - Sin rostro más de 2 s → "Mire al frente".

## Requisitos

- **Windows 10 u 11**
- **Python 3.12** (MediaPipe 0.10.14 no funciona con versiones más nuevas)
- Cámara web (probado con Logitech C505)

## Instalación

```powershell
git clone https://github.com/mealla-sebastian-pln-1c2026/CNN-PROYEC-ITFS.git
cd CNN-PROYEC-ITFS
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso

1. **Registrar un conductor:** abrir `gestor.bat`, cargar el nombre (y el chat de Telegram,
   si se usa). La cámara toma 40 fotos del rostro, una cada 0,2 s: mover un poco la cabeza
   mientras tanto.
2. **Iniciar el sistema:** abrir `sistema.bat`. Identifica al conductor y empieza a controlar la somnolencia.
3. **Salir:** tecla `q`.

## Configuración (`config.py`)

| Parámetro | Valor | Qué hace |
|---|---|---|
| `CAMARA` | 0 | Índice de la cámara. Si abre otra cámara (por ejemplo, una virtual), probar 1, 2... |
| `UMBRAL_ROSTRO` | 70 | Distancia máxima de LBPH para aceptar a un conductor (más bajo = más estricto) |
| `FRAMES_CONFIRMAR` | 15 | Frames seguidos reconociendo al mismo conductor para confirmarlo |
| `EAR_UMBRAL` | 0.21 | Por debajo de este EAR el ojo se considera cerrado (solo sin CNN) |
| `SEG_ALERTA` | 1.2 | Segundos con los ojos cerrados antes de la alarma |
| `MICROSUENOS_ALERTA` | 3 | Cada cuántos microsueños se avisa por Telegram |
| `COOLDOWN` | 120 | Segundos mínimos entre mensajes de Telegram |
| `SEG_SIN_ROSTRO` | 2.0 | Segundos sin ver la cara antes de "Mire al frente" |
| `MOSTRAR_MALLA` | True | Dibujar la malla de MediaPipe sobre la cara |

## Telegram (opcional)

Sin esta configuración el sistema funciona igual, pero no envía mensajes.

1. Pedir el token del bot al responsable del proyecto **por un medio privado**.
2. Guardarlo como variable de entorno y **abrir una terminal nueva**:

```powershell
setx TG_TOKEN "PEGAR_EL_TOKEN_ACA"
```

El token nunca se escribe en el código ni se sube a GitHub.

## Arduino (opcional)

Cargar `arduino/deteccion_rostro/deteccion_rostro.ino` en la placa. El sistema lo detecta
solo por USB y le envía `B` (buzzer encendido), `b` (apagado) y `X` (apagar todo al salir).
Sin Arduino conectado, el sistema funciona con alertas por voz y en pantalla.

## CNN de ojos

La CNN clasifica recortes de ojo de 24×24 en gris como abierto o cerrado:
`Conv(16) + MaxPool → Conv(32) + MaxPool → Dense(64) → Dropout(0.4) → Dense(1, sigmoid)`.
Se entrena en TensorFlow/Keras y se exporta a ONNX; en ejecución corre con OpenCV
(`cv2.dnn`), sin TensorFlow.

El paso a paso para capturar, validar y entrenar está en **[cnn/COMO_ENTRENAR.md](cnn/COMO_ENTRENAR.md)**.
Resumen:

```powershell
.\.venv\Scripts\python.exe cnn\capturar_ojos.py s1                 # capturar una sesión (s1, s2, s3...)
.\.venv-rostros\Scripts\python.exe cnn\entrenar_ojos.py --val s3   # medir con una sesión nunca vista
.\.venv-rostros\Scripts\python.exe cnn\entrenar_ojos.py            # modelo final con todas las sesiones
.\.venv\Scripts\python.exe cnn\comparar_ear.py prueba              # comparar CNN vs EAR
```

El entrenamiento usa un entorno aparte (`.venv-rostros`), porque TensorFlow y MediaPipe
necesitan versiones distintas de `protobuf`:

```powershell
py -3.12 -m venv .venv-rostros
.\.venv-rostros\Scripts\python.exe -m pip install -r requirements-entrenamiento.txt
```

### Resultados

Validación con sesiones de captura que el modelo nunca vio (un usuario, cámara Logitech C505):

| Entrenado con | Validado con | Ojos abiertos bien | Ojos cerrados bien |
|---|---|---|---|
| s1 | s2 | 2,6 % | 100 % |
| s1 | s3 | 82 % | 100 % |
| s1 + s2 | s3 | 100 % | 100 % |
| s1 + s3 | s2 | 100 % | 100 % |

Con una sola sesión el modelo aprende la luz de esa sesión; con dos sesiones de luz
distinta generaliza a una tercera.

**CNN vs EAR** (sesión `prueba`, 670 fotos, de frente y con buena luz): los dos métodos
aciertan el 100 %. En estas condiciones el EAR separa muy bien las clases (abiertos ≥ 0,31,
cerrados ≤ 0,06). Queda pendiente comparar en condiciones difíciles: cabeza girada, poca luz,
anteojos u ojos entrecerrados.

**Limitaciones:** una sola persona y una sola cámara.

## Problemas comunes

- **No abre la cámara / "No se pudo abrir la cámara":** revisar que esté conectada y probar
  otro valor de `CAMARA` en `config.py`.
- **No reconoce al conductor:** las fotos registradas son de otra luz o de otra cámara.
  Volver a cargarlo desde `gestor.bat` (las fotos nuevas se suman a las anteriores y se
  mantiene su chat de Telegram).

## Estructura

```
sistema.py               programa principal (identificación + somnolencia)
gestor_conductores.py    ventana para registrar/eliminar conductores (gestor.bat)
cargar_conductor.py      captura las fotos del rostro de un conductor
rostro.py                detección de rostro con MediaPipe
db.py                    base de datos SQLite (conductores.db)
config.py                parámetros del sistema
probar_contador.py       prueba del contador de frames de la identificación
cnn/                     CNN de ojos: captura, entrenamiento, modelo y comparación con EAR
arduino/                 programa de la placa (buzzer)
```

## Privacidad

`conductores.db`, `cnn/dataset/` y `cnn/descartadas/` contienen datos biométricos y
**no se suben a GitHub**. Cada persona registra sus propios datos en su PC.
