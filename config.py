import os

# El token de Telegram se lee de la variable de entorno TG_TOKEN (no se guarda en el código)
TG_TOKEN = os.environ.get("TG_TOKEN", "")

# Índice de la cámara para OpenCV. En esta PC: 0 = Logitech C505, 1 = Iriun Webcam (virtual).
# Si cambia el orden (otro puerto USB, otra cámara), ajustar solo este número.
# Los scripts la abren con cv2.CAP_DSHOW: con el backend por defecto (MSMF) la Logitech tarda ~1 minuto en abrir.
CAMARA = 0

# Enviar la ubicación aproximada (según la IP) junto con la alerta de Telegram.
# Es un dato personal: el conductor tiene que estar de acuerdo.
ENVIAR_UBICACION = True

UMBRAL_ROSTRO = 70
FRAMES_CONFIRMAR = 15

EAR_UMBRAL = 0.21
SEG_ALERTA = 1.2
MICROSUENOS_ALERTA = 3
COOLDOWN = 120
SEG_SIN_ROSTRO = 2.0

MOSTRAR_MALLA = True
