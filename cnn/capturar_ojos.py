import sys
import os
import cv2
import numpy as np
import mediapipe as mp

# config.py está en la carpeta de arriba (raíz del proyecto)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OJO_I = [33, 160, 158, 133, 153, 144]
OJO_D = [362, 385, 387, 263, 373, 380]
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"dataset")


def recorte(gray, pts, idx, m=0.6):
    xs, ys = pts[idx, 0], pts[idx, 1]
    cx, cy = xs.mean(), ys.mean()                    # centro del ojo
    lado = (xs.max() - xs.min()) * (1 + 2 * m)       # cuadrado según el ANCHO
    x1, y1 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    x2, y2 = int(cx + lado / 2), int(cy + lado / 2)
    c = gray[y1:y2, x1:x2]
    if c.size == 0:
        return None
    return cv2.resize(c, (24, 24))

def ear(p):
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = 2 * np.linalg.norm(p[0] - p[3])
    return v / (h + 1e-8)

def main():
    sesion = sys.argv[1] if len(sys.argv) > 1 else "s1"
    carpeta = f"{DIR}/{sesion}"
    for t in ("abiertos", "cerrados"):
        os.makedirs(f"{carpeta}/{t}", exist_ok=True)
    cont = {t: len(os.listdir(f"{carpeta}/{t}")) for t in ("abiertos", "cerrados")}
    csv = open(f"{carpeta}/ear.csv", "a", encoding="utf-8") 
    malla = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cam = cv2.VideoCapture(config.CAMARA, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print(f"No se pudo abrir la cámara {config.CAMARA}. ¿Está conectada? Revisá CAMARA en config.py")
        return
    modo = None
    print("Sesion:", sesion)
    print("a = guardar ABIERTOS   c = guardar CERRADOS   s = parar   q = salir")
    while True:
        ok, frame = cam.read()
        if not ok:
            print("La cámara dejó de enviar imagen (¿se desconectó?)")
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = malla.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks and modo:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([(p.x * w, p.y * h) for p in lm])
            for idx in (OJO_I, OJO_D):
                c = recorte(gray, pts, idx)
                if c is not None:
                    cv2.imwrite(f"{carpeta}/{modo}/{cont[modo]}.png", c)
                    csv.write(f"{modo}/{cont[modo]}.png,{ear(pts[idx]):.4f}\n")
                    cont[modo] += 1
        cv2.putText(frame, f"{sesion}  modo:{modo}  abiertos:{cont['abiertos']}  cerrados:{cont['cerrados']}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capturar ojos - a/c/s/q", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("a"):
            modo = "abiertos"
        elif k == ord("c"):
            modo = "cerrados"
        elif k == ord("s"):
            modo = None
        elif k == ord("q"):
            break
    cam.release()
    csv.close()
    cv2.destroyAllWindows()
    print("Guardado:", sesion, cont)


if __name__ == "__main__":
    main()
