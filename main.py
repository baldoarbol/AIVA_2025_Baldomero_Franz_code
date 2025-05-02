import sys
import os
import cv2
from PIL import Image
from components.manager import CapacitorDetectionManager

IMG_DIR = "img"
CAPTURED_IMG_PATH = os.path.join(IMG_DIR, "captura.png")
TARGET_RESOLUTION = (4928, 3280)  # ancho x alto

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Debes especificar la ruta de la imagen o 'cam' para usar la cámara.")
        print("Uso: python main.py <ruta_imagen | cam> [ruta_templates]")
        sys.exit(1)

    source = sys.argv[1]
    templates_path = sys.argv[2] if len(sys.argv) > 2 else "templates/"

    if source.lower() == "cam":
        os.makedirs(IMG_DIR, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            sys.exit(1)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            print("Error: No se pudo capturar una imagen válida desde la cámara.")
            sys.exit(1)

        resized_frame = cv2.resize(frame, TARGET_RESOLUTION)

        # Convertir a RGB y guardar con PIL
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        image_pil.save(CAPTURED_IMG_PATH)

        image_path = CAPTURED_IMG_PATH
    else:
        image_path = source

    manager = CapacitorDetectionManager(image_path, templates_path)
    manager.run()
