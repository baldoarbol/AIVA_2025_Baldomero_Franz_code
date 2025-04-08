import cv2
import numpy as np

# Ruta a la imagen
img_path = "../img/rec10-3.jpg"

# Modo de operación: 'denoising', 'edges', 'circles'
MODE = "circles"  # Cambia a 'edges' o 'circles' para probar

# Cargar y preprocesar imagen
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en {img_path}")


def reduce_scale(image, width=512):
    ratio = width / image.shape[1]
    height = int(image.shape[0] * ratio)
    return cv2.resize(image, (width, height))


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Preprocesamiento inicial
image_scaled = reduce_scale(image)
gray = to_grayscale(image_scaled)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Para emparejar canales al hacer hstack

# ---------- INTERFACES INTERACTIVAS POR MODO ---------- #

if MODE == "denoising":
    def update(val):
        ksize = cv2.getTrackbarPos('ksize', 'Preview')
        ksize = max(1, ksize | 1)  # Asegurar impar y ≥1
        denoised = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([gray_bgr, denoised_bgr])
        cv2.imshow('Preview', combined)


    cv2.namedWindow('Preview')
    cv2.createTrackbar('ksize', 'Preview', 3, 50, update)
    update(0)

elif MODE == "edges":
    def update(val):
        t1 = cv2.getTrackbarPos('Threshold1', 'Preview')
        t2 = cv2.getTrackbarPos('Threshold2', 'Preview')
        edges = cv2.Canny(gray, t1, t2)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([gray_bgr, edges_bgr])
        cv2.imshow('Preview', combined)


    cv2.namedWindow('Preview')
    cv2.createTrackbar('Threshold1', 'Preview', 100, 500, update)
    cv2.createTrackbar('Threshold2', 'Preview', 200, 500, update)
    update(0)

elif MODE == "circles":
    def update(val):
        # Leer parámetros de sliders (asegurar límites válidos)
        dp = cv2.getTrackbarPos('dp', 'Preview') / 10.0
        param1 = cv2.getTrackbarPos('param1', 'Preview')
        param2 = max(1, cv2.getTrackbarPos('param2', 'Preview'))  # Nunca 0
        minR = cv2.getTrackbarPos('minRadius', 'Preview')
        maxR = cv2.getTrackbarPos('maxRadius', 'Preview')

        # Mostrar imagen original (grayscale en BGR)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Bordes usando los parámetros de Hough
        edges = cv2.Canny(gray, param1, param2)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Detección de círculos
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=dp if dp > 0 else 1,
            minDist=20,
            param1=param1,
            param2=param2,
            minRadius=minR,
            maxRadius=maxR
        )

        output = gray_bgr.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cv2.circle(output, (c[0], c[1]), c[2], (0, 255, 0), 2)
                cv2.circle(output, (c[0], c[1]), 2, (0, 0, 255), 3)

        combined = np.hstack([gray_bgr, edges_bgr, output])
        cv2.imshow('Preview', combined)


    cv2.namedWindow('Preview')
    # Crear sliders ANTES de llamar a update
    cv2.createTrackbar('dp', 'Preview', 12, 30, update)  # dp = 1.2
    cv2.createTrackbar('param1', 'Preview', 100, 300, update)
    cv2.createTrackbar('param2', 'Preview', 100, 150, update)
    cv2.createTrackbar('minRadius', 'Preview', 5, 100, update)
    cv2.createTrackbar('maxRadius', 'Preview', 50, 200, update)

    # Llamar a update SOLO después de crear todos los sliders
    update(0)



else:
    raise ValueError(f"Modo no reconocido: {MODE}")

cv2.waitKey(0)
cv2.destroyAllWindows()
