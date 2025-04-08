import cv2
import numpy as np

# Ruta de la imagen
img_path = "../img/rec15-1.jpg"

# Cargar imagen
image = cv2.imread(img_path)
if image is None:
    print("No se pudo cargar la imagen.")
    exit()

# Redimensionar imagen para vista más cómoda
image = cv2.resize(image, (640, 480))

# Crear ventana para sliders
cv2.namedWindow("Sliders", cv2.WINDOW_NORMAL)

def nothing(x):
    pass

# Sliders para HSV
cv2.createTrackbar("H min", "Sliders", 0, 179, nothing)
cv2.createTrackbar("S min", "Sliders", 0, 255, nothing)
cv2.createTrackbar("V min", "Sliders", 0, 255, nothing)
cv2.createTrackbar("H max", "Sliders", 179, 179, nothing)
cv2.createTrackbar("S max", "Sliders", 50, 255, nothing)
cv2.createTrackbar("V max", "Sliders", 255, 255, nothing)

# Sliders para cantidad y tamaño de erosiones/dilataciones
cv2.createTrackbar("Erosiones", "Sliders", 0, 10, nothing)
cv2.createTrackbar("Dilataciones", "Sliders", 0, 10, nothing)
cv2.createTrackbar("Kernel size", "Sliders", 1, 20, nothing)  # Tamaño del kernel (1-20)

while True:
    # Obtener valores de sliders
    h_min = cv2.getTrackbarPos("H min", "Sliders")
    s_min = cv2.getTrackbarPos("S min", "Sliders")
    v_min = cv2.getTrackbarPos("V min", "Sliders")
    h_max = cv2.getTrackbarPos("H max", "Sliders")
    s_max = cv2.getTrackbarPos("S max", "Sliders")
    v_max = cv2.getTrackbarPos("V max", "Sliders")
    erosiones = cv2.getTrackbarPos("Erosiones", "Sliders")
    dilataciones = cv2.getTrackbarPos("Dilataciones", "Sliders")
    ksize = cv2.getTrackbarPos("Kernel size", "Sliders")
    ksize = max(1, ksize)  # Evitar kernel 0
    if ksize % 2 == 0:
        ksize += 1  # Hacerlo impar (recomendado en morfología)

    # Convertir imagen a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aplicar filtro HSV para detectar gris
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Crear kernel morfológico
    kernel = np.ones((ksize, ksize), np.uint8)

    # Aplicar erosión y dilatación
    morph = mask.copy()
    if erosiones > 0:
        morph = cv2.erode(morph, kernel, iterations=erosiones)
    if dilataciones > 0:
        morph = cv2.dilate(morph, kernel, iterations=dilataciones)

    # Mostrar vistas
    cv2.imshow("Original", image)
    cv2.imshow("Filtro HSV", mask)
    cv2.imshow("Morfología", morph)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
