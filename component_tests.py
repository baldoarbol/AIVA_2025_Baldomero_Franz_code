import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Usa un backend más estable
import matplotlib.pyplot as plt

# === CONFIGURACIÓN (ajustable) ===

# Suavizado (cuanto mayor el valor, más borroso → menos ruido pero también menos detalle)
GAUSSIAN_KERNEL_SIZE = (9, 9)  # Más grande = más borroso
GAUSSIAN_SIGMA = 2             # Mayor sigma = más desenfoque

# Canny (bordes): umbrales
CANNY_THRESH_LOW = 50          # Baja = más bordes detectados, incluso ruido
CANNY_THRESH_HIGH = 150        # Alta = menos bordes, más estricta

# HoughCircles (detección de círculos)
DP = 1.2                       # Inverso de resolución acumuladora. Más bajo = más preciso, más lento
MIN_DIST = 20                 # Distancia mínima entre centros de círculos detectados
PARAM1 = 100                  # Umbral alto para Canny usado internamente
PARAM2 = 30                   # Sensibilidad de detección (más bajo = más círculos detectados)
MIN_RADIUS = 5                # Radio mínimo del círculo detectado
MAX_RADIUS = 100               # Radio máximo

# Tolerancia para considerar círculos concéntricos
MAX_CENTER_DISTANCE = 5       # En píxeles
MIN_RADIUS_DIFF = 3           # Diferencia mínima de radio
MAX_RADIUS_DIFF = 10          # Diferencia máxima de radio

# === 1. Cargar imagen ===
imagen = cv2.imread('../img/rec1-2.jpg')  # Ajusta esta ruta
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# === 2. Preprocesado: desenfoque para reducir ruido ===
blur = cv2.GaussianBlur(gris, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

# === 3. Detección de bordes (Canny) ===
edges = cv2.Canny(blur, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)

# === 4. Detección de círculos (HoughCircles) ===
circulos = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=DP,
    minDist=MIN_DIST,
    param1=PARAM1,
    param2=PARAM2,
    minRadius=MIN_RADIUS,
    maxRadius=MAX_RADIUS
)

# === 5. Detectar pares de círculos concéntricos ===
imagen_bn = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
transistores_detectados = 0

if circulos is not None:
    circulos = np.round(circulos[0, :]).astype("int")
    usados = set()
    for i, c1 in enumerate(circulos):
        for j, c2 in enumerate(circulos):
            if i >= j or (i, j) in usados or (j, i) in usados:
                continue
            dist_centros = np.linalg.norm(np.array([c1[0], c1[1]]) - np.array([c2[0], c2[1]]))
            diff_radio = abs(c1[2] - c2[2])
            if dist_centros < MAX_CENTER_DISTANCE and MIN_RADIUS_DIFF < diff_radio < MAX_RADIUS_DIFF:
                # Dibuja el círculo exterior en rojo
                cv2.circle(imagen_bn, (c1[0], c1[1]), max(c1[2], c2[2]), (0, 0, 255), 2)
                transistores_detectados += 1
                usados.add((i, j))

# === 6. Mostrar resultados paso a paso ===

# Imagen preprocesada (desenfocada)
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(blur, cmap='gray')
plt.title("1. Imagen desenfocada")
plt.axis('off')

# Bordes detectados
plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("2. Bordes (Canny)")
plt.axis('off')

# Detección final de transistores
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(imagen_bn, cv2.COLOR_BGR2RGB))
plt.title(f"3. Transistores detectados: {transistores_detectados}")
plt.axis('off')

plt.tight_layout()
plt.show()

# === 7. Imprimir resultado por consola ===
print(f"✅ Transistores detectados: {transistores_detectados}")
