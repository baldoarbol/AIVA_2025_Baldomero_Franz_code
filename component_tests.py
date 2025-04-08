"""
Script para probar componentes
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import components.image_preprocessor as preprocessor

matplotlib.use('TkAgg')

# Cargar imagen
img_path = "../img/rec1-1.jpg"
image = cv2.imread(img_path)

# Validar carga
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen desde la ruta: {img_path}")

# Procesar imagen
image_scaled = preprocessor.reduce_scale(image)
image_Scaled = preprocessor.denoise_image(image_scaled)
gray = preprocessor.image_to_grayscale(image_scaled)
edges = preprocessor.detect_borders(gray)
circles_detected, color_mask = preprocessor.process_image(image)

# Convertir imágenes de BGR a RGB para mostrar con matplotlib
image_scaled_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
circles_detected_rgb = cv2.cvtColor(circles_detected, cv2.COLOR_BGR2RGB)

# Mostrar resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Imagen escalada")
plt.imshow(image_scaled_rgb)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Bordes detectados")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Círculos detectados")
plt.imshow(circles_detected_rgb)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Máscara de color gris")
plt.imshow(color_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
