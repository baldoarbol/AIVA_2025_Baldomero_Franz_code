"""
Script con las funciones de preprocesamiento de imagen
"""
import cv2
import numpy as np


def reduce_scale(image: np.ndarray, width: int = 512) -> np.ndarray:
    """
    Escala la imagen para que el ancho sea de 512 píxeles, manteniendo la relación de aspecto.
    """
    ratio = width / image.shape[1]
    height = int(image.shape[0] * ratio)
    return cv2.resize(image, (width, height))


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen a escala de grises.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_borders(image_gray: np.ndarray) -> np.ndarray:
    """
    Detecta los bordes en una imagen en escala de grises usando el algoritmo de Canny.
    """
    return cv2.Canny(image_gray, threshold1=100, threshold2=200)


def detect_circles(edges: np.ndarray) -> np.ndarray:
    """
    Detecta círculos en la imagen usando la transformada de Hough.
    """
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100
    )

    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    return output


def color_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica un filtro HSV para detectar tonos grises en una imagen.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 50, 220])
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    return mask


def process_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Procesa una imagen con una serie de transformaciones:
    - Reducción de escala
    - Detección de círculos sobre imagen en escala de grises
    - Filtro de color HSV sobre la copia original

    :param image: Imagen de entrada en formato numpy.ndarray.
    :return: (imagen con círculos detectados, máscara de color gris)
    """
    image_scaled = reduce_scale(image)
    image_copy = image_scaled.copy()

    # Flujo 1: Círculos detectados
    gray = image_to_grayscale(image_scaled)
    edges = detect_borders(gray)
    circles_detected = detect_circles(edges)

    # Flujo 2: Filtro de color gris
    color_mask = color_filter(image_copy)

    return circles_detected, color_mask
