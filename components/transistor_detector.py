"""
Script con las funciones para detectar transistores en una imagen preprocesada.
"""

import numpy as np
import cv2

def detect_transistors(image: np.ndarray):
    """
    Detección de transistores en una imagen.

    :param image: Imagen de entrada en formato numpy.ndarray.
    :return: Dos diccionarios, small_trans y big_trans, con información sobre los transistores detectados.
    """
    # Transistores pequeños para MOCKUP
    small_trans = {
        "num": 2,
        "position": [(50, 80), (120, 200)],  # Coordenadas ficticias (x, y)
        "size": [(20, 30), (22, 28)]  # Ancho y alto en píxeles
    }

    # Transistor grande para MOCKUP
    big_trans = {
        "num": 1,
        "position": [(200, 150)],  # Coordenadas ficticias (x, y)
        "size": [(50, 80)]  # Ancho y alto en píxeles
    }

    return small_trans, big_trans
