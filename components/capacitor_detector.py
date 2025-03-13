"""
Script con las funciones para detectar transistores en una imagen preprocesada.
"""

import numpy as np
import cv2

def detect_capacitors(image: np.ndarray):
    """
    Detección de capacitores en una imagen.

    :param image: Imagen de entrada en formato numpy.ndarray.
    :return: Dos diccionarios, small_caps y big_caps, con información sobre los capacitores detectados.
    """
    # Capacitores pequeños para MOCKUP
    small_caps = {
        "num": 2,
        "position": [(50, 80), (120, 200)],  # Coordenadas ficticias (x, y)
        "size": [(20, 30), (22, 28)]  # Ancho y alto en píxeles
    }

    # Capacitores grande para MOCKUP
    big_caps = {
        "num": 1,
        "position": [(200, 150)],  # Coordenadas ficticias (x, y)
        "size": [(50, 80)]  # Ancho y alto en píxeles
    }

    return small_caps, big_caps
