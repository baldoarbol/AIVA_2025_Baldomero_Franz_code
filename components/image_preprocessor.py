"""
Script con las funciones de preprocesamiento de imagen
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Procesa una imagen con una serie de transformaciones.

    :param image: Imagen de entrada en formato numpy.ndarray.
    :return: Imagen preprocesada en formato numpy.ndarray.
    """

    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return result
