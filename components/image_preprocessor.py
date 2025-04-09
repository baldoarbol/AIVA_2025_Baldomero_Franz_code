import cv2
import numpy as np


class Preprocessor:
    """
    Clase encargada de preprocesar imágenes para facilitar su análisis posterior.
    Aplica reducción de ruido y conversión a escala de grises.
    """

    def __init__(self):
        """
        Inicializa el preprocesador. Actualmente no recibe parámetros.
        """
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen aplicando reducción de ruido y conversión a escala de grises.

        Parámetros:
            image (np.ndarray): Imagen en color a procesar.

        Devuelve:
            np.ndarray: Imagen procesada en escala de grises.
        """
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convertir a escala de grises
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        return gray
