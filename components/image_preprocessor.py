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
        Preprocesa una imagen aplicando reducción de ruido, escalado y conversión a escala de grises.
        """
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Escalar a la mitad
        height, width = denoised.shape[:2]
        scaled = cv2.resize(denoised, (width // 2, height // 2))

        # Convertir a escala de grises
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        return gray

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rota una imagen alrededor de su centro sin recortarla.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
