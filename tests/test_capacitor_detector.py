import cv2
import os
from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.capacitor_detector import Detector
from AIVA_2025_Baldomero_Franz_code.components.image_preprocessor import Preprocessor


class TestCapacitorDetector(TestCase):
    def setUp(self):
        """
        Carga la imagen y la preprocesa antes de cada test.
        """
        input_directory = "../../img/"
        image_name = "rec1-1.jpg"
        image_path = os.path.join(input_directory, image_name)
        raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.assertIsNotNone(raw_img, "No se pudo cargar la imagen de entrada")

        preprocessor = Preprocessor()
        self.processed_img = preprocessor.preprocess_image(raw_img)
        self.detector = Detector(self.processed_img, "../templates")

    def test_detect_capacitors(self):
        """
        Verifica que el detector encuentra capacitores grandes y pequeños sin errores.
        """
        big_caps, small_caps = self.detector.detect()

        # Verificaciones básicas
        self.assertIsInstance(big_caps, list)
        self.assertIsInstance(small_caps, list)

        for cap in big_caps + small_caps:
            self.assertGreater(cap.w, 0)
            self.assertGreater(cap.h, 0)
            self.assertGreaterEqual(cap.x, 0)
            self.assertGreaterEqual(cap.y, 0)

        # Al menos una detección debe existir en la imagen de prueba
        self.assertTrue(len(big_caps) + len(small_caps) > 0, "No se detectaron capacitores en la imagen")
