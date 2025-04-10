from unittest import TestCase
import cv2
import numpy as np

from components.image_preprocessor import Preprocessor

input_directory = "img/"
image_name = "rec1-1.jpg"
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)


class TestPreprocessor(TestCase):
    def setUp(self):
        self.preprocessor = Preprocessor()
        self.raw_img = raw_img
        self.processed_img = self.preprocessor.preprocess_image(self.raw_img)

    def test_not_none(self):
        """La imagen procesada no debe ser None"""
        self.assertIsNotNone(self.processed_img)

    def test_is_grayscale(self):
        """La imagen procesada debe tener un solo canal (grises)"""
        self.assertEqual(len(self.processed_img.shape), 2)

    def test_same_dimensions(self):
        """La imagen procesada debe ser la mitad de alto y ancho que la original"""
        self.assertEqual(self.raw_img.shape[0], self.processed_img.shape[0] * 2)
        self.assertEqual(self.raw_img.shape[1], self.processed_img.shape[1] * 2)

    def test_output_type(self):
        """La imagen procesada debe ser un ndarray"""
        self.assertIsInstance(self.processed_img, np.ndarray)
