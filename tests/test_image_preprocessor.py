from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.image_preprocessor import process_image


class Test(TestCase):
    def test_process_image(self):
        TestCase.assertEqual(self, process_image(), "Imagen procesada")
