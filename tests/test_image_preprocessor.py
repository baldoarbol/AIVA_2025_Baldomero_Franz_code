from unittest import TestCase

import cv2

from AIVA_2025_Baldomero_Franz_code.components.image_preprocessor import process_image

input_directory = "../test_img/"
image_name = "im01.png"
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)

class Test(TestCase):
    def test_process_image(self):
        TestCase.assertEqual(self, raw_img.shape[0], process_image(raw_img).shape[0])
        TestCase.assertEqual(self, raw_img.shape[1], process_image(raw_img).shape[1])
