from unittest import TestCase

import cv2

from AIVA_2025_Baldomero_Franz_code.components.capacitor_detector import detect_capacitors

input_directory = "../test_img/"
image_name = "im01.png"
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)


class Test(TestCase):
    def test_detect_transistor(self):
        small_c, big_c = detect_capacitors(raw_img)
        TestCase.assertNotEqual(self, small_c["num"], None)
        TestCase.assertNotEqual(self, small_c["position"], None)
        TestCase.assertNotEqual(self, small_c["size"], None)
        TestCase.assertEqual(self, small_c["num"], len(small_c["position"]))
        TestCase.assertEqual(self, small_c["num"], len(small_c["size"]))
