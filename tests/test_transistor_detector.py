from unittest import TestCase

import cv2

from AIVA_2025_Baldomero_Franz_code.components.transistor_detector import detect_transistors

input_directory = "../test_img/"
image_name = "im01.png"
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)


class Test(TestCase):
    def test_detect_transistor(self):
        small_t, big_t = detect_transistors(raw_img)
        TestCase.assertNotEqual(self, small_t["num"], None)
        TestCase.assertNotEqual(self, small_t["position"], None)
        TestCase.assertNotEqual(self, small_t["size"], None)
        TestCase.assertEqual(self, small_t["num"], len(small_t["position"]))
        TestCase.assertEqual(self, small_t["num"], len(small_t["size"]))
