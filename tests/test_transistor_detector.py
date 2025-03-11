from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.transistor_detector import detect_transistors


class Test(TestCase):
    def test_detect_transistor(self):
        TestCase.assertEqual(self, detect_transistors(t_type="big"), 2)
        TestCase.assertEqual(self, detect_transistors(t_type="small"), 7)
        TestCase.assertEqual(self, detect_transistors(t_type="incorrect"), 0)
