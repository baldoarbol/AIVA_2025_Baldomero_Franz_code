from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.price_calculator import compute_price


class Test(TestCase):
    def test_compute_price(self):
        TestCase.assertEqual(self, compute_price(3, 6), 9)
