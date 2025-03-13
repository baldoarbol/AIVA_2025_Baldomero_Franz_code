from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.price_calculator import compute_price


class Test(TestCase):
    def test_compute_price(self):
        TestCase.assertEqual(self, compute_price(num_big_c=10, num_small_c=10, price_small_c=0.1, price_big_c=0.9), 10)
