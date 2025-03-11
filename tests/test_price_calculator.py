from unittest import TestCase

from AIVA_2025_Baldomero_Franz_code.components.price_calculator import compute_price


class Test(TestCase):
    def test_compute_price(self):
        TestCase.assertEqual(self, compute_price(num_big_t=10, num_small_t=10, price_small_t=0.1, price_big_t=0.9), 10)
