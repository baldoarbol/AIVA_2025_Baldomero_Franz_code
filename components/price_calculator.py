"""
Script con las funciones de cálculo de beneficio y exposición de resultados
"""

import numpy as np
import matplotlib as mpl


def compute_price(num_big_c: int, num_small_c: int, price_big_c: float, price_small_c: float) -> float:
    """
    Cálculo del precio de los componentes en la imagen.
    """

    total_price = num_big_c * price_big_c + num_small_c * price_small_c
    return total_price
