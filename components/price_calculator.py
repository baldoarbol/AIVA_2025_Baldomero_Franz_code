"""
Script con las funciones de cálculo de beneficio y exposición de resultados
"""

import numpy as np
import matplotlib as mpl


def compute_price(num_big_t: int, num_small_t: int, price_big_t: float, price_small_t: float) -> float:
    """
    Cálculo del precio de los componentes en la imagen.
    """

    total_price = num_big_t * price_big_t + num_small_t * price_small_t
    return total_price
