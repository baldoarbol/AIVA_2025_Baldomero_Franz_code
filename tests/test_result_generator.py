import unittest
import numpy as np
import cv2
from components.result_generator import ResultGenerator
from components.capacitor_detector import BigCapacitor, SmallCapacitor


class DummyImage:
    """
    Utilidad para generar una imagen falsa con valores válidos.
    """

    @staticmethod
    def generate(width=400, height=300):
        return np.zeros((height, width, 3), dtype=np.uint8)


class TestResultGenerator(unittest.TestCase):
    def setUp(self):
        """Configura imagen y capacitores simulados"""
        self.image = DummyImage.generate()
        self.big_caps = [BigCapacitor(10, 10, 20, 20) for _ in range(2)]  # 2 * 10 = 20
        self.small_caps = [SmallCapacitor(30, 30, 10, 10) for _ in range(3)]  # 3 * 5 = 15
        self.scale = 1.0
        self.generator = ResultGenerator(self.image, self.big_caps, self.small_caps, self.scale)

    def test_compute_price(self):
        """Verifica que el cálculo de precio y beneficio sea correcto"""
        total, profit = self.generator.compute_price(board_cost=10)
        self.assertEqual(total, 35)  # 20 + 15
        self.assertEqual(profit, 25)

    def test_show_result_runs(self):
        """Asegura que show_result no lanza errores"""
        # Redefinimos métodos de OpenCV para evitar abrir ventana y guardar archivo
        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: None
        cv2.destroyAllWindows = lambda *args, **kwargs: None
        cv2.imwrite = lambda *args, **kwargs: True  # Simula éxito al guardar

        try:
            self.generator.show_result(board_cost=10)
        except Exception as e:
            self.fail(f"show_result lanzó una excepción: {e}")


if __name__ == "__main__":
    unittest.main()
