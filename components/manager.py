import cv2
from .image_preprocessor import Preprocessor
from .capacitor_detector import Detector
from .result_generator import ResultGenerator


class CapacitorDetectionManager:
    """
    Clase encargada de coordinar la detección de capacitores.
    """

    def __init__(self, image_path: str, templates_path: str = "./templates"):
        """
        Inicializa el gestor con las rutas de imagen y templates.
        """
        self.image_path = image_path
        self.templates_path = templates_path
        self.image = self._load_image()

    def _load_image(self):
        """
        Carga la imagen desde la ruta indicada.
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {self.image_path}")
        return image

    def run(self):
        """
        Ejecuta el pipeline de detección y muestra resultados.
        """
        pre = Preprocessor()
        preprocessed = pre.preprocess_image(self.image)

        detector = Detector(preprocessed, self.templates_path)
        big_caps, small_caps = detector.detect()

        scale = 0.5
        board_cost = 40
        result_gen = ResultGenerator(self.image, big_caps, small_caps, scale)
        result_gen.show_result(board_cost)
