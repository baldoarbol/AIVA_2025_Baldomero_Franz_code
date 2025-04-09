import cv2
from components.capacitor_detector import Detector
from components.image_preprocessor import Preprocessor
from components.result_generator import ResultGenerator

# Cargar imagen original
image_path = "../img/rec1-1.jpg"
image = cv2.imread(image_path)

# Preprocesar imagen
pre = Preprocessor()
preprocessed = pre.preprocess_image(image)

# Detectar capacitores
templates_path = "./templates"
detector = Detector(preprocessed, templates_path)
big_caps, small_caps = detector.detect()

# Generar resultados
scale = 0.5
board_cost = 40  # Precio de compra de la placa
result_gen = ResultGenerator(image, big_caps, small_caps, scale)
result_gen.show_result(board_cost)