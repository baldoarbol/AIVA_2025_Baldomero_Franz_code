"""
Script principal del sistema
"""
import cv2

import components.image_preprocessor as preprocessor
import components.capacitor_detector as detector
import components.price_calculator as calculator

input_directory = "./test_img/"
image_name = "im01.png"

PRICE_SMALL_C = 0.25
PRICE_BIG_C = 0.5

# Leer la imagen
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)

# Preprocesar la imagen
preprocessed_img = preprocessor.process_image(raw_img)

# Obtener bounding boxes de los transistores detectados
small_c, big_c = detector.detect_capacitors(preprocessed_img)

# Calcular el precio total de los componentes
total_price = calculator.compute_price(num_small_c=small_c["num"], num_big_c=big_c["num"], price_small_c=PRICE_SMALL_C,
                                       price_big_c=PRICE_BIG_C)

# Mostrar por pantalla el resultado
print(f"PRECIO TOTAL: {total_price}")
