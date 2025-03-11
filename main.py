"""
Script principal del sistema
"""
import cv2

import components.image_preprocessor as preprocessor
import components.transistor_detector as detector
import components.price_calculator as calculator

input_directory = "./test_img/"
image_name = "im01.png"

PRICE_SMALL_T = 0.25
PRICE_BIG_T = 0.5

# Leer la imagen
raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)

# Preprocesar la imagen
preprocessed_img = preprocessor.process_image(raw_img)

# Obtener bounding boxes de los transistores detectados
small_t, big_t = detector.detect_transistors(preprocessed_img)

# Calcular el precio total de los componentes
total_price = calculator.compute_price(num_small_t=small_t["num"], num_big_t=big_t["num"], price_small_t=PRICE_SMALL_T,
                                       price_big_t=PRICE_BIG_T)

# Mostrar por pantalla el resultado
print(f"PRECIO TOTAL: {total_price}")
