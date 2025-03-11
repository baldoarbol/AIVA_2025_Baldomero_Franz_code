"""
Script principal del sistema
"""
import cv2

import components.image_preprocessor as preprocessor
import components.transistor_detector as detector
import components.price_calculator as calculator

input_directory = "./test_img/"
image_name = "im01.png"

raw_img = cv2.imread(input_directory + image_name, cv2.IMREAD_COLOR)

preprocessed_img = preprocessor.process_image(raw_img)

num_small_t = detector.detect_transistor(t_type="small")
num_big_t = detector.detect_transistor(t_type="big")

total_price = calculator.compute_price(num_small_t, num_big_t)

print(f"PRECIO TOTAL: {total_price}")


