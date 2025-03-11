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

small_t, big_t = detector.detect_transistors(preprocessed_img)

total_price = calculator.compute_price(small_t["num"], big_t["num"])

print(f"PRECIO TOTAL: {total_price}")


