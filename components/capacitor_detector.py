import cv2
import os
import csv
import numpy as np
from AIVA_2025_Baldomero_Franz_code.components.image_preprocessor import Preprocessor


class Capacitor:
    """
    Clase base para representar un capacitor detectado.
    """

    def __init__(self, x: int, y: int, w: int, h: int, price: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.price = price


class SmallCapacitor(Capacitor):
    """
    Representa un capacitor pequeño con precio fijo de 5.
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        super().__init__(x, y, w, h, price=5)


class BigCapacitor(Capacitor):
    """
    Representa un capacitor grande con precio fijo de 10.
    """

    def __init__(self, x: int, y: int, w: int, h: int):
        super().__init__(x, y, w, h, price=10)


class Detector:
    """
    Detector de capacitores basado en template matching y configuración de tipos.
    """

    def __init__(self, preprocessed_img: np.ndarray, templates_dir: str):
        self.image = preprocessed_img
        self.templates_dir = templates_dir
        self.preprocessor = Preprocessor()
        self.template_type_map = self.load_config()

    def load_config(self) -> dict:
        """
        Carga el mapeo de templates a tipo de capacitor desde config.csv
        """
        config_path = os.path.join(self.templates_dir, "config.csv")
        mapping = {}
        with open(config_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1].strip().lower()
        return mapping

    def non_max_suppression(self, rects, scores, overlap_thresh):
        if len(rects) == 0:
            return []

        rects = np.array(rects)
        scores = np.array(scores)

        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = rects[:, 2]
        y2 = rects[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[order[1:]]

            order = order[np.where(overlap <= overlap_thresh)[0] + 1]

        return keep

    def detect(self):
        """
        Ejecuta el proceso de detección y devuelve listas de capacitores grandes y pequeños.
        """
        all_rects = []
        all_scores = []
        all_types = []

        for fname in os.listdir(self.templates_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                cap_type = self.template_type_map.get(fname)
                if cap_type not in ('small', 'big'):
                    continue

                template_path = os.path.join(self.templates_dir, fname)
                template_img = cv2.imread(template_path)

                for angle in range(0, 360, 45):
                    rotated = self.preprocessor.rotate_image(template_img, angle)
                    template_gray = self.preprocessor.preprocess_image(rotated)
                    h, w = template_gray.shape

                    result = cv2.matchTemplate(self.image, template_gray, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.76
                    loc = np.where(result >= threshold)

                    for pt in zip(*loc[::-1]):
                        all_rects.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
                        all_scores.append(result[pt[1], pt[0]])
                        all_types.append(cap_type)

        # Aplicar NMS global sobre todas las detecciones
        keep = self.non_max_suppression(all_rects, all_scores, 0.3)

        small_caps = []
        big_caps = []
        for i in keep:
            x1, y1, x2, y2 = all_rects[i]
            w, h = x2 - x1, y2 - y1
            cap_type = all_types[i]
            if cap_type == 'small':
                small_caps.append(SmallCapacitor(x1, y1, w, h))
            elif cap_type == 'big':
                big_caps.append(BigCapacitor(x1, y1, w, h))

        return big_caps, small_caps
