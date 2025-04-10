import cv2
import os
import csv
from unittest import TestCase

from components.capacitor_detector import Detector
from components.image_preprocessor import Preprocessor


def iou(boxA, boxB):
    """Calcula el IoU entre dos cajas."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value


class TestCapacitorDetector(TestCase):
    def setUp(self):
        """
        Carga la imagen y la preprocesa antes de cada test.
        """
        input_directory = "img/"
        image_name = "rec1-1.jpg"
        image_path = os.path.join(input_directory, image_name)
        raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.assertIsNotNone(raw_img, "No se pudo cargar la imagen de entrada")

        preprocessor = Preprocessor()
        self.processed_img = preprocessor.preprocess_image(raw_img)
        self.detector = Detector(self.processed_img, "templates/")

    def test_detect_capacitors(self):
        """
        Verifica que el detector encuentra capacitores grandes y pequeños sin errores.
        """
        big_caps, small_caps = self.detector.detect()
        self.assertIsInstance(big_caps, list)
        self.assertIsInstance(small_caps, list)

        for cap in big_caps + small_caps:
            self.assertGreater(cap.w, 0)
            self.assertGreater(cap.h, 0)
            self.assertGreaterEqual(cap.x, 0)
            self.assertGreaterEqual(cap.y, 0)

        self.assertTrue(len(big_caps) + len(small_caps) > 0, "No se detectaron capacitores en la imagen")

    def test_accuracy_vs_labels(self):
        """
        Evalúa la precisión de la detección comparando con las anotaciones en el CSV.
        Muestra una imagen combinada con ground truth y predicciones.
        """

        def normalize_type(label_type: str) -> str:
            """
            Convierte el tipo del CSV al formato usado internamente: 'big' o 'small'.
            """
            label_type = label_type.strip().lower()
            if "big" in label_type:
                return "big"
            elif "small" in label_type:
                return "small"
            return label_type

        label_path = "labels/labels_rec1-1.csv"
        self.assertTrue(os.path.exists(label_path), "No se encontró el archivo de etiquetas")

        # Cargar anotaciones del CSV
        with open(label_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # saltar encabezado
            labels = []
            for row in reader:
                if len(row) >= 5:
                    tipo, x, y, w, h = row[:5]
                    try:
                        labels.append((normalize_type(tipo), int(x), int(y), int(w), int(h)))
                    except ValueError:
                        continue

        # Ejecutar detección
        big_caps, small_caps = self.detector.detect()

        # Escalar detecciones a coordenadas originales
        scale = 2.0
        detected = {
            'big': [(int(c.x * scale), int(c.y * scale), int(c.w * scale), int(c.h * scale)) for c in big_caps],
            'small': [(int(c.x * scale), int(c.y * scale), int(c.w * scale), int(c.h * scale)) for c in small_caps]
        }

        # Contar cuántas etiquetas están correctamente detectadas
        matched = 0
        for tipo, x, y, w, h in labels:
            gt_box = (x, y, w, h)
            for d_box in detected.get(tipo, []):
                if iou(gt_box, d_box) > 0.5:
                    matched += 1
                    break

        total = len(labels)
        accuracy = matched / total if total > 0 else 0

        # --- Visualización combinada ---
        raw_image = cv2.imread("img/rec1-1.jpg")
        gt_image = raw_image.copy()
        pred_image = raw_image.copy()

        # Dibujar ground truth (verde)
        for tipo, x, y, w, h in labels:
            color = (0, 255, 0)
            cv2.rectangle(gt_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(gt_image, tipo, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Dibujar predicciones (rojo)
        for tipo in detected:
            for x, y, w, h in detected[tipo]:
                color = (0, 0, 255)
                cv2.rectangle(pred_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(pred_image, tipo, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Concatenar imágenes horizontalmente
        combined = cv2.hconcat([gt_image, pred_image])

        # Redimensionar para visualización
        preview = cv2.resize(combined, (0, 0), fx=0.20, fy=0.20)
        cv2.imshow("GT vs Predicciones", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Accuracy vs etiquetas: {accuracy:.2f} ({matched}/{total})")
        self.assertGreaterEqual(accuracy, 0.7, "El accuracy está por debajo del umbral aceptable (70%)")
