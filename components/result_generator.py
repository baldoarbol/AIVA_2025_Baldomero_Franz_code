import cv2
import os
import datetime
from typing import List
from AIVA_2025_Baldomero_Franz_code.components.capacitor_detector import SmallCapacitor, BigCapacitor


class ResultGenerator:
    """
    Clase para generar resultados visuales y económicos tras la detección de capacitores.
    """

    def __init__(self, image_original, big_caps: List[BigCapacitor], small_caps: List[SmallCapacitor], scale_factor: float):
        """
        Inicializa el generador con imagen original, listas de capacitores y factor de escala.
        """
        self.image = image_original.copy()
        self.big_caps = big_caps
        self.small_caps = small_caps
        self.scale = scale_factor

    def compute_price(self, board_cost: float) -> tuple:
        """
        Calcula el precio total de componentes y el beneficio.
        """
        total = sum([c.price for c in self.big_caps + self.small_caps])
        profit = total - board_cost
        return total, profit

    def show_result(self, board_cost: float):
        """
        Muestra y guarda la imagen con los resultados dibujados.
        """
        annotated = self.image.copy()

        # Dibujar capacitores grandes (rojo)
        for cap in self.big_caps:
            x, y = int(cap.x / self.scale), int(cap.y / self.scale)
            w, h = int(cap.w / self.scale), int(cap.h / self.scale)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 4)

        # Dibujar capacitores pequeños (verde)
        for cap in self.small_caps:
            x, y = int(cap.x / self.scale), int(cap.y / self.scale)
            w, h = int(cap.w / self.scale), int(cap.h / self.scale)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Calcular precios
        total, profit = self.compute_price(board_cost)

        # Texto informativo
        info = f"Grandes: {len(self.big_caps)} | Pequeños: {len(self.small_caps)}"
        info2 = f"Precio componentes: {total} | Beneficio: {profit}"

        # Posicionar texto con margen desde la parte inferior izquierda
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 4
        y_offset = 40

        # Dibujar sombra negra
        cv2.putText(annotated, info, (20, h - y_offset - 60), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(annotated, info2, (20, h - y_offset), font, font_scale, (0, 0, 0), thickness + 2)

        # Dibujar texto blanco encima
        cv2.putText(annotated, info, (20, h - y_offset - 60), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(annotated, info2, (20, h - y_offset), font, font_scale, (255, 255, 255), thickness)

        # Mostrar imagen en pantalla con escala reducida
        preview = cv2.resize(annotated, (int(w * 0.25), int(h * 0.25)))
        cv2.imshow("Resultado", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Guardar imagen
        os.makedirs("./output", exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"./output/output-{date_str}.png"
        success = cv2.imwrite(filename, annotated)
        if success:
            print(f"Resultado guardado en {filename}")
        else:
            print("Error al guardar la imagen")