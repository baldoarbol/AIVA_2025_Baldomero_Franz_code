import cv2
import numpy as np
import os

# Rutas
img_path = "../img/rec1-1.jpg"  # 1, 14
templates_dir = "./templates"

# Cargar imagen principal en color y también en escala de grises para matching
img_color = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_draw = img_color.copy()

# Thresholds por template
template_thresholds = {
    "template01.png": 0.6,
    "template02.png": 0.6,
    "template03.png": 0.6,
    "template04.png": 0.6,
    "template05.png": 0.6,
    "template06.png": 0.4,
}


# Función de Non-Maximum Suppression
def non_max_suppression(rects, scores, overlapThresh):
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
        order = order[np.where(overlap <= overlapThresh)[0] + 1]

    return keep


# Procesar cada template
template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for template_file in template_files:
    template_path = os.path.join(templates_dir, template_file)
    template_color = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

    if template_gray is None:
        print(f"Advertencia: No se pudo cargar el template {template_file}")
        continue

    h, w = template_gray.shape
    threshold = template_thresholds.get(template_file, 0.8)

    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    # Guardar rectángulos y puntuaciones para NMS
    rects = []
    scores = []

    for pt in zip(*loc[::-1]):
        rects.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        scores.append(result[pt[1], pt[0]])

    # Aplicar Non-Maximum Suppression
    keep_indices = non_max_suppression(rects, scores, overlapThresh=0.3)

    for i in keep_indices:
        x1, y1, x2, y2 = rects[i]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img_draw, template_file, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"{template_file}: {len(keep_indices)} detecciones válidas")

# Redimensionar para mostrar
scale_percent = 25
width = int(img_draw.shape[1] * scale_percent / 100)
height = int(img_draw.shape[0] * scale_percent / 100)
resized = cv2.resize(img_draw, (width, height), interpolation=cv2.INTER_AREA)

# Mostrar y guardar
cv2.imshow("Detecciones (sin solapamientos)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
