import cv2
import numpy as np
import os

# Rutas
img_path = "../img/rec14-1.jpg"
templates_dir = "./templates"

# Cargar imagen principal en color
img = cv2.imread(img_path)
img_draw = img.copy()  # copia para dibujar encima

# Thresholds por template (puedes ajustarlos)
template_thresholds = {
    "template01.png": 0.6,
    "template02.png": 0.6,
    "template03.png": 0.6,
    "template04.png": 0.6,
}

# Listar todos los archivos de template en la carpeta
template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Procesar cada template
for template_file in template_files:
    template_path = os.path.join(templates_dir, template_file)
    template = cv2.imread(template_path)

    if template is None:
        print(f"Advertencia: No se pudo cargar el template {template_file}")
        continue

    h, w, _ = template.shape

    # Usar threshold personalizado o uno por defecto
    threshold = template_thresholds.get(template_file, 0.8)

    # Template matching
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    # Dibujar coincidencias
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_draw, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        cv2.putText(img_draw, template_file, (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"{template_file}: {len(list(zip(*loc[::-1])))} coincidencias")

# Redimensionar para mostrar
scale_percent = 25
width = int(img_draw.shape[1] * scale_percent / 100)
height = int(img_draw.shape[0] * scale_percent / 100)
resized = cv2.resize(img_draw, (width, height), interpolation=cv2.INTER_AREA)

# Mostrar y guardar
cv2.imshow("Detecciones", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
