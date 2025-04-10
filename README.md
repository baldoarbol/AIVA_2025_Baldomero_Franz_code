# SOFTWARE PONDERADOR DE EMBEBIDOS Y DETECTOR DE CAPACITORES

## Autores:
**Baldomero Rodríguez Árbol**  
**Franz Jesús Israel Herrera Cervellón**  

### Aplicaciones Industriales y Comerciales  
Máster Universitario en Visión Artificial  
Universidad Rey Juan Carlos – Curso 2024/2025

---

## Descripción

Este software permite detectar y clasificar capacitores en imágenes de placas base mediante técnicas de template matching. Está diseñado para estimar automáticamente el beneficio económico de los componentes detectados, diferenciando entre capacitores grandes y pequeños.

---

## Requisitos del sistema

- **Python 3.10** (recomendado)
- Sistema operativo Windows, Linux o macOS
- Bibliotecas necesarias incluidas en `requirements.txt`

---

## Instalación

1. Clonar el repositorio en tu máquina local:
```sh
git clone https://github.com/baldoarbol/AIVA_2025_Baldomero_Franz_code.git
cd AIVA_2025_Baldomero_Franz_code
```

2. Crear un entorno virtual (opcional pero recomendado):
```sh
python -m venv venv
venv\Scripts\activate  # En Windows
source venv/bin/activate  # En Linux/macOS
```

3. Instalar las dependencias del proyecto:
```sh
pip install -r requirements.txt
```

4. Establecer la variable de entorno `PYTHONPATH` desde el directorio raíz del proyecto:
```sh
$env:PYTHONPATH = "."   # En PowerShell (Windows)
export PYTHONPATH=.     # En Bash (Linux/macOS)
```

---

## Ejecución del programa

Para ejecutar el sistema sobre una imagen, utilizar:

```sh
python main.py img/rec1-1.jpg
```

Donde `img/rec1-1.jpg` es la ruta a una imagen de placa base. Puedes sustituirla por cualquier otra imagen incluida en la carpeta `/img`.

> ⚠️ Solo se adjunta una imagen de ejemplo. Para hacer pruebas completas, es necesario añadir más imágenes a la carpeta `img/`.

El resultado se mostrará en pantalla y se guardará automáticamente en la carpeta `output`, que se creará si no existe.

---

## Ejecución de Tests

Para ejecutar los tests, utilizar:

```sh
pytest ./tests
```

Como parte de los tests, se mostrará la imgen con los condensadores reales etiquetados en verde (imagen izquierda) junto con los capacitores detectados en rojo (imagen derecha).

---

## Estructura del proyecto

```

AIVA_2025_Baldomero_Franz_code/
│
├── components/
│   ├── capacitor_detector.py
│   ├── image_preprocessor.py
│   ├── manager.py
│   └── result_generator.py
│
├── tests/
│   ├── test_capacitor_detector.py
│   ├── test_image_preprocessor.py
│   └── test_result_generator.py
│
├── img/              # Imágenes de entrada (debe añadirse manualmente)
├── labels/           # Etiquetado de transistores en imágenes (groundtruth)
├── templates/        # Templates para detección y archivo config.csv
├── output/           # Se genera automáticamente al ejecutar el programa
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Configuración de templates

Para modificar los templates utilizados en la detección o añadir nuevos tipos de capacitores:

1. Añadir la imagen del template a la carpeta `templates/`.
2. Editar el archivo `templates/config.csv` y añadir una nueva línea con el siguiente formato:

```
nombre_archivo.png,tipo
```

Ejemplo:
```
template_big_01.png,big
template_small_02.png,small
```

---

## Créditos

Desarrollado como proyecto académico para la asignatura de Aplicaciones Industriales y Comerciales del Máster en Visión Artificial (URJC).
