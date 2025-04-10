import sys
from components.manager import CapacitorDetectionManager

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Debes especificar la ruta de la imagen.")
        print("Uso: python main.py <ruta_imagen> [ruta_templates]")
        sys.exit(1)

    image_path = sys.argv[1]
    templates_path = sys.argv[2] if len(sys.argv) > 2 else "templates/"

    manager = CapacitorDetectionManager(image_path, templates_path)
    manager.run()
