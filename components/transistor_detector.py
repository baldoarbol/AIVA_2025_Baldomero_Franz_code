"""
Script con las funciones para detectar transistores en una imagen preprocesada.
"""


def detect_transistor(t_type: str) -> int:
    if t_type == 'big':
        result = 2
    elif t_type == 'small':
        result = 7
    else:
        result = 0
    
    return result
