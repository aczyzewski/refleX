from math import cos, sin
import numpy as np

def bresenham_line(x1, y1, x2, y2):
    """
    Zwraca listę punktów, przez które przechodzić będzie prosta
    o zadanym początku i końcu

    Parametry
    ----------
    x1, y1, x2, y2 : int
        (x1, y1) - punkt poczatkowy
        (x2, y2) - punkt końcowy

    """
    # Zmienne pomocnicze
    d = dx = dy = ai = bi = xi = yi = 0
    x = x1
    y = y1
    points = []

    # Ustalenie kierunku rysowania
    xi = 1 if x < x2 else -1
    dx = abs(x1 - x2)

    # Ustalenie kierunku rysowania
    yi = 1 if y1 < y2 else -1
    dy = abs(y1 - y2)

    # Pierwszy piksel
    points.append((x, y))

    ai = -1 * abs(dy - dx) * 2
    bi = min(dx, dy) * 2
    d = bi - max(dx, dy)

    # Oś wiodąca OX
    if dx > dy:
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi

            points.append((x, y))

    # Oś wiodąca OY
    else:
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi

            points.append((x, y))

    return points
def sampling_vectors(img, center, num_samples, offset=0):

    """
    Example:
    ------
    >>> sampling_vectors(img, (3, 3), 4, 0)
    [[(3, 3), (4, 3), (5, 3), (6, 3)],
     [(3, 3), (3, 4), (3, 5), (3, 6)],
     [(3, 3), (2, 3), (1, 3), (0, 3)],
     [(3, 3), (3, 2), (3, 1), (3, 0)]]

    """
    x, y = center
    if not 0 < x < img.shape[1] or not 0 < y < img.shape[0]:
        return []

    angles = np.linspace(0 + offset, 360 + offset, num_samples + 1)
    radius = np.min([img.shape[1] - x - 1, x, img.shape[0] - y - 1, y])

    vectors = []
    for angle in angles:
        point = (round(x + (radius * cos(np.radians(angle)))),  round(y + (radius * sin(np.radians(angle)))))
        vectors.append(bresenham_line(*center, *point))

    return vectors[:-1]
# --- REFLEX