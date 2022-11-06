# Написать функцию, которая возвращает тензор представляющий изображение круга с заданным цветом
# и радиусом в схеме rgd на черном фоне

# A function returns a tensor that represents a circle image with selected color in RGB profile on black background

import numpy as np
import math

def circle(radius, r, g, b):
    radius = int(radius)
    SIZE = 2 * radius + 1 + 2

    canvas = np.zeros((SIZE, SIZE, 3), dtype=int)

    center = int(SIZE / 2)

    for angle in np.linspace(0, 2 * math.pi, 100):
        x = center + round(radius * math.cos(angle))
        y = center + round(radius * math.sin(angle))

        canvas[x][y] = np.array([r, g, b])

    return canvas