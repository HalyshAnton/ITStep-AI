# Курс: AI+Python
# Модуль 12. Аналіз даних
# Тема: Стеки. Частина 2
# Завдання 1
# Відкрийте зображення data/lesson3/notes.png.
# Проведіть наступні дії:
#  проведіть бінарізацію(звичайну та адаптивну)
#  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та sigmaX 0, 2, 10
#  повторіть бінарізацію, але перед тим застосуйте bilateral filter

import cv2
import numpy as np
from sympy.physics.units.definitions.unit_definitions import gauss

# img = cv2.imread('data/lesson3/notes.png')
#
# img = cv2.resize(img, (600, 600))
# cv2.imshow('original', img)
#
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grey_image', gray_image)
# #
# # threshold = 128
# #
# # mask = gray_image < threshold
# #
# # gray_image[mask] = 0
# # gray_image[~mask] = 255
# # cv2.imshow('grey_image', grey_image)
#
#
#
# # ADAPTIVE
# res = cv2.adaptiveThreshold(
#     gauss,
#         255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,
#     2,
# )
# cv2.imshow('res', res)



# Завдання 2
# Відкрийте зображення data/lesson3/sudoku.jpg. Проведіть
# для нього бінарізацію, а саме
#  CLAHE
#  гаусове розмиття
#  адаптивна бінарізація
#  NLMean
# Самостійно підберіть параметри, збережіть результат.
# Порівняйте результати для гаусової та середньої адаптивної бінарізації

sudoku_image = cv2.imread('data/lesson3/sudoku.jpg')
cv2.imshow('sudoku', sudoku_image)

gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
result = clahe.apply(gray)


gauss = cv2.GaussianBlur(gray, (3, 3), 1.5)
cv2.imshow('gauss', gauss)

res = cv2.adaptiveThreshold(
    result,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)
cv2.imshow('Adaptive Gauss', res)

result_gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
cv2.imshow('NLMean', result_gray)



# Завдання 3
# Використовуючи utils.trackbar_decorator Побудуйте class
# ThresholdingParameterSelector для підбору параметрів для
# адаптивної бінарізації.
# Методи:
# o run(img, **kwargs) – головний метод який запускає застосовує усі перетворення до
# зображення і повертає маску.
# Решту парметрів підберіть самостійно
# Можливі методи:
# o _apply_blur(img, ksize, sigmaX)
# o _apply_bilateral(img, d, sigmaS, sigmaC)
# o _apply_threshold(img, ksize, C)
# o _apply_denoising(img, h, search_size, tamplate_size)

cv2.waitKey(0)

cv2.destroyAllWindows()