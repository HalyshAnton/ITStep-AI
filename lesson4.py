# згортка

#ЗНО(НМТ)
# математика -- 180
# укр мова -- 175
# англ мова -- 190

# коефіцієнти( важливість) -- вагами важливості
# математика -- 60%
# укр мова -- 10%
# англ мова -- 30%

# оцінку -- (180 + 175 + 190) / 3  середнє арифметичне
# оцінку -- 180*0.6 + 175*0.1 + 190*0.3 -- середнє зважене
# grades = [180, 175, 190]
# coefs = [0.6, 0.1, 0.3]

# згортка
import cv2
import utils
import numpy as np

img = cv2.imread('data/lesson3/castello_noised.png')

# # ядро згортки(фільтр, масив з коефіцієнтами)
# kernel = np.array(
#     [[0.1, 0.1, 0.1],
#      [0.1, 0.1, 0.1],
#      [0.1, 0.1, 0.1]]
# )
#
#
#
# kernel = np.array(
#     [[-1, -2, -1],
#      [0., 0., 0.],
#      [1, 2, 1]]
# )
#
# kernel = np.array(
#     [[-1, -2, -1],
#      [0., 0., 0.],
#      [1, 2, 1]]
# )
# res = cv2.filter2D(img, -1, kernel)
#
# cv2.imshow('res1', res)
#
# kernel = np.array(
#     [[-1, 0, 1],
#      [-2, 0., 2],
#      [-1, 0, 1]]
# )
#
#
# # згортка
# res = cv2.filter2D(img, -1, kernel)
#
# cv2.imshow('res', res)

# застосування -- усунення шуму
#img = utils.add_salt_and_pepper_noise(img, 0.001, 0.001)
img = utils.add_gaussian_noise(img, 0, 5)

# гаусове розмиття
res = cv2.GaussianBlur(
    img,
    (13, 13),   # розмір ядра
    2       # чим більше тим більше розвиття
)

# двосторонній фільтр
res = cv2.bilateralFilter(
    img,
    d=9,  # розмір ядра
    sigmaColor=75,   # наскільки зберігати різкість кольору
    sigmaSpace=75,   # те ж саме що й в GaussianBlur
)

cv2.imshow('res', res)
cv2.imshow('original', img)
cv2.waitKey(0)