# # дз завдання 3
# import numpy as np
#
# nums = np.array([1, 2, 3], dtype=np.uint8)
#
# # намагається помістити результат(типу float64) у ті самі
# # комірки(типу uint8) що не можливо -- error
# # nums *= 0.2
#
# # спочатку запускається nums * 0.2 -- створюється новий масив
# # далі запускається nums = ... -- змінюється вказівник
# nums = nums * 0.2
#
# print(nums)
# print(nums.dtype)
# import numpy as np
#
# # uint8 -- 0-255
#
# nums = np.array([2], dtype=np.uint8)
#
# res = nums - 10
# print(res)


# # # зображення у opencv
# import cv2
#
# # читання злображення
# img = cv2.imread(
#     'data/lesson1/cameraman.png',  # шлях до файлу
#     cv2.IMREAD_GRAYSCALE # формат зображення
# )
#
#
# print(type(img))
# print(img)
# print(img.shape)
# print(img.dtype)
#
# # показати зображення
# cv2.imshow(
#     'image',   # назву
#     img
# )
#
#
# # зміна розміру зображення
# new_img = cv2.resize(img, (500, 500))
#
# # зміна у відсотках(на 50%)
# new_img = cv2.resize(img, None, fx=1.5, fy=1.5)
#
# cv2.imshow('resized_image', new_img)
# # програма чекає поки буде натиснута будь-яка кнопка
# cv2.waitKey(0)
# print('End')


# import utils
#
# utils.lesson1_image()

import cv2
import numpy as np

# читання злображення
img = cv2.imread(
    'data/lesson1/cameraman.png',  # шлях до файлу
    cv2.IMREAD_GRAYSCALE # формат зображення
)

img = cv2.resize(img, (500, 500))

# збільшення значення пікселів

cv2.imshow('original', img)

new_img = img.astype(np.int16)
new_img -= 80

# пікселів які опинились за межами діапазону 0-255
# треба повернути назад

# mask_255 = new_img > 255
# new_img[mask_255] = 255
# mask_0 = new_img < 0
# new_img[mask_0] = 0

# # те саме
# new_img = np.clip(new_img, 0, 255)
#
# new_img = new_img.astype(np.uint8)
#
# cv2.imshow('new', new_img)

# частина зображення з 200 по 400 рядок
segment = img[200:400]  # ті самі пікселі що і в img

cv2.imshow('segment', segment)

segment += 80

cv2.imshow('original', img)
cv2.imshow('segment', segment)

cv2.waitKey(0)