# колір
import cv2
import numpy as np

# # читати як чорно біле
# image = cv2.imread("data/lesson2/lego.jpg", cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (500, 500))
#
# print(image.dtype)
# print(image.shape)
# print(image[0, 0])
#
# cv2.imshow("gray", image)
#
# # читати як кольорове
# # за замовчуванням читає як кольорове
# image = cv2.imread("data/lesson2/lego.jpg")
# image = cv2.resize(image, (500, 500))
#
# print(image.dtype)
# print(image.shape)
# print(image[0, 0])
#
# cv2.imshow("color", image)

# формат(кольорові простори)
# bgr -- blue green red

# import utils
# utils.lesson2_bgr_range()



# дістати колір з зображення
image = cv2.imread("data/lesson2/lego.jpg")
image = cv2.resize(image, (500, 500))

print(image.shape) # (рядки, стовпчики, колір)

# # отримати червоний канал(bgr)
# red = image[:, :, 2]
#
#
# # imshow сприймає як чорноюіле зображення
# print(red.shape)
# cv2.imshow("red", red)


# # правильний спосіб
# red = image.copy()
#
# # зберігаємо червоний колір, все інше в 0
#
# # синій
# red[:, :, 0] = 0
#
# # зелений
# red[:, :, 1] = 0
#
#
# print(red.shape)
# cv2.imshow("red", red)


# # збільшити червоного на 10
# image[:, :, 2] += 10
#
# cv2.imshow(",ore red", image)


# кольоровий простір hsv
# import utils
# utils.lesson2_hsv_range()
#
#
# cv2.waitKey(0)



# отримати пікселі жовтого кольору

# cv2.imshow("orig", image)

# межі для кольору в hsv

# h -- 40 - 80  # колір(кути ділимо на два)
# s -- 150 - 255  # насиченість
# v -- 150 - 255  # скравість кольору


# lower = (40, 100, 100)  # нижні межі
# upper = (80, 255, 255)  # верхні межі
#
#
# # перевести з bgr в hsv
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# # отримати маску для правильних піксесів
# mask = cv2.inRange(
#     hsv,
#     lower,
#     upper
# )
#
# cv2.imshow("mask", mask)
#
#
# cv2.waitKey(0)


# # зменшити насиченість зображення
# image = cv2.imread("data/lesson1/Lenna.png")
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# cv2.imshow("Lenna", image)  # треба bgr


# розбиваємо зображення на три канала
# h = hsv[:, :, 0]
# s = hsv[:, :, 1]
# v = hsv[:, :, 2]

# # те саме
# h, s, v = cv2.split(hsv)
#
# # зменшити насиченість зображення на 10
#
# # # враховуємо обмеження типу даних uint8
# # s = s.astype(np.int64)
# #
# # s += 30
# #
# # # перевірити межі 0 - 255
# # mask = s < 0
# # s[mask] = 0
# #
# # s = s.astype(np.uint8)
#
# # збільшити яскравість
# v = v.astype(np.int64)
#
# v += 50
#
# mask = v > 255
# v[mask] = 255
#
# v = v.astype(np.uint8)
#
# # збираємо канали назад
# new_hsv = cv2.merge((h, s, v))
#
# # показати результат
# new_image = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow("result", new_image)
#
#
#
#
#
# cv2.waitKey(0)
#
# image = cv2.imread("data/lesson2/marbles.png")
# cv2.imshow("orig", image)
#
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#
# lower = (0, 120, 150)
# upper = (10, 255, 255)
# mask_red = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow("mask", mask_red)
#
# print(mask_red.shape)
# print(mask_red.dtype)
#
#
# # перевести в bool
# mask_bool = mask_red.astype(bool)
#
# image[~mask_bool] = 0
#
# cv2.imshow("result", image)
#
# cv2.waitKey(0)
#
# lower = ()


# lab

# import utils
# utils.lesson2_lab_range()


image = cv2.imread("data/lesson2/evening2.jpg")

cv2.imshow("orig", image)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

l = l.astype(np.int64)
l += 80
l = np.clip(l, 0, 255)  # виправлення меж
l = l.astype(np.uint8)

new_lab = cv2.merge((l, a, b))

new_image = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

cv2.imshow("redult", new_image)



cv2.waitKey(0)