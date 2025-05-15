import cv2
import utils

img = cv2.imread("data/lesson2/marbles.png")
# # кольорове зображення у форматі bgr
#
# # print(img)
# #
# # print(img.shape)
# # print(img.dtype)
# #
# # blue = img[:, :, 0]
# # green = img[:, :, 1]
# # red = img[:, :, 2]
# #
# # # # замінити червоний та залений на 0
# # # img[:, :, 1] = 0
# # # img[:, :, 2] = 0
# #
# # # замінити синій на 0
# # img[:, :, 0] = 0
#
#
#
# cv2.imshow("image", img)
# cv2.waitKey(0)


img_bgr = cv2.imread("data/lesson2/lego.jpg")

# конвертація в HSV
img_hsv = cv2.cvtColor(img_bgr,
                       cv2.COLOR_BGR2HSV
                       )

# print(img_hsv)
#
# print(img_hsv.shape)
# print(img_hsv.dtype)

# utils.lesson2_hsv_range(img_bgr)


# кольорова сегментація

# # конвертація в HSV
# img_hsv = cv2.cvtColor(img_bgr,
#                        cv2.COLOR_BGR2HSV
#                        )
#
# # межі червоного кольору
# lower = (0, 150, 180)
# upper = (10, 255, 255)
#
# # кольорова маска
# mask = cv2.inRange(img_hsv, lower, upper)
#
# print(mask)
# print(mask.shape)
# print(mask.dtype)
#
# cv2.imshow("mask", mask)
# cv2.imshow("original", img_bgr)
#
# cv2.waitKey(0)


# # підбір параметрів
# import utils
#
#
# @utils.trackbar_decorator(min_h=(0, 180), min_s=(0, 255), min_v=(0, 255),
#                           max_h=(0, 180), max_s=(0, 255), max_v=(0, 255))
# def func(img, min_h, min_s, min_v, max_h, max_s, max_v):
#     lower = (min_h, min_s, min_v)
#     upper = (max_h, max_s, max_v)
#
#     # конвертацію в hsv
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # маска
#     mask = cv2.inRange(hsv, lower, upper)
#
#     return mask
#
#
# img = cv2.imread("data/lesson2/lego.jpg")
# func(img)

# img = cv2.imread("data/lesson2/bio_low_contrast.jpg", cv2.IMREAD_GRAYSCALE)
#
# cv2.imshow("original", img)
#
# # вирівнювання гістограм
# new_img = cv2.equalizeHist(img)
#
# cv2.imshow("eqhist", new_img)

# з кольоровим
img = cv2.imread("data/lesson2/cell.png")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# канал l(luminance) -- яскравість

lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])

new_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow("original", img)
cv2.imshow("eq", new_img)

cv2.waitKey(0)
