import cv2
import numpy as np
import utils

# в opencv кольорове зображення у форматі BGR
img = cv2.imread("data/lesson4/castello.png")

# межі шукають на чорнобілому зображені
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# kernel = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]])
#
# vert = cv2.filter2D(gray, -1, kernel)
#
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]])
#
# horiz = cv2.filter2D(gray, -1, kernel)
#
# cv2.imshow("original", img)
# cv2.imshow("vertical", vert)
# cv2.imshow("horizontal", horiz)
# cv2.waitKey(0)

# пошук меж

# edged = cv2.Canny(gray,  # зображення де шукаємо межі
#                   100,  # нижня межі інтенсивності межі
#                   150   # верхня межі інтенсивності межі
#                   )
#
# cv2.imshow("original", img)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

# функція для меж

# @utils.trackbar_decorator(lower=(0, 255), upper=(0, 255))
# def func(img, lower, upper):
#     # перетворення в чорнобіле зображення
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # розмити зображення
#     gray = cv2.GaussianBlur(gray,
#                             (5, 5),
#                             sigmaX=2)
#
#     # алгоритм Canny(пошук меж)
#     edged = cv2.Canny(gray, lower, upper)
#
#     return edged
#
# func(img)


img = cv2.imread("data/lesson4/j.png", cv2.IMREAD_GRAYSCALE)

# ерозія
# якщо навколо пікселя є хоча б один чорний -- то піксель стає чорним

# піксель по сусідству -- в сежах квадрату 3х3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded = cv2.erode(img, kernel)

# dilate(розширення?)
# якщо навколо пікселя є хоча б один білий -- то піксель стає білим
dilated = cv2.dilate(img, kernel)


both = cv2.erode(img, kernel)
both = cv2.dilate(both, kernel, iterations=2)

cv2.imshow("original", img)
cv2.imshow("eroded", eroded)
cv2.imshow("dilate", dilated)
cv2.imshow("both", both)
cv2.waitKey(0)

