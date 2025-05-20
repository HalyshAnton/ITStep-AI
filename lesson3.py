# шуми та згортка
import cv2
import numpy as np

img = cv2.imread("data/lesson3/castello_noised.png")


# згортка
# математика - 180 балів - 60%
# укр мова -- 170 балів -- 20%
# анг мова -- 190 балів -- 20%

# ядро згортки(фільтр згортки)
# масив з коефіцієнтами для пікселів

# kernel = np.array([[0.05, 0.1, 0.05],
#                    [0.1,  0.4,  0.1],
#                    [0.05, 0.1, 0.05]])
#
# new_img = cv2.filter2D(img,  # оригільне зображення
#                        -1,  # згортка для кожного кольору
#                        kernel  # ядро з коефіцієнтами
#                        )
#
# cv2.imshow("result", new_img)

# гаусове розмиття

# new_img1 = cv2.GaussianBlur(img,  # оригільне зображення
#                            ksize=(3, 3), # розмір ядра\фільра\рамки
#                            sigmaX=1  # чим більше тим сильніше розмиття
#                            )
#
# cv2.imshow("result1", new_img1)
#
# new_img100 = cv2.GaussianBlur(img,  # оригільне зображення
#                            ksize=(3, 3), # розмір ядра\фільра\рамки
#                            sigmaX=100  # чим більше тим сильніше розмиття
#                            )
#
# cv2.imshow("result100", new_img100)
# cv2.imshow("original", img)

# # фільтри для різних sigmaX
# kernel1D = cv2.getGaussianKernel(3, sigma=0)
# kernel2D = kernel1D @ kernel1D.T
#
# print(kernel2D)

# двосторонній фільтр

# new_img = cv2.bilateralFilter(img,  # оригільне зображення
#                               d=5,  # розмір ядра\фільра\рамки
#                               sigmaColor=75,  # впливає на коефіцієнт за кольором
#                               sigmaSpace=75,  # вплива на коефіцієнти як в гауса
#                               )
#
# cv2.imshow("bilateral", new_img)
# cv2.imshow("original", img)
# cv2.waitKey(0)


# бінарізація(звичайна)

img = cv2.imread("data/lesson3/sudoku.jpg")

# перевести в чорнобілий формат
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# обираємо поріг
threshold = 70

# всі пікселі які менше зробити 0
# всі пікселі які більше зробити 255

result = gray.copy()

mask = gray > threshold

result[mask] = 255
result[~mask] = 0


# бінарізація(адаптивна)
result2 = cv2.adaptiveThreshold(gray,  # оригільне зображення(чорнобіле)
                                255, # інтенсивність пікселів білого кольору
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # алгоритм як рахувати threshold
                                cv2.THRESH_BINARY,  # тип бінарізації
                                11,  # розмір ядра\фільра\рамки
                                2,  # наскільки сильною є бінарізацію
                                )

cv2.imshow("original", img)
cv2.imshow("gray", gray)
cv2.imshow("result", result)
cv2.imshow("result2", result2)
cv2.waitKey(0)