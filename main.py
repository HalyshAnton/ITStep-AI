import numpy as np
import cv2


# масив
# array = np.array([[1, 2, 3], # перший рядок масиву
#                   [4, 5, 6] # другий рядок масив
#                   ])
#
# print(array)
#
# print(array.dtype)  # тип даних одного елемента
# print(array.shape)  # розмір (рядочки, стовпчики)
#
# # індексація
# print(array[0, 2])   # елемент рядок 0 та стовпчик 2
# print(array[0])      # рядок з індексом 0
# print(array[0:2])    # рядки з 0 по 2
# print(array[:, 1])   # стовпчик з індексом 1


# зображення
# читання

img = cv2.imread("data/lesson1/cameraman.png", # шлях до файлу
                 cv2.IMREAD_GRAYSCALE          # зображення чорнобіле
                 )

# print(img)
# print(img.dtype)
# print(img.shape)

# uint8 -- ціле число в діапазоні 0 до 255

# виведення
# cv2.imshow("test img",  # назва зображення
#            img)

# індексаці
segment = img[50:200]  # рядки з 50 по 200

print(segment)
print(segment.dtype)
print(segment.shape)

#cv2.imshow('segment', segment)

# збільшити всі пікселі у segment на 20
segment += 20

cv2.imshow("test img",  # назва зображення
           img)

# головний цикл
cv2.waitKey(0)