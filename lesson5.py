import numpy as np
import utils
import cv2

# # відкрити відео
# cap = cv2.VideoCapture("data/lesson7/cars.mp4")
#
# #cap = cv2.VideoCapture(0)  # відео з вашої камери
#
# # отримання кадр
# ret, img = cap.read()
#
# # ret -- True/False чи вдалось отримати кадр
# print(ret)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)

# показ відое покадрово

cap = cv2.VideoCapture("data/lesson7/cars.mp4")
cap = cv2.VideoCapture(0)


# fps відео
fps = cap.get(cv2.CAP_PROP_FPS)


# збереження відео
fps = cap.get(cv2.CAP_PROP_FPS)   # чатота відео
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # ширина кадру
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # висота кадру

# перевести в тип int
width = int(width)
height = int(height)

# кодек
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    "new_video.mp4",  # назва файлу
    fourcc,   #  кодек
    fps,      # частота кадрів
    (width, height),   # розмір кадру
    isColor=False  # чи кольорове зображення
)


while True:
    # отримуєте наступний кадр
    ret, img = cap.read()

    # якщо не вдалось прочитати кадр -- цикл
    if not ret:
        break

    # обробка зображення
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 50, 100)

    # показ зображення
    cv2.imshow('video', img)
    cv2.imshow('gray', gray)
    cv2.imshow('edge', edge)

    # збереження кадру у файл
    writer.write(gray)

    # добавити waitKey
    # cv2.waitKey(1)  # чекати 1 мілісекунду
    # cv2.waitKey(int(1000 // fps))  # оригільна затримка

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# звільнення пам'яті
writer.release()
cap.release()

# закрити усі вікна
cv2.destroyAllWindows()


