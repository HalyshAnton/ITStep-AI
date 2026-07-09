# детекція об'єктів(YOLO)

import cv2
import ultralytics
from fontTools.varLib.instancer import names

# створення моделі
# s -- small(розмір моделі)
model = ultralytics.YOLO("yolo11s.pt")

# отримати зображення з ввідео
cap = cv2.VideoCapture('data/lesson8/cars+bikes.mp4')
success, img = cap.read()
img = cv2.resize(img, None, fx=0.5, fy=0.5)
print(img.shape)

cv2.imshow("orig", img)


# застосування моделі
# модель може одночасно обробити декілька зображень [img1, img2, img3, ..]
# на виході results -- список результів [result1, result2, result3, ..]
results = model.predict(
    img,  # зображення
    device="cuda",  # процесор де робити обчисдення
                # cpu --звичайний процесор
                # cuda -- графічний процесор(gpu) на Windows\Linux
                # mps -- графічний процесор(gpu) на MacOS

    conf=0.25,   # мінімальна ймовірність для об'єктів,
                # все що менше відсіюється

    iou=0.5,   # наскільки сильно можуть перетинаться рамки,
               # якщорамки перетинаються сильніше то залишаєму ту
                # в якої більшо ймовірність

    #classes=[0, 1],  # класи які враховувати(див result.names)
)
# print(type(results))
# print(results)

# results -- список з одним елементом
# отримати результ
result = results[0]



# отримати назви класів(об'єкти які вмієме визначати модель)
names = result.names
print(type(names))
print(names)


# самі об'єкти
boxes = result.boxes
# print(type(boxes))
# print(boxes)


# візуалізація результів
res_img = result.plot()
cv2.imshow("result", res_img)

# ймовірності
conf = boxes.conf
print(type(conf))

# відклюсти від графічного процесора
conf = conf.cpu()

# перевести в масив numpy
conf = conf.numpy()

print(conf)
print(conf.shape)
print(conf.dtype)


# рамка(box)
box = boxes[0]  # дані першого обєкта

print(box.conf)
print(box.cls)  # індекс класу
print(box.xyxy)  # координати меж


# вивести назву та ймовірність
conf = box.conf
conf = conf.cpu().numpy()
print(f"Ймовірність першого обєкта {conf[0]}")

cls = box.cls
cls = cls.cpu().numpy()
print(f"Індекс класу першого обєкта {cls[0]}")

class_id = int(cls[0])
class_name = names[class_id]
print(f"Клас першого обєкта {class_name}")

# отримати перший об'єкт


# координати

# переведення координат в int



# вирізати об'єк з всього зображення


# # відео

cv2.waitKey(0)
