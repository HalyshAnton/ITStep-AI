import ultralytics
import cv2

# дістати перший кадр з відео
cap = cv2.VideoCapture("data/lesson8/cars.mp4")

ret, img = cap.read()
# =========

# використання моделі
model = ultralytics.YOLO("yolov8n.pt")

results = model.predict(
    img,
    conf=0.25,  # мінімальний відсоток з яким можна визначити об'єкт
    iou=0.75,    # якщо для двох рамок iuo менший за це число, то вважаємо що це 2 різних об'єкта
    #classes=[0, 2]  # індекси класів(тип об'єктів) які будуть детектитись
)
# results -- список з результами на кожне зображення в predict

# результати для першого зображення
result = results[0]

#print(result)

# отримати зображення з результатами детекції
result_img = result.plot()

# словник з назвами класів
names = result.names

# рамки
boxes = result.boxes

#print(boxes)

# візуалізація
cv2.imshow('original', img)
cv2.imshow('result', result_img)
cv2.waitKey(0)


# отримання рамок для об'єктів

for cls, xyxy in zip(boxes.cls, boxes.xyxy):
    # print(cls, xyxy)

    # перевести все в int
    cls = int(cls)
    x1, y1, x2, y2 = map(int, xyxy)

    #print(cls, x1, y1, x2, y2)

    # отримати назву об'єкти
    cls_name = names[cls]

    # вирізати рамку з зображення
    # Region of Interest -- область яка нас цікавить
    roi = img[y1:y2, x1:x2]

    # візуалізація
    cv2.imshow(cls_name, roi)

cv2.waitKey(0)
