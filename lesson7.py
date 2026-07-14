# сегментація зображень
from ultralytics import YOLO
import numpy as np
import cv2

# модель для сегментації
model = YOLO("yolo11s-seg.pt")

# img = cv2.imread("data/lesson_seg/human.jpg")
# cv2.imshow("orig", img)
# print(img.shape)
#
# # # застосування моделі
# results = model.predict(
#     img,
#     device="cpu",
# )
# # results -- список результатів для кожного зображення
#
# # # дістати результати для першого(єдиного) зображення
# result = results[0]
# print(result)
#
#
# # # візуалізація результату
# res_img = result.plot()
# cv2.imshow("res", res_img)
#
#
# # # маски об'єктів
# masks = result.masks
#
# # print(masks)
#
#
# # # класи об'єктів
# boxes = result.boxes
# print(boxes)
#
#
# # # назви класів
# print(boxes.cls)
#
#
# # # де знаходиться людина
# masks_data = masks.data
# # # маска людини
#
# human_mask = masks_data[0]
#
# # # переведення маски у формат opencv
# human_mask = human_mask.cpu().numpy()
# print(human_mask.shape)  # розмір не співпадає з оригінальним зображеннямр
# print(human_mask.dtype)
#
# # перевести в тип даних uint8
# human_mask = human_mask.astype(np.uint8)
#
# # там 0 та 1, мають бути 0 та 255
# human_mask *= 255
#
# # змінити розмір до оригінального
# human_mask = cv2.resize(human_mask, (600, 400))
#
# cv2.imshow("mask", human_mask)


# відео
cap = cv2.VideoCapture("data/lesson8/cars+bikes.mp4")  # відеокамера


# зображення фону
background = cv2.imread("data/lesson4/canal.png")

# отримати розміри зображення
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # фон має бути того ж розміру що і зображення на відео
background = cv2.resize(background, (width, height))

while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("orig", frame)

    # модель
    results = model.predict(
        frame,
        device="cpu"
    )

    result = results[0]
    res = result.plot()

    # дістати маску
    masks = result.masks
    masks_data = masks.data

    # маска об'єкта з найвищою ймовірністю(людина в кадрі)
    human_mask = masks_data[0]

    # обробка маски(щоб намалюти)
    human_mask = human_mask.cpu().numpy()
    human_mask = human_mask.astype(np.uint8)
    human_mask *= 255
    human_mask = cv2.resize(human_mask, (width, height))

    cv2.imshow("mask", human_mask)

    # заміна фону

    # маска має місти True або False
    mask = human_mask.astype(bool)

    #frame[~mask] = background[~mask]

    # розмиття фону
    blured_frame = cv2.GaussianBlur(frame, (51, 51), sigmaX=10)
    frame[~mask] = blured_frame[~mask]

    cv2.imshow("with background", frame)

    cv2.imshow("res", res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



cv2.waitKey(0)