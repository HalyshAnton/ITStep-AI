import cv2
import ultralytics

model = ultralytics.YOLO('yolo11n-pose.pt')

img = cv2.imread('data/lesson_pose/human.jpg')

results = model.predict(img)

res = results[0]

# result_img = res.plot()
#
# # вивід результатів
# print(res)
# print(res.keypoints)  # ключові точки
#
# cv2.imshow("result", result_img)
# cv2.waitKey(0)

# координати точок
xy_coords = res.keypoints.xy

# дістати координати точок для першого об'єктів
xy_coords = xy_coords[0]

# змінити масив на numpy
xy_coords = xy_coords.numpy()

# змінити тип даних на int
xy_coords = xy_coords.astype(int)

# координати правої долоні
x, y = xy_coords[10]

# намалювати круг в даній точці
# res_img = cv2.circle(
#     img,  # зображення на якому намалювати коло
#     (x, y),   # координати центру кола
#     5,       # радіус у пікселях
#     (0, 255, 0),  # колір у bgr форматі(тут -- зелений)
#     -1      # товщина лінії(-1 -- запоанити коло повністю)
# )
#
# cv2.imshow("right arm", res_img)
# cv2.waitKey(0)


# # перевиріти що права долоня вища за праве коліно
#
# # коорлинати долоні
# x_arm, y_arm = xy_coords[10]
#
# # коордлинати коліна
# x_knee, y_knee = xy_coords[14]
#
# # вивід коорлинати
# print(f"Координати долоні -- {x_arm}, {y_arm}")
# print(f"Координати коліна -- {x_knee}, {y_knee}")
#
# # перевірка
# if y_arm < y_knee:
#     print("Долоня вище зо коліно")
# else:
#     print("Долоня нижче зо коліно")

# перевірити чи правий плече знаходиться правіше за ліву плече

x_right, y_right = xy_coords[6]
x_left, y_left = xy_coords[5]

# вивід коорлинати
print(f"Координати лівого плеча  -- {x_left}, {y_left}")
print(f"Координати правого плеча -- {x_right}, {y_right}")

# перевірка
if x_right > x_left:
    text = "human back"  # людина повернута спиною
else:
    text = "human face"  # людина повернута до вас

# нанести текст на зображення

img = cv2.putText(
    img,  # зображення на яке насти текст
    text,   # сам текст
    (50, 350),   # координати тексту
    cv2.FONT_HERSHEY_SIMPLEX,   # шрифт
    1.5,   # розмір шрифту
    (255, 255, 255),   # колір у форматі bgr(тут -- білий)
    2   # товщина лінії
)

cv2.imshow("", img)
cv2.waitKey(0)


