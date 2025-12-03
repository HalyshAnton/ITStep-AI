import cv2
import ultralytics

model = ultralytics.YOLO('yolo11s-pose.pt')

img = cv2.imread('data/lesson_pose/human.jpg')

results = model.predict(img)
result = results[0]

res_img = result.plot()

# ключові точки
keypoints = result.keypoints

# ймовірності для кожної точки
conf = keypoints.conf

# print(conf)
# print(conf.shape)  # (кількість людей, кількість точок(17))

# координати xy
xy = keypoints.xy

# print(xy)
# print(xy.shape) # (кількість людей, кількість точок(17), координати)

# координати правої долоні
xy_right_hand = xy[0, 10]  # людина 0, точка 10
xy_right_hand = xy_right_hand.cpu()  # відключити від графічного процесора
xy_right_hand = xy_right_hand.numpy()  # перевести у звичайний масив

x, y = xy_right_hand

# треба перевести в int
x = int(x)
y = int(y)

# print(x)
# print(y)

# намалювати коло на зображення
cv2.circle(
    img,   # зображення де малювати коло
    (x, y),     # координати центру
    20,         # радіус кола
    (255, 0, 0),  # колір у bgr(тут синій)
    -1                 # товщина лінії(-1 означає повністю заповнене коло)
)

# накласти текст на зображення
cv2.putText(
    img,                  # зображення
    'Right hand',    # текст
    (x+30, y-30),     # нижня ліва точка початку тексту
    cv2.FONT_HERSHEY_SIMPLEX,    # шрифт
    0.8,          # розмір шрифту(відсоток до стандарту)
    (0, 0, 0),      # колір у bgr(тут чорний)
    2           # товщина ліній

)

xy = xy.cpu().numpy()
# ліва стопа
x_left_foot, y_left_foot = xy[0, 15]

# праве плече
x_right_shoulder, y_right_shoulder = xy[0, 6]

# чи справді праве плече знаходиться правіше за ліву стопу
if x_right_shoulder > x_left_foot:
    print("праве плече знаходиться правіше за ліву стопу")
else:
    print("праве плече знаходиться лівіше за ліву стопу")

# чи справді праве плече знаходиться вище за ліву стопу
if y_right_shoulder < y_left_foot:
    print("праве плече знаходиться вище за ліву стопу")
else:
    print("праве плече знаходиться нижче за ліву стопу")

cv2.imshow('original', img)
cv2.imshow('result', res_img)
cv2.waitKey(0)