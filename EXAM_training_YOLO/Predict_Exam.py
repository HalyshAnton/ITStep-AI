import cv2
from ultralytics import YOLO

# 1. Завантажуємо скачану модель
model = YOLO(r"D:\IT_STEP_Academy\Model_Exam\weights\best.pt")

# 2. Шлях до будь-якого тестового фото на вашому ПК
image_path = r"D:\IT_STEP_Academy\EXAM\train\images\8df07836-4606-446e-9880-6ed9e0f74543_0_0_jpg.rf.2a5e0f0ba0301eb86af1169701a3e5a6.jpg"

# 3. Запуск детекції
results = model.predict(source=image_path, conf=0.25)

# 4. Показ результату у вікні OpenCV
res_img = results[0].plot()
cv2.imshow("YOLO Exam Result", res_img)

print("Детекцію успішно завершено! Результати збережено в папці runs/detect/predict.")
cv2.waitKey(0)
cv2.destroyAllWindows()