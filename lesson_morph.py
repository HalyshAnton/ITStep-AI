import cv2



img = cv2.imread("data/lesson4/j.png", cv2.IMREAD_GRAYSCALE)


cv2.imshow("orig", img)


# морфологічні опереати(очистка шуму на бінарних зображеннях)

# якщо в рамці є хоча б 1 білий піксель то робимо його білим
dilat = cv2.dilate(
    img,
    (3, 3),
    iterations=2  # скільки разів застосувати
)

# якщо в рамці є хоча б 1 чорний піксель то робимо його чорним
erode = cv2.erode(
    img,
    (3, 3),
    iterations=2  # скільки разів застосувати
)

# обидва
res = cv2.dilate(img, (3, 3), iterations=1)
res = cv2.erode(res, (3, 3), iterations=4)

cv2.imshow("dilat", dilat)
cv2.imshow("erode", erode)
cv2.imshow("res", res)
cv2.waitKey(0)