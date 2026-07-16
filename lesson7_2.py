# from ultralytics import YOLO
# import numpy as np
# import cv2
#
# # модель для сегментації
# model = YOLO("yolo11s-seg.pt")
#
# img = cv2.imread("data/lesson_seg/human.jpg")
#
# cv2.imshow("orig", img)
#
#
# results = model.predict(
#     img,
#     device="cuda"
# )
# result = results[0]
#
# res = result.plot()
# cv2.imshow("result", res)
#
#
# print(result)
#
#
# masks = result.masks
# print(masks)
#
#
# masks_data = masks.data
# masks_data = masks_data.cpu().numpy()
#
#
# # маска третього об'єкта
# mask3 = masks_data[2]
#
# # зміна розміру до оригінального
# height, width, colors = img.shape
#
# mask3 = cv2.resize(mask3, (width, height))
#
# # зміна типів даних
#
# mask3_bool = mask3.astype(bool)
#
#
# mask3_uint = mask3.astype(np.uint8)
# mask3_uint *= 255
#
#
# cv2.imshow("mask", mask3_uint)
#
# # все що не відповідає масці замінити на 0
# img[~mask3_bool] = 0
# cv2.imshow("with mask", img)
#
#
# cv2.waitKey(0)
