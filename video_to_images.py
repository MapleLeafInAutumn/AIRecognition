import cv2
import numpy as np
import os
reader_video = cv2.VideoCapture("D:/videos/4_1/car001__10_25_18.mp4")
save_path = 'D:/images/4_1/fu_gui_to_dao_cha_right_open/'
count = 0
start = 0
fps = 15
video_name = "car001__10_25_18"
# 读取图片
while reader_video.isOpened():
    retval, img = reader_video.read()
    start = 1*60*fps + 36*fps
    end = 1*60*fps + 38*fps
    count = count + 1
    if count < start:
        continue

    b, g, r = cv2.split(img)
    b2 = cv2.equalizeHist(b)
    g2 = cv2.equalizeHist(g)
    r2 = cv2.equalizeHist(r)
    image = cv2.merge([b2, g2, r2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray

    img_save_name = save_path + 'img_4_1_' + video_name + str(count) + '.jpg'
    cv2.imwrite(img_save_name, img)

    if count > end:
        break

    cv2.imshow("image", img)
    cv2.waitKey(1)

reader_video.release()