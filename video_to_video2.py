import cv2
import numpy as np
import os

sz = (640, 480)
fps = 15
videoWriter = cv2.VideoWriter("D:/my_temp_copy/out3_16.mp4",
                              -1, fps, sz)
reader_video = cv2.VideoCapture("D:/my_temp_copy/2023.3.4..8.47.462output_forward001.avi")
count = 0
# 读取图片
while reader_video.isOpened():
    retval, img = reader_video.read()
    start = 1*60*15 + 8*15
    end = 2*60*15 + 50*15
    count = count + 1
    if count < start:
        continue

    videoWriter.write(img)

    if count > end:
        break

    cv2.imshow("image", img)
    cv2.waitKey(1)


videoWriter.release()
reader_video.release()