import cv2
import numpy as np
import os

sz = (640, 480)
fps = 15
videoWriter = cv2.VideoWriter("D:/3_13/yolov5-5.x-annotations-main/yolov5-5.x-annotations-main/data/images/out3.mp4",
                              -1, fps, sz)
reader_video = cv2.VideoCapture("D:/3_9/ai_out2/car001__02_56_09.mp4")
count = 0
# 读取图片
while reader_video.isOpened():
    retval, img = reader_video.read()
    quad = [(0, 0), (640, 0), (640, 225), (320, 195), (0, 225)]

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([quad], dtype=np.int32)

    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_img = cv2.bitwise_and(img, mask)
    img[np.where(masked_img > 0)] = 0

    videoWriter.write(img)

    cv2.imshow("image", img)
    cv2.waitKey(1)


videoWriter.release()
reader_video.release()