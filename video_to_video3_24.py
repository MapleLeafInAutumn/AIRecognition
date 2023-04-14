import cv2
import numpy as np
import os

sz = (640, 640)
fps = 15


reader_name = "D:\\videos\\4_3\\car001__208_38_44.avi"
writer_name = "D:\\videos\\4_3_out\\car001__208_38_44.mp4"
reader_video = cv2.VideoCapture(reader_name)
videoWriter = cv2.VideoWriter(writer_name, -1, fps, sz)
kernel = np.ones((7, 7), np.uint8)
while reader_video.isOpened():
    retval, img = reader_video.read()
    img_crop = img[280:420, 100:540, :]
    b, g, r = cv2.split(img_crop)
    max_b = np.max(b)
    max_r = np.max(r)
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    ret, binary = cv2.threshold(b, max_b*0.85, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, kernel)

    ret2, binary2 = cv2.threshold(r, mean_r, 255, cv2.THRESH_BINARY)
    binary2 = cv2.dilate(binary2, kernel)
    binary3 = cv2.bitwise_and(binary2, binary)
    binary[binary3 > 0] = 0
    binary2[binary3 > 0] = 0
    b[binary > 0] = mean_g
    r[binary2 > 0] = mean_g
    img_crop[:, :, 0] = b
    img_crop[:, :, 2] = r

    img[280:420, 100:540, :] = img_crop
    videoWriter.write(img)
    cv2.imshow("image", img)
    cv2.waitKey(1)
reader_video.release()
videoWriter.release()
cv2.destroyAllWindows()
