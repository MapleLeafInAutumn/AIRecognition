import cv2
cap = cv2.VideoCapture('rtsp://47.243.7.221/car002')
from datetime import datetime
import time

height = 480
width = 640
framesize = height * width * 3 // 2
h_h = height // 2
h_w = width // 2

while True:
    ret, frame = cap.read()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
    cv2.putText(frame, dt_string, (210, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                cv2.LINE_AA)

    cv2.imshow('frame', frame)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
