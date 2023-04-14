import os, cv2
import numpy as np

base_path = 'E:\\my_code\\python_code\\23_3_21\\yolov5-5.x-annotations-main\\datasets\\mydata\\images\\train'
save_path = 'E:\\my_code\\python_code\\23_3_21\\yolov5-5.x-annotations-main\\datasets\\mydata\\images\\train3_23\\'

for path in os.listdir(base_path):
    image = cv2.imread(f'{base_path}\\{path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray
    img_save_name = save_path + path
    cv2.imwrite(img_save_name, image)