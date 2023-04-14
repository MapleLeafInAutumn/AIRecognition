import os, cv2
import numpy as np


base_path = 'F:\\python_code\\4_11\\datasets\\mydata\\images\\train'
save_path = 'F:\\python_code\\4_11\\datasets\\mydata\\images\\train2\\'

for path in os.listdir(base_path):
    image = cv2.imread(f'{base_path}\\{path}')
    img = image
    img[0:200, 0:640, :] = 0
    img_save_name = save_path + path
    cv2.imwrite(img_save_name, img)