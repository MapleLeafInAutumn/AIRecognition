import os, cv2
import numpy as np


base_path = 'F:\\python_code\\3_23\\datasets\\mydata\\images\\val'
save_path = 'F:\\python_code\\3_23\\datasets\\mydata\\images\\val4_6\\'

for path in os.listdir(base_path):
    image = cv2.imread(f'{base_path}\\{path}')
    img = image
    b, g, r = cv2.split(img)
    if np.sum(np.abs(b-g)) < 100:
        continue
    if np.sum(np.abs(g-r)) < 100:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    img_save_name = save_path + path
    cv2.imwrite(img_save_name, img)