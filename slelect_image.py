import os, cv2
import numpy as np


base_path = 'D:\\3_13\\yolov5-5.x-annotations-main\\datasets\\mydata\\images\\val'
save_path = 'D:\\3_13\\yolov5-5.x-annotations-main\\datasets\\mydata\\images\\val2\\'

for path in os.listdir(base_path):
    image = cv2.imread(f'{base_path}\\{path}')
    img = image
    quad = [(0, 0), (640, 0), (640, 225), (320, 195), (0, 225)]

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([quad], dtype=np.int32)

    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_img = cv2.bitwise_and(img, mask)
    img[np.where(masked_img > 0)] = 0
    img_save_name = save_path + path
    cv2.imwrite(img_save_name, img)