#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray, Float64MultiArray

import platform
import threading
import string
import struct

import logging
import time
import datetime
from sensor_msgs.msg import NavSatFix

from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


RKNN_MODEL = "/home/firefly/yahboomcar_ws/src/ai_class/weights/best3_14.rknn"

time_str3 = time.strftime('%H_%M_%S', time.localtime(time.time()))
log = Logger("/home/firefly/data_save/ai_log/log" + time_str3 + "ai.log", level='debug')
log2 = Logger("/home/firefly/data_save/ai_log/log" + time_str3 + "hei.log", level='debug')

global ai_list
ai_list = []

global road_type_queue
global scale_factor
global correction
road_type_queue = []

gps_lat = 0
gps_lon = 0
gps_alt = 0


QUANTIZE_ON = True

BOX_THRESH = 0.45
NMS_THRESH = 0.25
IMG_SIZE = 640

CLASSES = ( 'jing_ru_dao_cha_right_open', 'jing_ru_dao_cha_left_open', 'hu_gui_li_kai', 'hu_gui_jing_ru',
         'fu_gui_to_dao_cha_right_open', 'fu_gui_to_dao_cha_left_open')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= (int(IMG_SIZE/grid_h), int(IMG_SIZE/grid_w))

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= BOX_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score* box_confidences >= BOX_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print(cl)
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def get_file(filepath):
    templist = []
    with open(filepath, "r") as f:
        for item in f.readlines():
            templist.append(item.strip())
    return templist


class Nodo(object):
    def __init__(self):
        print("begin ros")

        self.br = CvBridge()
        self.forward_or_back = 1
        self.road_type = 0
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(10)

        # subscribers
        self.subimage1 = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_img1)
        self.subimage2 = rospy.Subscriber("/camera/image_raw", Image, self.callback_img2)
        self.forward_or_back_sub = rospy.Subscriber("/forward_or_back", Int32, self.callback_forward)
        # self.road_type_pub = rospy.Publisher('/road_type', Int32, queue_size=1)
        self.sub_gps = rospy.Subscriber("/pub_gps_mag", NavSatFix, self.callback_gps)

        # Create RKNN object

    def callback_img1(self, msg):
        if self.forward_or_back == 0:
            log.logger.info("/usb_cam/image_raw {}".format(self.forward_or_back))
            image_ai2 = self.br.imgmsg_to_cv2(msg)
            b, g, r = cv2.split(image_ai2)
            b2 = cv2.equalizeHist(b)
            g2 = cv2.equalizeHist(g)
            r2 = cv2.equalizeHist(r)
            img22 = cv2.merge([b2, g2, r2])

            img_rgb = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
            img_height = img_rgb.shape[0]
            img_width = img_rgb.shape[1]
            image_process_mode = "letter_box"
            if image_process_mode == "resize":
                img33 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            elif image_process_mode == "letter_box":
                global scale_factor
                global correction
                img33, scale_factor, correction = letterbox(img_rgb)
                global ai_list
                ai_list.append(img33)
                if len(ai_list) > 3:
                    ai_list.pop(0)
            # cv2.imshow("img",img_test)
            # cv2.waitKey(30)
            # self.loop_rate.sleep()

    def callback_img2(self, msg):
        if self.forward_or_back == 1:
            log.logger.info("/camera/image_raw".format(self.forward_or_back))

            image_ai2 = self.br.imgmsg_to_cv2(msg)
            b, g, r = cv2.split(image_ai2)
            b2 = cv2.equalizeHist(b)
            g2 = cv2.equalizeHist(g)
            r2 = cv2.equalizeHist(r)
            img22 = cv2.merge([b2, g2, r2])

            img_rgb = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
            img_height = img_rgb.shape[0]
            img_width = img_rgb.shape[1]
            image_process_mode = "letter_box"
            if image_process_mode == "resize":
                img33 = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            elif image_process_mode == "letter_box":
                global scale_factor
                global correction
                img33, scale_factor, correction = letterbox(img_rgb)
                global ai_list
                ai_list.append(img33)
                if len(ai_list)>3:
                    ai_list.pop(0)
            # cv2.imshow("img",img_test2)
            # cv2.waitKey(30)
            # self.loop_rate.sleep()
            # globvar.set("image",img_test)

    def callback_forward(self, msg):
        self.forward_or_back = int(msg.data)
        log.logger.info("forward callbacl {}".format(self.forward_or_back))

    def callback_gps(self, msg):
        global gps_lat
        global gps_lon
        global gps_alt
        gps_lat = msg.data.latitude
        gps_lon = msg.data.longitude
        gps_alt = msg.data.altitude
        log.logger.info("gps_lat {}".format(gps_lat))


def func_ai2():
    global log2
    global CLASSES
    font = cv2.FONT_HERSHEY_SIMPLEX

    global road_type_queue
    image_process_mode = "letter_box"
    for i in range(1):
        road_type_queue.append(0)

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load rknn model failed!')
        exit(ret)
    print('done')

    log.logger.info("begin ai classes ")
    log.logger.debug("begin redis fail_ly")
    road_type_pub = rospy.Publisher('/road_type2', Float64MultiArray, queue_size=1)
    count_frame = 0
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    while True:
        global ai_list
        if len(ai_list) < 1:
            continue
        writer_video = None
        if count_frame == 0:
            sz = (640, 640)
            fps = 13
            time_str = time.strftime('%H_%M_%S', time.localtime(time.time()))
            save_video_path = "/home/firefly/data_save/ai__output_video/" + "car001__" + time_str + ".avi"
            writer_video = cv2.VideoWriter(save_video_path, 0x7634706d, fps, sz)

        while True:
            road_clc = []
            start_time = time.process_time()
            img_test = np.copy(ai_list[-1])

            outputs = rknn.inference(inputs=[img_test])
            # post process
            input0_data = outputs[0].transpose(0, 1, 4, 2, 3)
            input1_data = outputs[1].transpose(0, 1, 4, 2, 3)
            input2_data = outputs[2].transpose(0, 1, 4, 2, 3)

            input0_data = input0_data.reshape(*input0_data.shape[1:])
            input1_data = input1_data.reshape(*input1_data.shape[1:])
            input2_data = input2_data.reshape(*input2_data.shape[1:])

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)
            # if image_process_mode == "resize":
            #     scale_h = IMG_SIZE / img_height
            #     scale_w = IMG_SIZE / img_width
            #     boxes = resize_postprocess(boxes, scale_w, scale_h)
            # elif image_process_mode == "letter_box":
            #     if boxes is not None:
            #         global scale_factor
            #         global correction
            #         boxes = letterbox(boxes, scale_factor[0], correction)

            # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # if boxes is not None:
            #     draw(img_test, boxes, scores, classes)
            str111 = str(datetime.datetime.now().time())
            if classes is not None:
                for i in range(len(classes)):
                    road_clc.append(classes[i])
                    road_clc.append(scores[i])
                    road_clc.append(boxes[i, 0])
                    road_clc.append(boxes[i, 1])
                    road_clc.append(boxes[i, 2])
                    road_clc.append(boxes[i, 2])
                    top2 = int(boxes[i, 0])
                    left2 = int(boxes[i, 1])
                    right2 = int(boxes[i, 2])
                    bottom2 = int(boxes[i, 3])
                    hei = (top2 + bottom2) / 2
                    hei2 = int(hei)
                    time_str9 = str(datetime.datetime.now().time())
                    str111 = time_str9
                    label_current_time9 = "current: " + time_str9 + "center height: " + str(hei2) + "type :" + str(
                        classes[i])
                    label_current_time10 = "rode type: " + CLASSES[classes[i]]
                    log2.logger.debug(label_current_time9)
                    log2.logger.debug(label_current_time10)
            cv2.putText(img_test, 'time: ' + str111, (25, 60), font, 1, (0, 255, 255), 1)

            data_road = Float64MultiArray()
            data_road.data = road_clc
            print(road_clc)
            road_type_pub.publish(data_road)

            end_time = time.process_time()
            time_length = (end_time - start_time) * 1000
            label_time = "ai shi bei time length: " + "{}".format(int(time_length)) + ": ms"

            time_str2 = str(datetime.datetime.now().time())
            label_current_time = "cur: " + time_str2
            # cv2.putText(img_test, label_current_time, (25, 60), font, 1, (0, 255, 255), 1)

            label_log_time = label_time + "  " + str(datetime.datetime.now()) + "  frame number:  " + str(
                count_frame) + " road_type: " + str(road_clc)
            log.logger.debug(label_log_time)

            label_gps = "gps current frame {}  current time: ".format(count_frame) + "  gps lat: {}".format(
                gps_lat) + "  gps lon:  {}".format(gps_lon) + "  gps_alt:  {}".format(gps_alt) + label_current_time
            log.logger.debug(label_gps)
            writer_video.write(img_test)
            cv2.imshow("image", img_test)
            cv2.waitKey(3)
            count_frame = count_frame + 1
            if count_frame > 2400:
                count_frame = 0
                writer_video.release()
                break


if __name__ == '__main__':
    try:
        rospy.init_node("node_name2")
        my_node = Nodo()

        th = threading.Thread(target=func_ai2)
        th.start()
        th.join()

        rospy.spin()
    except Exception as e:
        log.logger.exception(e)
