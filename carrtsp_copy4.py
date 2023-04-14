#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
import os
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray, Float64MultiArray
import subprocess
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
import time

import platform
import threading
import string
import struct

import logging
import datetime
from sensor_msgs.msg import NavSatFix

from logging import handlers

global ai_list
ai_list = []
# rtmp = "rtmp://192.168.1.101:1935/live"
rtsp = "rtsp://47.243.7.221/car003"

sizeStr = "640x480"
fb = 0

command = ['ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-r', '14',
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-tune', 'zerolatency',
           # '-vf',"drawtext=expansion=strftime:  :text='%Y-%m-%d %H-%M-%S':fontsize=30:fontcolor=white:box=1:x=10:y=30:boxcolor=black@0.5:",
           '-f', 'rtsp',
           '-rtsp_transport', 'tcp',
           rtsp]

pipe = subprocess.Popen(command
                        , shell=False
                        , stdin=subprocess.PIPE
                        )
pipe.terminate()
pipe = subprocess.Popen(command
                        , shell=False
                        , stdin=subprocess.PIPE
                        )


class Nodo(object):
    def __init__(self):
        print("begin ros")

        self.br = CvBridge()
        self.forward_or_back = 1
        self.road_type = 0
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(17)

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

            global ai_list
            ai_list.append(image_ai2)
            if len(ai_list) > 3:
                ai_list.pop(0)
            img5 = ai_list[-1]
            img5[0:180, 0:640, :] = 0

            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
            cv2.putText(img5, dt_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        1, cv2.LINE_AA)
            pipe.stdin.write(img5.tobytes())



    def callback_img2(self, msg):
        if self.forward_or_back == 1:
            log.logger.info("/camera/image_raw".format(self.forward_or_back))

            image_ai2 = self.br.imgmsg_to_cv2(msg)

            global ai_list
            ai_list.append(image_ai2)
            if len(ai_list)>3:
                ai_list.pop(0)
            img5 = ai_list[-1]
            img5[0:180, 0:640, :] = 0

            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
            cv2.putText(img5, dt_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        1, cv2.LINE_AA)
            pipe.stdin.write(img5.tobytes())


    def callback_forward(self, msg):
        self.forward_or_back = int(msg.data)




if __name__ == '__main__':
    try:
        rospy.init_node("node_rtsp2")
        my_node = Nodo()

        rospy.spin()
    except Exception as e:
        pass
