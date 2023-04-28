#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
from queue import Queue
from tkinter import image_names

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

# 针对ros节点的库

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
font = cv2.FONT_HERSHEY_SIMPLEX

class sematic_slam:
    def __init__(self) :
        self.image_pub = rospy.Publisher("image_orioi", Image, queue_size=10)
        self.sematic_img = rospy.Publisher("sematic_image", Image, queue_size=10)
        self.img_dect_pub=rospy.Publisher("image_dect",Image,queue_size=10)
        self.image_sub = rospy.Subscriber("/kitti/camera_color_left/image_raw",Image,self.callback)
        self.bridge = CvBridge()

        self.decode_fn = Cityscapes.decode_target
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=16)
        self.checkpoint = torch.load("/home/wayne/catkin_slam/src/deeplabv3/src/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model = nn.DataParallel(self.model)
        self.model=self.model.to(self.device).eval()
        self.model.to(self.device)
        del self.checkpoint


    # 数据处理
        self.transform= T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def callback(self,data):
        print("666")
        img_bgr = self.bridge.imgmsg_to_cv2(data,"bgr8")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        imgcv_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")  # 从opencv到ros类型
        self.image_pub.publish(imgcv_msg)  # 发布图像

        img = self.transform(img).unsqueeze(0)  # To tensor of NCHW
        img = img.to(self.device)

        # 开始预测
        pred = self.model(img).max(1)[1].cpu().numpy()[0]  # HW
        colorized_preds = self.decode_fn(pred).astype('uint8')

        # 数组到opencv
        img_cv = cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR)
        self.dectImage(img_bgr,img_cv)

        # 转换成消息
        img_msg = self.bridge.cv2_to_imgmsg(img_cv, "bgr8")
        img_msg.header.stamp=data.header.stamp
        # 发布消息
        self.sematic_img.publish(img_msg)        

    def dectImage(self ,img_orioi,img_sematic):
        img_orioi_copy=img_orioi
        img_sematic_copy=img_sematic
        hsv_image_sematic=cv2.cvtColor(img_sematic_copy,cv2.COLOR_BGR2HSV)

        lower_blue=np.array([100,43,46])
        upper_blue=np.array([124,255,153])
        lower_red = np.array([0,122,120])
        upper_red = np.array([10,255,255])

        mask_red = cv2.inRange(hsv_image_sematic,lower_red,upper_red)
        mask_red = cv2.medianBlur(mask_red, 5)
        # mask_red,
        contours_red,hierarchy = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask_blue = cv2.inRange(hsv_image_sematic,lower_blue,upper_blue)
        mask_blue = cv2.medianBlur(mask_blue, 5)
        # mask_blue,
        contours_blue,hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        if contours_red:
            for cnt in contours_red:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(img_orioi_copy,(x,y),(x+w,y+h),(0,255,255),2)
                cv2.putText(img_orioi_copy,"Person",(x,y),font,0.7,(0,255,0),2)

            #cv->rosmsg
                imgcv_dect=self.bridge.cv2_to_imgmsg(img_orioi_copy,"bgr8")
                self.img_dect_pub.publish(imgcv_dect)
                print("dect pub seccess")

        elif contours_blue:
            for cnt in contours_blue:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(img_orioi_copy,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img_orioi_copy,"Car",(x,y),font,0.7,(0,0,255),2)

          #cv->rosmsg
                imgcv_dect=self.bridge.cv2_to_imgmsg(img_orioi_copy,"bgr8")
                self.img_dect_pub.publish(imgcv_dect)
                print("dect pub seccess")
        else:

            imgcv_dect=self.bridge.cv2_to_imgmsg(img_orioi_copy,"bgr8")
            self.img_dect_pub.publish(imgcv_dect)
            print("dect pub seccess")
# def main():
#     print("starting!")  # 程序启动
#     rospy.init_node('kitti', anonymous=True)
#     print("node starting!")
def main(args):
    print("666")
    sema = sematic_slam()
    rospy.init_node('image_converter',anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("SHut")
    cv2.destroyAllWindows()

    # 针对在道路环境的送货小车，选择cityscapes
    
    # print("decode!")

    # print("Device: %s" % device)

    # Setup dataloader
    # image_files = []
    # input = "/home/wayne/slam_program/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/"
    # if os.path.isdir(input):

    #     for filename in os.listdir(input):
    #         data_collect = input + '/' + filename
    #         image_files.append(data_collect)

    # elif os.path.isfile(input):
    #     image_files.append(input)

    # Set up model (all models are 'constructed at network.modeling)
    # print("model setting!")
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    # print("model loading!")

    # ckpt = "/home/wayne/catkin_slam/src/deeplabv3/src/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    # if ckpt is not None and os.path.isfile(ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan


        # print("Resume model from %s" % ckpt)
    # else:
        # print("[!] Retrain")
        # model = nn.DataParallel(model)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images


    # if opts.save_val_results_to is not None:
    #     os.makedirs(opts.save_val_results_to, exist_ok=True)

    # with torch.no_grad():
    #     model = model.eval()
    #     for img_path in tqdm(image_files):
    #         print("predict!")
    #         print(img_path)
    #         ext = os.path.basename(img_path).split('.')[-1]
    #         img_name = os.path.basename(img_path)[:-len(ext) - 1]
    #         print(img_name)

            # img = Image.open(img_path).convert('RGB')

            # read imgs
            # img_bgr = cv2.imread(img_path)
            # from bgr to rgb

            # image_sematic_pub.publish(img_msg)

            # cv2.imwrite(os.path.join("/home/wayne/catkin_slam/src/deeplabv3/src/test", img_name + ".png"), img_cv)

            # colorized_preds = Image.fromarray(colorized_preds)
            # if opts.save_val_results_to:
            #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))


if __name__ == '__main__':
    main(sys.argv)
