#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
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


def main():
    print("starting!")  # 程序启动
    rospy.init_node('kitti', anonymous=True)
    print("node starting!")
    bridge = CvBridge()
    image_pub = rospy.Publisher("image_orioi", Image, queue_size=10)
    image_sematic_pub = rospy.Publisher("sematic_image", Image, queue_size=10)

    # 针对在道路环境的送货小车，选择cityscapes
    decode_fn = Cityscapes.decode_target
    print("decode!")

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    input = "/home/wayne/slam_program/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/"
    if os.path.isdir(input):

        for filename in os.listdir(input):
            data_collect = input + '/' + filename
            image_files.append(data_collect)

    elif os.path.isfile(input):
        image_files.append(input)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=16)
    print("model setting!")
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    print("model loading!")

    ckpt = "/home/wayne/catkin_slam/src/deeplabv3/src/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    if ckpt is not None and os.path.isfile(ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan

        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    # 数据处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    # if opts.save_val_results_to is not None:
    #     os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            print("predict!")
            print(img_path)
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            print(img_name)

            # img = Image.open(img_path).convert('RGB')

            # read imgs
            img_bgr = cv2.imread(img_path)
            # from bgr to rgb
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            imgcv_msg = bridge.cv2_to_imgmsg(img, "bgr8")  # 从opencv到ros类型
            image_pub.publish(imgcv_msg)  # 发布图像

            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            # 开始预测
            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = decode_fn(pred).astype('uint8')

            # 数组到opencv
            img_cv = cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR)

            # 转换成消息
            img_msg = bridge.cv2_to_imgmsg(img_cv, "bgr8")

            # 发布消息
            image_sematic_pub.publish(img_msg)

            cv2.imwrite(os.path.join("/home/wayne/catkin_slam/src/deeplabv3/src/test", img_name + ".png"), img_cv)

            # colorized_preds = Image.fromarray(colorized_preds)
            # if opts.save_val_results_to:
            #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))


if __name__ == '__main__':
    main()
