import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

lines = []

with open('image_list.txt', encoding='utf-8') as file:
    lines = file.readlines()  # 将image_list.txt中的内容（图片的路径）读取至lines列表中


def read_image(img_path, resize_size=256, crop_size=224):  # img_path参数的类型为什么直接是{seek}，{seek}是什么类型

    if '\n' in img_path[-1:]:
        img_path = img_path[:-1]

    img = Image.open(img_path).convert('RGB')

    # Imagenet数据集的均值和方差为：mean=(0.485, 0.456, 0.406)，std=(0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 数据标准化处理
                                     std=[0.229, 0.224, 0.225])
    """
    transforms.Normalize
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    transform_test = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),  # TODO 在图片的中间区域进行裁剪
        transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
        transforms.ToTensor(),  # ToTensor()能够把灰度范围从0-255变换到0-1之间
        normalize  # 灰度值经过ToTensor之后变为0-1之间，经过Normalize，灰度值=(灰度值-mean)/std。
    ])

    img = transform_test(img)

    return img


class MOSI:
    def __init__(self):

        """
        tf = pd.read_csv('Overview.csv')

        tf['DeviceTimeStamp'] = pd.to_datetime(tf['DeviceTimeStamp'], format='%Y-%m-%d %H:%M:%S')

        cv = pd.read_csv('CurrentVoltage.csv')

        cv['DeviceTimeStamp'] = pd.to_datetime(cv['DeviceTimeStamp'], format='%Y-%m-%d %H:%M:%S')

        transformer = pd.merge(tf, cv, on='DeviceTimeStamp')

        """
        tf = pd.read_csv('Transformer_Data_1.csv', encoding='gb18030')  # 读取transformerData.csv中内容
        multi_model = tf[:312]

        # spilt train valid test dataset
        train_dataset, val_dataset = train_test_split(multi_model, test_size=0.16666666)
        # 从multi_model中将train_dataset和val_dataset按1:5的比例划分出来且顺序随机
        val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5)
        # 从val_dataset中将val_dataset和test_dataset按1:1的比例划分出来

        train_dataset, val_dataset, test_dataset = train_dataset.values, val_dataset.values, test_dataset.values
        # .values将文字部分去除，保留数字部分
        test_dataset[25:26] = tf[312:313].values

        # define train valid test list
        train = []  # list
        valid = []  # list
        test = []  # list

        # build multimodal dataset
        # 构建多模态数据集
        for index in range(0, len(train_dataset)):  # 15 minute a epoch
            # define tuple
            image_index = train_dataset[index:index+1, 16:17]
            train_image = read_image(lines[int(image_index[0][0])])  # {seek}
            row = [train_dataset[index:index+1, :5], train_dataset[index:index+1, 5:13],
                   train_dataset[index:index+1, 13:16], train_image.numpy(), train_dataset[index:index+1, 17:18]]
            # train_image.numpy()将Tensor转化为ndarray
            train.append(row)

        for index in range(0, len(val_dataset)):
            # row = val_dataset[index:index+1, :]  # 行
            image_index = val_dataset[index:index + 1, 16:17]
            val_image = read_image(lines[int(image_index[0][0])])
            row = [val_dataset[index:index + 1, :5], val_dataset[index:index + 1, 5:13],
                   val_dataset[index:index + 1, 13:16], val_image.numpy(), val_dataset[index:index + 1, 17:18]]
            valid.append(row)

        for index in range(0, len(test_dataset)):
            image_index = test_dataset[index:index + 1, 16:17]
            test_image = read_image(lines[int(image_index[0][0])])
            row = [test_dataset[index:index + 1, :5], test_dataset[index:index + 1, 5:13],
                   test_dataset[index:index + 1, 13:16], test_image.numpy(), test_dataset[index:index + 1, 17:18]]
            test.append(row)

        self.train = train
        self.dev = valid
        self.test = test

        # y = transformer['MOG_A']

    def get_data(self, mode):
        if mode == "train":
            return self.train
        elif mode == "dev":
            return self.dev
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()
