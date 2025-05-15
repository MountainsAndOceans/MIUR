"""
get data loader
"""
from create_data import MOSI
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *


def get_loader(config, shuffle=True):

    """Load DataLoader of given DialogDataset"""
    """加载给定DialogDataset的DataLoader"""

    dataset = MOSI().get_data(config.mode)

    print(config.mode)

    def collate_fn(batch):
        """
        Collate functions assume batch = [Dataset[i] for i in index_set]  排序 ？
        """
        # for later use we sort the batch in descending order of length 长度降序排序
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        """
        shape[0]--矩阵行数
        shape[1]--矩阵列数
        image.shape[0]——图片高
        image.shape[1]——图片长
        image.shape[2]——图片通道数
        """
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample[4]) for sample in batch], dim=0)
        # torch.from_numpy把array转换成tensor
        # labels提取的是故障类型
        mode1 = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch])  # mode1为第一个模态
        # torch.FloatTensor生成64位浮点型的张量
        mode2 = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch])  # mode2为第二个模态
        mode3 = pad_sequence([torch.FloatTensor(sample[2]) for sample in batch])  # mode3为第三个模态
        mode4 = pad_sequence([torch.FloatTensor(sample[3]) for sample in batch])  # mode4为第四个模态

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])  # lengths貌似全为1，batch_size = 8，即8个1

        return mode1, mode2, mode3, mode4, labels, lengths

    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return data_loader
