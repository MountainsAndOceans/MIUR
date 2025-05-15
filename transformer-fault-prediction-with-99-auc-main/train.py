import os
import pickle
import numpy as np
from random import random

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F


if __name__ == '__main__':

    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle=True)  # shuffle是对数据进行洗牌
    dev_data_loader = get_loader(dev_config, shuffle=False)
    test_data_loader = get_loader(test_config, shuffle=False)

    # Solver is a wrapper for model training and testing
    solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader,
                    is_train=True)

    solver.build()

    solver.train()
