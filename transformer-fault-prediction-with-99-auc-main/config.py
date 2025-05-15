import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

username = Path.home().name  # 返回当前代码所在路径
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}  # 优化器字典
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}  # 行为字典


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():  # 将上述字典中的key和value初始化到config中
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)  # self.key = value

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        """优化输出界面"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    获取作为类属性的配置
    1.使用argparse分析配置。
    2.创建用解析的kwargs初始化的Config类。
    3.返回Config类。
    """
    parser = argparse.ArgumentParser()  # 对象，用于将命令行字符串解析为Python对象。

    parser.add_argument('--num_classes', type=int, default=1)  # 是不是请求num_classes位置的参数
    parser.add_argument('--batch_size', type=int, default=26)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--use_cmd_sim', type=bool, default=True)
    parser.add_argument('--name', type=str, default=f"best")

    parser.add_argument('--model1_size', type=int, default=5)
    parser.add_argument('--model2_size', type=int, default=8)
    parser.add_argument('--model3_size', type=int, default=3)
    parser.add_argument('--model4_size', type=int, default=128)

    parser.add_argument('--diff_weight', type=float, default=0.5)
    parser.add_argument('--sim_weight', type=float, default=0.5)
    parser.add_argument('--sp_weight', type=float, default=0.5)
    parser.add_argument('--recon_weight', type=float, default=0.5)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=128)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')

    # Model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, }')
    # Data
    parser.add_argument('--data', type=str, default='mosi')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()  # 通过kwargs = parser.parse_args()把刚才的属性从parser给kwargs，后面直接通过kwargs使用。
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs.num_classes = 1

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
