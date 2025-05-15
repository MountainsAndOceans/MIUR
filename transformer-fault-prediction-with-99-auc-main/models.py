import numpy as np
import random
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.init import xavier_normal
import torch.nn.functional as F
from utils import to_gpu
from utils import ReverseLayerF
from torchvision import models


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)  # 矩阵点乘
    return masked.sum(dim=dim) / mask.sum(dim=dim)  # 2维的时候，dim=0横向求和，dim=1纵向求和


def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)  # 生成和括号内变量维度维度一致的全是零的内容
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


# let's define a simple model that can deal with multimodal variable length sequence
# 定义一个可以处理多模态变长序列的简单模型
class MISA(nn.Module):  # 继承nn.Module
    def __init__(self, config):  # config是config.py中所赋的初值
        super(MISA, self).__init__()  # 调用nn.Module的构造函数

        model_resnet = models.resnet50()
        """
        resnet50()参数：
        pretrained（bool）：如果为True，则返回在ImageNet上预先训练的模型，默认False
        progress（bool）：如果为True，则显示下载到stderr的进度条，默认True
        """
        self.conv1 = model_resnet.conv1  # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 网络输入通道数：3  网络输出的通道数：64  卷积核的大小：7×7  步长：2代表着高（h）进行步长为2；2代表着宽（w）进行步长为2
        # padding=(3，3)，左右方向添加3，上下方向添加3，各总共添加6
        self.bn1 = model_resnet.bn1  # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # 输入BN层的通道数：64  affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
        self.relu = model_resnet.relu  # (relu): ReLU(inplace=True)
        self.maxpool = model_resnet.maxpool
        # 池化层(maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool  # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        """
        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        """
        self.config = config
        self.model1_size = config.model1_size  # 5
        self.model2_size = config.model2_size  # 8
        self.model3_size = config.model3_size  # 3
        self.model4_size = config.model4_size  # 128

        self.full_layer = nn.Linear(2048, self.model4_size)  # 通过线性变换改变样本大小y=Ax+b
        # in_features = 2048  out_features = 128
        self.input_sizes = input_sizes = [self.model1_size, self.model2_size, self.model3_size, self.model4_size]  # 列表
        self.hidden_sizes = hidden_sizes = [int(self.model1_size), int(self.model2_size), int(self.model3_size), int(self.model4_size)]
        self.output_size = output_size = config.num_classes

        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()  # config.activation() = ReLU(),默认激活函数
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        self.vrnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        # 输入层节点数：5  隐含层节点数：5  num_layer = 1  bidirectional=True双向LSTM
        self.vrnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        # 输入层节点数：10  隐含层节点数：5  num_layer = 1  bidirectional=True双向LSTM
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        # normalized_shape = 10，tensor(list)的最后一个维度的大小

        self.arnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.alayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))

        self.rrnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.rrnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.rlayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        self.ornn1 = rnn(input_sizes[3], hidden_sizes[3], bidirectional=True)
        self.ornn2 = rnn(2 * hidden_sizes[3], hidden_sizes[3], bidirectional=True)

        self.olayer_norm = nn.LayerNorm((hidden_sizes[3] * 2,))

        ##########################################
        # mapping modalities to same sized space
        # 将模态映射到相同大小的空间
        ##########################################
        self.project_v = nn.Sequential()  # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)  # self.activation = ReLU()
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        # nn.LayerNorm(config.hidden_size) = out_features

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_r = nn.Sequential()
        self.project_r.add_module('project_r',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=config.hidden_size))
        self.project_r.add_module('project_r_activation', self.activation)
        self.project_r.add_module('project_r_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_o = nn.Sequential()
        self.project_o.add_module('project_o',
                                  nn.Linear(in_features=hidden_sizes[3], out_features=config.hidden_size))
        self.project_o.add_module('project_o_activation', self.activation)
        self.project_o.add_module('project_o_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())  # Sigmoid激活函数计算 1/(1 + e ^(-x))

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        self.private_r = nn.Sequential()
        self.private_r.add_module('private_r_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_r.add_module('private_r_activation_3', nn.Sigmoid())

        self.private_o = nn.Sequential()
        self.private_o.add_module('private_o_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_o.add_module('private_o_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # shared-private collaborative discriminator 共享专用协作鉴别器
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))

        ##########################################
        # reconstruct
        ##########################################
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_r = nn.Sequential()
        self.recon_r.add_module('recon_r_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_o = nn.Sequential()
        self.recon_o.add_module('recon_o_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        # the number of expected features in the input = 128，类似CNN用128个filter去提取特征
        # the number of heads in the multiheadattention models = 2
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # TransformerEncoder是一个由N个编码器层组成的堆栈，第一个参数即为TransformerEncoderLayer的一个实例，即上述encoder_layer
        # num_layers = 1 编码器中的子编码器层数

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 12,
                                                           out_features=self.config.hidden_size * 1))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))  # dropout_rate = 0.5 神经元有50%概率不被激活
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 1, out_features=output_size))
        # config.num_classes = output_size = 1

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        # https://blog.csdn.net/weixin_44783002/article/details/120575497 pack_padded_sequence和pad_packed_sequence
        # 提取特征并转化为序列形式
        # sequence为已根据长度大小排好序, lengths需要从大到小排序

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)  # LSTM网络训练，省略了h0和c0
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)  # final_h1进行二合一
        # 将转化成的序列转回去
        normed_h1 = layer_norm(padded_h1)  # 归一化
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, model1, model2, model3, model4, lengths):

        batch_size = lengths.size(0)

        # movie

        # extract features from model1 modality
        final_h1_model1, final_h2_model1 = self.extract_features(model1, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_model1 = torch.cat((final_h1_model1, final_h2_model1), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        # 将经过2次LSTM的结果合并，得到utterance_model

        # extract features from model2 modality
        final_h1_model2, final_h2_model2 = self.extract_features(model2, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_model2 = torch.cat((final_h1_model2, final_h2_model2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        final_h1_model3, final_h2_model3 = self.extract_features(model3, lengths, self.rrnn1, self.rrnn2,
                                                                 self.rlayer_norm)
        utterance_model3 = torch.cat((final_h1_model3, final_h2_model3), dim=2).permute(1, 0, 2).contiguous().view(
            batch_size, -1)

        model4 = model4.transpose(1, 0)  # Tensor的维度转置
        model4 = self.feature_layers(model4)
        model4 = model4.view(model4.size(0), -1)
        utterance_model4 = self.full_layer(model4)

        # Shared-private encoders
        self.shared_private(utterance_model1, utterance_model2, utterance_model3, utterance_model4)

        self.domain_label_v = None
        self.domain_label_a = None
        self.domain_label_r= None
        self.domain_label_o = None

        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator((self.utt_shared_v +  self.utt_shared_a +
                                                          self.utt_shared_r + self.utt_shared_o) / 4.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        ######
        h = torch.stack((self.utt_private_v, self.utt_private_a, self.utt_private_r, self.utt_private_o,
                         self.utt_shared_v, self.utt_shared_a, self.utt_shared_r, self.utt_shared_o,
                         self.utt_v_recon, self.utt_a_recon, self.utt_r_recon, self.utt_o_recon), dim=0)
        h_shared = self.transformer_encoder(h)
        # h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]), dim=1)
        h_shared = torch.cat((h_shared[0], h_shared[1], h_shared[2], h_shared[3], h_shared[4], h_shared[5], h_shared[6],
                              h_shared[7], h_shared[8], h_shared[9], h_shared[10], h_shared[11]), dim=1)
        # o = self.fusion(h)
        o = self.fusion(h_shared)
        return o

    def reconstruct(self,):

        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)
        self.utt_r = (self.utt_private_r + self.utt_shared_r)
        self.utt_o = (self.utt_private_o + self.utt_shared_o)

        self.utt_v_recon = self.recon_v(self.utt_v)  # ReconLoss: v_recon, v_orig
        self.utt_a_recon = self.recon_a(self.utt_a)  # ReconLoss: a_recon, a_orig
        self.utt_r_recon = self.recon_r(self.utt_r)
        self.utt_o_recon = self.recon_o(self.utt_o)

    def shared_private(self, utterance_v, utterance_a, utterance_r, utterance_o):

        # Projecting to same sized space
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)
        self.utt_r_orig = utterance_r = self.project_r(utterance_r)
        self.utt_o_orig = utterance_o = self.project_o(utterance_o)

        # Private-shared components
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)
        self.utt_private_r = self.private_r(utterance_o)
        self.utt_private_o = self.private_r(utterance_o)

        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)
        self.utt_shared_r = self.shared(utterance_r)
        self.utt_shared_o = self.shared(utterance_o)

    def forward(self, model1, model2, model3, model4, lengths):
        batch_size = lengths.size(0)
        # v = self.cnn1(model1)
        o = self.alignment(model1, model2, model3, model4, lengths)

        return o
