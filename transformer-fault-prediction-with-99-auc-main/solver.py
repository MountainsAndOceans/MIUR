import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from torchvision import transforms

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader,
                 is_train=True, model=None):

        self.train_config = train_config
        self.dev_config = dev_config
        self.test_config = test_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)

        # Final list
        for name, param in self.model.named_parameters():
            # named_parameters()返回的list中，每个元组(与list相似，只是数据不可修改)打包了2个内容，分别是layer-name和layer-param(网络层的名字和参数的迭代器)
            # weight_ih是LSTM公式中Wi权重，weight_hh是LSTM公式中Wh权重，bias_ih和bias_hh是LSTM公式中的2个偏置项
            # Wi的shape为(hidden_size*4)×input_size维
            # Wh的shape为(hidden_size*4)×hidden_size维
            """
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
            g_t = \Tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
            c_t = f_t \odot c_{t-1} + i_t \odot g_t
            h_t = o_t \odot \Tanh(c_t)
            """
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)  # 使用正交矩阵填充张量
            print('\t' + name, param.requires_grad)

        if self.is_train:
            self.optimizer = self.train_config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.train_config.learning_rate)
            # 将param中requires_grad值为False的过滤掉，实际上全是True。 requires_grad为True：可以求导

    @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 6

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        # if self.train_config.data == 'ur_funny':
        # self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        # else:  # mosi and mosei are regression datasets
        self.criterion = criterion = nn.MSELoss(reduction="mean")  # MSE损失函数，计算output和target之差的平方

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")  # 交叉熵损失函数
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()  # 计算MSE(均方误差)
        self.loss_cmd = CMD()

        best_valid_loss = float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)  # 指数衰减学习率控制器 调整公式：lr = lr*gamma**epoch
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=80, eta_min=0.0000001)

        train_losses = []
        valid_losses = []
        for e in range(self.train_config.n_epoch):  # n_epoch = 100
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            for batch in self.train_data_loader:
                self.model.zero_grad()  # 将模型的参数梯度初始化为0
                model1, model2, model3, model4, y, l = batch
                batch_size = model1.size(0)
                y_tilde = self.model(model1, model2, model3, model4, l)

                cls_loss = criterion(y_tilde.to(torch.float32), y.to(torch.float32))  # y_tilde是输入，y是目标
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                # diff_loss = 0
                # domain_loss = 0
                # recon_loss = 0
                # cmd_loss = 0

                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss

                loss = cls_loss + \
                       self.train_config.diff_weight * diff_loss + \
                       self.train_config.sim_weight * similarity_loss + \
                       self.train_config.recon_weight * recon_loss

                loss.backward()  # 损失函数反向传播，逐项求导

                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad],
                                                self.train_config.clip)  # 避免梯度爆炸
                self.optimizer.step()  # 一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数

                train_loss_cls.append(cls_loss.item())
                # 在训练时统计loss变化时，会用到loss.item()，能够防止tensor无限叠加导致的显存爆炸
                train_loss_diff.append(diff_loss.item())
                train_loss_recon.append(recon_loss.item())
                train_loss.append(loss.item())
                train_loss_sim.append(similarity_loss.item())

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")
            # print(f"Training loss: {round(np.mean(train_loss), 4)}")

            valid_loss, valid_acc = self.eval(mode="dev")

            print(f"Current patience: {curr_patience}, current trial: {num_trials}, curr_porch: {e}")

            if valid_loss <= best_valid_loss:
                print(f"valid_loss:{valid_loss}")
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
                ############################################
                # self.eval(mode="test", to_print=True)
                ############################################
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        self.eval(mode="test", to_print=True)

    def eval(self, mode=None, to_print=False):
        assert (mode is not None)  # 表达式为真继续运行，否则报错
        self.model.eval()
        """
        model.eval()的作用是不启用 Batch Normalization 和 Dropout。

        如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方
        差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

        训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会
        改变权值。这是model中含有BN层和Dropout所带来的的性质。
        """

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
            dataconfig = self.dev_config
        elif mode == "test":
            dataloader = self.test_data_loader
            dataconfig = self.test_config

            if to_print:
                self.model.load_state_dict(torch.load(
                    f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()

                model1, model2, model3, model4, y, l = batch
                y_tilde = self.model(model1, model2, model3, model4, l)

                cls_loss = self.criterion(y_tilde, y)  # MSELoss
                diff_loss = self.get_diff_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                loss = cls_loss + \
                       dataconfig.diff_weight * diff_loss + \
                       dataconfig.sim_weight * cmd_loss + \
                       dataconfig.recon_weight * recon_loss

                eval_loss.append(loss.item())  # .item()用于在只包含一个元素的tensor中提取值
                y_pred.append(y_tilde.detach().cpu().numpy())  #

                y_true.append(y.detach().cpu().numpy())  # .numpy将tensor格式转化为array格式

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()  # 原先8×1，改为1×8
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))  # 效果不佳，通常acc不会太高

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """

        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))

            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])  # 取出实际有故障的

            test_preds_a7 = np.clip(test_preds, a_min=-1., a_max=1.)  # 限制test_preds中元素的范围在-3~3之间
            test_truth_a7 = np.clip(test_truth, a_min=-1., a_max=1.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
            # np.absolute求绝对值
            corr = np.corrcoef(test_preds, test_truth)[0][1]  # np.corrcoef计算相关系数矩阵
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            # multiclass_acc计算方式：np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

            if non_zeros.size == 0:
                if test_preds.any() > 0:
                    f_score = 1.0
                else:
                    f_score = 0
            else:
                f_score = f1_score((test_preds[non_zeros] > 0.0), (test_truth[non_zeros] > 0.0), average='weighted')

            # pos - neg
            if non_zeros.size == 0:
                binary_truth = [True]
                binary_preds = [True]
                mult_a7 = 0
            else:
                # binary_truth = (test_truth[non_zeros] > 0)  # 仅统计有故障的那个
                # binary_preds = (test_preds[non_zeros] > 0)
                binary_truth = (test_truth != 0)
                binary_preds = [True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True]
                for i in range(len(test_preds)):
                    if np.round(test_preds)[i] == test_truth[i]:
                        if not test_truth[i]:
                            binary_preds[i] = False
                        else:
                            binary_preds[i] = True
                    else:
                        if not test_truth[i]:
                            binary_preds[i] = True
                        else:
                            binary_preds[i] = False

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc(变压器故障率): ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)

            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

            return accuracy_score(binary_truth, binary_preds)

    def get_domain_loss(self, ):

        if self.train_config.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels  # 保留
        domain_true_v = torch.LongTensor([1] * domain_pred_v.size(0))
        domain_true_a = torch.LongTensor([2] * domain_pred_a.size(0))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self, ):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_r, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_o, 5)
        loss += self.loss_cmd(self.model.utt_shared_v, self.model.utt_shared_r, 5)
        loss += self.loss_cmd(self.model.utt_shared_v, self.model.utt_shared_o, 5)
        loss += self.loss_cmd(self.model.utt_shared_r, self.model.utt_shared_o, 5)
        loss = loss / 6.0
        # loss = loss / 3.0

        return loss

    def get_diff_loss(self):

        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        shared_r = self.model.utt_shared_r
        shared_o = self.model.utt_shared_o
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a
        private_r = self.model.utt_private_r
        private_o = self.model.utt_private_o

        # Between private and shared
        loss = self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)
        loss += self.loss_diff(private_r, shared_r)
        loss += self.loss_diff(private_o, shared_o)

        # Across privates
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_a, private_r)
        loss += self.loss_diff(private_a, private_o)
        loss += self.loss_diff(private_v, private_r)
        loss += self.loss_diff(private_v, private_o)
        loss += self.loss_diff(private_r, private_o)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss += self.loss_recon(self.model.utt_r_recon, self.model.utt_r_orig)
        loss += self.loss_recon(self.model.utt_o_recon, self.model.utt_o_orig)
        loss = loss / 4.0
        return loss
