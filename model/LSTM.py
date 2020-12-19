# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :TextClassificationPytorch
# @File     :LSTM
# @Date     :2020/12/19 17:51
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.savepth = 'results/LSTM.ckpt'  # 模型训练结果保存路径
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 1  # lstm层数
        self.emb_dim=300 #词向量维度
        self.pad_size = 32 #句子填充长度
        self.num_classes=10 #类别数
        self.dropout = 0.5  # 随机失活
        self.pretrained=get_pretrained('./data/pretrained/new.npz')
        self.embedding = nn.Embedding.from_pretrained(self.pretrained, freeze=False)
        self.lstm=nn.LSTM(
            self.emb_dim, self.hidden_size, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout
        )
        self.max_pool = nn.MaxPool1d(self.pad_size)
        #self.fc = nn.Linear(self.hidden_size * 2 + self.emb_dim, self.num_classes)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        #out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pool(out).squeeze()
        return self.fc(out)

