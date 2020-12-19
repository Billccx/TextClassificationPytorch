# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :TextClassificationPytorch
# @File     :utils
# @Date     :2020/12/19 17:36
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
from os.path import join
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import pkuseg
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataset import Subset

UNK, PAD = '<UNK>', '<PAD>'

class MyDataset(Dataset):
    '''
    dataset类的构建
    '''
    def __init__(self,dic,mode):
        super(Dataset, self).__init__()

        #根据不同模型而修改
        self.pad_size = 32  # 每句话处理成的长度(短填长切)

        data_dir = 'data/pretreatment'
        self.train_path = join(data_dir, 'train.txt')
        self.dev_path = join(data_dir, 'dev.txt')
        self.test_path = join(data_dir, 'test.txt')

        self.dic=dic
        if mode=='train':
            self.data=self.load_dataset(self.train_path)
        elif mode=='test':
            self.data = self.load_dataset(self.test_path)

    def load_dataset(self, path):
        """
        加载指定路径数据集
        """
        pad_size = self.pad_size # 每句话处理成的长度(短填长切) 100
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue

                lin = lin.split('\t')
                words = lin[1]     #句子
                label = lin[0][9:]  #标签

                words_line = []
                seq_len = len(words)

                # 将每个句子划分的结果截长补短成定长
                if pad_size:
                    if len(words) < pad_size:  #短，填充pad补为100
                        words=[x for x in words]
                        words.extend([PAD] * (pad_size - len(words)))
                    else:#长，保留前100
                        words = words[:pad_size]
                        seq_len = pad_size

                # word to id 将句子列表转化为ID列表，未登录词设为UNK
                for word in words:
                    words_line.append(self.dic.get(word, self.dic.get(UNK)))

                #将每句话组织成元组（id列表，标签号，句子的原本长度），压入content
                contents.append((words_line, int(label), seq_len))
        return contents#元组列表


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        vec=np.array(self.data[item][0]).T
        vec=torch.from_numpy(vec).long()
        sample=(vec,self.data[item][1])
        return sample



def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    '''
    绘制混淆矩阵
    @param cm:混淆矩阵
    @param classes:类别
    @param title:
    @param cmap:
    @return:
    '''
    plt.rc('font', family='Times New Roman', size='8')  #设置字体样式、大小
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) #侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('ConfusionMatrix.jpg', dpi=600)
    plt.show()

def get_pretrained(file_path):
    '''
    读取预训练词向量
    :param file_path: npz文件路径
    :return:
    '''
    return torch.tensor(np.load(file_path)["embeddings"].astype('float32'))


def ClassAnalyse(cm,Tags):
    '''
    混淆矩阵分析
    :param cm: 混淆矩阵
    :param Tags: 各类别
    :return: 各类别准确率，召回率，F1
    '''
    cm=np.array(cm,dtype=float)
    pricision=[0. for _ in range(len(Tags))]
    recall=[0. for _ in range(len(Tags))]
    F1=[0. for _ in range(len(Tags))]
    l=len(Tags)
    for i in range(l):
        pricision[i]+=cm[i][i]/np.sum(cm[i])
    cm2=cm.T
    for i in range(l):
        recall[i]+=cm2[i][i]/np.sum(cm2[i])
    for i in range(l):
        F1[i]+=2*pricision[i]*recall[i]/(pricision[i]+recall[i])
    for i in range(l):
        s=''
        s+=Tags[i]+'  precision: '+str(pricision[i])+'  recall: '+str(recall[i])\
           +'  F1: '+str(F1[i])+'\n'
        print(s)
    return pricision,recall,F1

if __name__ == '__main__':
    pass