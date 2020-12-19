# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :TextClassificationPytorch
# @File     :PreProcess
# @Date     :2020/12/19 17:34
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import os
import json
from os.path import join
from utils import *
import torch.nn.functional as F
from sklearn import metrics
import torch
import pkuseg
import pickle as pkl
from tqdm import tqdm
import numpy as np
from model.Transformer import *

UNK, PAD = '<UNK>', '<PAD>'

def label_map(label):
    """
    将不连续label变为连续
    """
    label = int(label) - 100
    if 5 < label < 11:
        label = label - 1
    elif 11 < label:
        label = label - 2
    return str(label)


class Helper(object):
    def __init__(self):

        #是否分词
        self.use_word=False
        self.seg = pkuseg.pkuseg() if self.use_word else None
        self.vocab_path='./results/vocab.pkl'

        dataset_dir = 'data/dataset'
        #self.ori_train_path = join(dataset_dir, 'train.json')  # 原始训练集
        self.ori_train_path = join(dataset_dir, 'train.txt')
        self.ori_dev_path = join(dataset_dir, 'dev.json')  # 原始验证集
        #self.ori_test_path = join(dataset_dir, 'test.json')  # 原始测试集
        self.ori_test_path = join(dataset_dir, 'test.txt')
        self.ori_label_path = join(dataset_dir, 'labels.json')

        self.min_freq = 1
        self.pad_size = 32  # 每句话处理成的长度(短填长切)

        data_dir = 'data/pretreatment'
        self.train_path = join(data_dir, 'train.txt')
        self.dev_path = join(data_dir, 'dev.txt')
        self.test_path = join(data_dir, 'test.txt')

        #load vocab if exist else create
        vocab_path = self.vocab_path
        print('Loading vocab from', vocab_path, ' ...')
        #如果pkl存在，则载入，否则生成
        self.vocab = pkl.load(open(vocab_path, 'rb')) if \
            os.path.exists(vocab_path) else self.word2v2()
        print('Complete! Vocab size: {}'.format(len(self.vocab)))

    def word2v(self):
        '''
        将训练集数据分词，统计词频，并进行embedding
        :return:
        '''
        print("Vocab isn't exist, creating...")

        #定义词典，（键：出现次数）
        vocab_dic = {}

        # 读取训练数据集
        with open(self.ori_train_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                json_item = line.strip()
                if not json_item:
                    continue
                json_item = json.loads(json_item)
                content = json_item['sentence']
                keywords = json_item['keywords']

                #分词
                if self.use_word:
                    words = self._sentence_segment(content, keywords)
                else:
                    words=[x for x in content]
                    words.extend([x for x in keywords])

                # 统计词频
                for word in words:
                    #若在词典中，词频++,否则设为1
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1


            # 删除不到最小词频的词语，按词频降序
            vocab_list = [item for item in vocab_dic.items() if item[1] >= self.min_freq]
            # 制作成 {词:索引号} 的字典
            vocab_dic = {word_count[0]: index for index, word_count in enumerate(vocab_list)}
            vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

            # save to disk 保存到results/vocab.pkl
            pkl.dump(vocab_dic, open(self.vocab_path, 'wb'))


            #修改新增
            embeddings = np.random.rand(len(vocab_dic), 300)
            f = open('./data/pretrained/wordvector.char', "r", encoding='UTF-8')
            for i, line in enumerate(f.readlines()):
                if i == 0:  # 若第一行是标题，则跳过
                     continue
                lin = line.strip().split(' ')
                if lin[0] in vocab_dic:
                    idx = vocab_dic[lin[0]]
                    emb = [float(x) for x in lin[1:301]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
            f.close()
            np.savez_compressed('./data/pretrained/new', embeddings=embeddings)


        # dataset pretreatment
        self._dataset_pretreatment(self.ori_train_path, self.train_path)
        self._dataset_pretreatment(self.ori_test_path, self.test_path)
        self._dataset_pretreatment(self.ori_dev_path, self.dev_path)
        print('Dataset pretreatment complete')
        return vocab_dic

    def word2v2(self):
        '''
        当使用子向量是调用此函数
        将训练集数据分词，统计词频，并进行embedding
        :return:
        '''
        print("Vocab isn't exist, creating...")

        #定义词典，（键：出现次数）
        vocab_dic = {}

        # 读取训练数据集
        with open(self.ori_train_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line=line.strip()
                line=line.split('\t')
                content=line[0]

                #分词
                # word-level:以空格隔开 char-level:单个字符隔开
                if self.use_word:
                    words = self._sentence_segment(content)
                else:
                    words=[x for x in content]

                # 统计词频
                for word in words:
                    #若在词典中，词频++,否则设为1
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1


            # 删除不到最小词频的词语，按词频降序
            vocab_list = [item for item in vocab_dic.items() if item[1] >= self.min_freq]
            # 制作成 {词:索引号} 的字典
            vocab_dic = {word_count[0]: index for index, word_count in enumerate(vocab_list)}
            vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

            # save to disk 保存到results/vocab.pkl
            pkl.dump(vocab_dic, open(self.vocab_path, 'wb'))


            #修改新增
            embeddings = np.random.rand(len(vocab_dic), 300)
            f = open('./data/pretrained/wordvector.char', "r", encoding='UTF-8')
            for i, line in enumerate(f.readlines()):
                if i == 0:  # 若第一行是标题，则跳过
                     continue
                lin = line.strip().split(' ')
                if lin[0] in vocab_dic:
                    idx = vocab_dic[lin[0]]
                    emb = [float(x) for x in lin[1:301]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
            f.close()
            np.savez_compressed('./data/pretrained/new', embeddings=embeddings)


        # dataset pretreatment
        self._dataset_pretreatment2(self.ori_train_path, self.train_path)
        self._dataset_pretreatment2(self.ori_test_path, self.test_path)
        #self._dataset_pretreatment2(self.ori_dev_path, self.dev_path)
        print('Dataset pretreatment complete')
        return vocab_dic

    #调用pkuseg进行分词的函数
    def _sentence_segment(self, sentence, keywords):
        if self.use_word:
            # 句子分词后添加,关键词直接添加，接在句子后面，返回一个列表
            return self.seg.cut(sentence) + keywords.split(',')
        else:
            #如按字级分词，则将每个char分开即可
            return [char for char in sentence + keywords]

    def _dataset_pretreatment(self, file_path, save_path):
        '''
        将数据集划分结果和标签保存至txt文件中
        pretreatment文件夹下
        '''
        contents = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                json_item = line.strip()
                if not json_item:
                    continue
                json_item = json.loads(json_item)
                label = '__label__' + label_map(json_item['label'])
                words = self._sentence_segment(json_item['sentence'], json_item['keywords'])
                contents.append(label + ' ' + ' '.join(words))
        with open(save_path, 'w', encoding='UTF-8') as w:
            for item in contents:
                w.writelines(item + '\n')

    def _dataset_pretreatment2(self, file_path, save_path):
        '''
        将数据集划分结果和标签保存至txt文件中
        pretreatment文件夹下
        '''
        contents = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line=line.strip()
                line=line.split('\t')
                try:
                    s='__label__'+line[1]+'\t'+line[0]
                except:
                    print(line[0])
                    exit(11)
                contents.append(s)
        with open(save_path, 'w', encoding='UTF-8') as w:
            for item in contents:
                w.writelines(item + '\n')

    def getVocab(self):
        """
        获取词表
        """
        return self.vocab











