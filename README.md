# TextClassificationPytorch
TextClassification

# 项目介绍
该项目使用Pytorch框架完成对系统中基于LSTM和Transformer的分类器的训练

## 项目环境
torch 1.7.0

numpy

pkuseg

matplotlib.pyplot

sklearn

## 项目文件功能介绍
main.py 模型训练入口

utils.py dataset类的构建

PreProcess.py 数据预处理，更改数据格式，统计出现的词语构成词典，精简预训练词向量文件，构建向量映射

data

----dataset

--------test.txt 测试集

--------test.txt 训练集

----pretrained

--------new.npz 精简后的预训练词向量   

----pretreatment

--------test.txt 处理后的测试集

--------test.txt 处理后的训练集

model

----LSTM.py 基于LSTM的模型

----Transformer.py 基于Encoder的模型
        
        
