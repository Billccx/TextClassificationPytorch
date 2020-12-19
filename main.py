# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :TextClassificationPytorch
# @File     :main
# @Date     :2020/12/19 17:34
# @Author   :CuiChenxi
# @Email    :billcuichenxi@163.com
# @Software :PyCharm
-------------------------------------------------
"""

from PreProcess import *
from utils import *
from model.Transformer import *
from model.LSTM import *
import matplotlib.pyplot as plt


def main():
    BATCH_SIZE = 128
    Learning_Rate = 0.001
    EPOCH = 20

    tags = ['finance',
            'realty',
            'stocks',
            'education',
            'science',
            'society',
            'politics',
            'sports',
            'game',
            'entertainment', ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #数据预处理 构建词典，向量映射，简化词向量文件
    helper = Helper()

    #获取向量映射
    word2id = helper.getVocab()

    #初始化训练集
    mydataset = MyDataset(word2id, mode='train')
    # 初始化测试集
    testset = MyDataset(word2id, mode='test')

    #设置训练集随机采样器
    indices = range(len(mydataset))
    train_sampler = SubsetRandomSampler(indices[:])

    #构建dataloader
    train_loader = DataLoader(mydataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=64)

    #初始化模型
    my_model = Transformer().to(device)
    #my_model=LSTM().to(device)

    #定义优化器
    optimizer = torch.optim.Adam(my_model.parameters(), lr=Learning_Rate, betas=(0.9, 0.99))

    #开始训练
    train_accuracy = []
    val_accuracy = []
    for epoch in range(EPOCH):
        predict = torch.tensor([])
        target = torch.tensor([])
        for index, (train_x, train_y) in enumerate(train_loader):
            # print(index)
            outputs = my_model.forward(train_x.to(device))
            my_model.zero_grad()
            loss = F.cross_entropy(outputs, train_y.to(device))
            loss.backward()
            optimizer.step()

            predict = torch.cat((predict, outputs.argmax(dim=1).cpu().float()), 0)
            target = torch.cat((target, train_y.cpu().float()), 0)

            if index % 100 == 0:
                # 每100各epoch轮输出在训练集和验证集上的效果
                testpredict = torch.tensor([])
                testtarget = torch.tensor([])
                for i, (test_x, test_y) in enumerate(test_loader):
                    out = my_model.forward(test_x.to(device))
                    loss = F.cross_entropy(out, test_y.to(device))
                    pred = torch.max(out.data, 1)[1].cpu()

                    testpredict = torch.cat((testpredict, out.argmax(dim=1).cpu().float()), 0)
                    testtarget = torch.cat((testtarget, test_y.cpu().float()), 0)

                test_acc = metrics.accuracy_score(testtarget, testpredict)
                train_accuracy.append(metrics.accuracy_score(target, predict))
                val_accuracy.append(test_acc)
                if epoch == EPOCH - 1 and index == 1000:
                    # if epoch == 0 and index == 100:
                    print('ready to paint')
                    cm = metrics.confusion_matrix(testtarget, testpredict)
                    plot_Matrix(np.array(cm, dtype=float), tags)
                    ClassAnalyse(cm, tags)
                print('index {} : on test set accuracy is {}'.format(index, test_acc))

        train_acc = metrics.accuracy_score(target, predict)
        print('Epoch {} : on train set accuracy is {}'.format(epoch, train_acc))
    torch.save(my_model.state_dict(), my_model.savepth)

    #绘制曲线
    x = range(len(val_accuracy))
    plt.plot(x, val_accuracy, label='val_acc')
    plt.plot(x, train_accuracy, label='train_acc')
    plt.xlabel("batch/100")
    plt.ylabel("accuracy")
    plt.legend(loc='best')
    plt.savefig('accuracy_cnn.png', dpi=300)


if __name__ == '__main__':
    main()

