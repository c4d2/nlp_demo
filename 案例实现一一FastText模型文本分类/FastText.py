import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 编写参数配置类
class Config(object):
    # 配置参数
    def __init__(self):
        self.model_name = 'FastText'
        self.train_path = './data/train.txt'
        # 训练集
        self.dev_path = './data/dev.txt'
        # 验证集
        self.test_path = './data/test.txt'
        # 测试集
        self.predict_path = './data/predict.txt'
        self.class_list = [x.strip() for x in open('./data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = './data/vocab.pkl'  # 词表
        # 模型训练结果
        self.save_path = './saved dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5 #随机失活
        #若超过1000 patch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)#类别数
        self.n_vocab = 0 #词表大小，在运行时赋值
        self.num_epochs = 5 #epoch数
        self.batch_size = 32 #mini-batch大小
        self.pad_size = 32 #每句话处理成的长度（短填长切）
        self.learning_rate = 1e-3 #学习率
        self.embed = 300 #字向量维度
        self.hidden_size =256 #隐藏层大小
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)

# 编写模型类
class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.embedding = nn.Embedding(
            config.n_vocab,  # 词汇表达大小
            config.embed,    # 词向量维度
            padding_idx=config.n_vocab-1 # 填充
        )
        self.dropout = nn.Dropout(config.dropout)   # 丢弃
        self.fc1 = nn.Linear(config.embed, config.hidden_size)   # 全连接层
        self.dropout = nn.Dropout(config.dropout)  # 丢弃
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)   #全连接层

    # 前向传播计算
    def forward(self, x):
        # 词嵌入
        out_word = self.embedding(x[0])
        out = out_word.mean(dim=1)
        out = self.dropout(out)
        # print(out.shape)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
