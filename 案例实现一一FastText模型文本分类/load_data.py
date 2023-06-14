import os
import pickle as pkl
from tqdm import tqdm
MAX_VOCAB_SIZE = 10000 #词表长度限制
UNK,PAD = '<UNK>','<PAD>' #未知字，padding符号

# 编辑词典函数
def build_vocab(file_path,tokenizer,max_size,min_freq):
    vocab_dic = {}
    # 打开路径文件
    with open(file_path,'r',encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            # 去掉其中的空行
            if not lin:
                continue
            # 去除后面的数字
            content = lin.split('\t')[0]
            # 对单词进行分词操作（字符级别）
            for word in tokenizer(content):
                # 构建词典  统计每个词出现的频率
                vocab_dic[word] = vocab_dic.get(word, 0)+1
        # 将出现频率高的词排在前面
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 将所有的词重新按照频率高到低顺序  依次编号
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 向 vocab_dic 中添加两个特殊单词的映射：UNK 表示未知单词，PAD 表示填充单词。UNK 的编号为词汇表大小，而 PAD 的编号为词汇表大小加 1。
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1})
    return vocab_dic

# 编辑建立数据集函数
def build_dataset(config,ues_word):
    # 根据 ues_word 变量的值在单词级别和字符级别之间切换分词方式
    if ues_word:
        tokenizer = lambda x: x.split('') # 以空格隔开，word-level  单词级别
    else:
        tokenizer = lambda x: [y for y in x] # char-level  字符级别

    # 如果存在 词典文件则直接读取
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path,'rb'))
    # 不存在则创建词典文件
    else:
        # config.train_path = './data/train.txt'
        vocab = build_vocab(config.train_path, tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
        print(f"Vocab size:{len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        # 读取路径
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):

                lin = line.strip()
                if not lin:
                    continue
                # 存储内容和标签
                content, label = lin.split('\t')
                words_line = []
                # 以字符方式进行分词处理
                token = tokenizer(content)
                # 记录所有文件中词的数量
                seq_len = len(token)
                # token = ['传', '凡', '客', '诚', '品', '裁', '员', '5', '%', ' ', '电', '商', '寒', '冬', '或', '提', '前', '到', '来']
                # 将token固定为同样长度
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD]*(pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # 讲统一填充的词完成词到编号的转换
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                contents.append((words_line, int(label), seq_len))
        return contents

    # 加载训练集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回训练集、验证集和测试集
    return vocab, train, dev, test