import torch
import numpy as np
from train import evaluate
MAX_VOCAB_SIZE = 10000
UNK,PAD = '<UNK>','<PAD>'
tokenizer = lambda x:[y for y in x]     #char-level

# 编写测试函数
def test(config,model,test_iter):
    # test
    # 加载训练好的模型
    model.load_state_dict(torch.load(config.save_path))
    model.eval()#开启评价模式
    test_acc,test_loss,test_report,test_confusion = evaluate(config,model,test_iter,test=True)
    msg = 'Test Loss:{0:>5.2},Test Acc:{1:>6.28}'
    print(msg.format(test_loss,test_acc))
    print("Precision,Recall and Fl-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

# 编写加载数据函数
def load_dataset(text, vocab, config, pad_size=32):
    contents = []
    for line in text:
        lin = line.strip()
        if not lin:
            continue
        words_line = []
        token = tokenizer(line)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD](pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # 单词到编号的转换
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        contents.append((words_line, int(0), seq_len))
    return contents  # 数据格式为[([..]，O),([.·],1),.]

# 编写标签匹配函数
def match_label(pred,config):
    label_list = config.class_list
    return label_list[pred]

# 编写预测函数
def final_predict(config, model, data_iter):
    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.load(config.save_path, map_location=map_location))

    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config)for i in pred]
            predict_all = np.append(predict_all, pred_label)
    return predict_all