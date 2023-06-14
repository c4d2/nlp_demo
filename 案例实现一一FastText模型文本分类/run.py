from FastText import Config
from FastText import Model
from load_data import build_dataset
from load_data_iter import build_iterator
from train import train
from predict import test,load_dataset,final_predict

# 测试文本
text = ['国考网上报名序号查询后务必牢记。报名参加2011年国家公务员考试的考生：如果您已通过资格审查，那么请于10月28日8：00后，登录考录专题网站查询自己的报名序号']
if __name__ == "__main__":
    config = Config()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, False)
    #1,批量加载测试数据
    # 批量记载数据的原因：深度学习模型的参数非常多，为了得到模型的参数，需要用大量的数据对模型进行训练，所以数据量一般是相当大的，
    # 不可能一次性加载到内存中对所有数据进行向前传播和反向传播，因此需要分批次将数据加载到内存中对模型进行训练。使用数据加载器的
    # 目的就是方便分批次将数据加载到模型，以分批次的方式对模型进行迭代训练。
    train_iter = build_iterator(train_data, config, False)
    dev_iter = build_iterator(dev_data, config, False)
    test_iter = build_iterator(test_data, config, False)
    config.n_vocab = len(vocab)
    #2,加载模型结构
    model = Model(config).to(config.device)
    train(config, model, train_iter, dev_iter)
    #3.测试
    test(config, model, test_iter)
    print("+++++++++++++++++")
    #4.预测
    content = load_dataset(text, vocab, config)
    predict_iter = build_iterator(content, config, predict=True)
    result = final_predict(config, model, predict_iter)
    for i, j in enumerate(result):
        print('text:{}'.format(text[i]), '\t', 'label:{}'.format(j))