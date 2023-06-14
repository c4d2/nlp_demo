import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
#编写训练函数
# 传入的是 测试集和验证集
def train(config,model,train_iter,dev_iter):
    print("begin")
    model.train()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0 # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0 # 记录上次验证集loss下降的batch数
    flag = False # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        print('Epoch[{}/{}]'.format(epoch+1,config.num_epochs))
        # 批量训练
        for i ,(trains ,labels) in enumerate(train_iter):
            # 将训练集放在模型中
            outputs = model(trains)
            # 清空模型梯度信息  在每次迭代时，需要将参数的梯度清空，以免当前的梯度信息影响到下一次迭代
            model.zero_grad()
            # 计算损失函数
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            loss.backward()
            # 根据上面计算得到的梯度信息和学习率  对模型的参数进行优化
            optimizer.step()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 存储模型
                    torch.save(model.state_dict(), config.save_path)
                    # 记录batch数
                    last_improve = total_batch

                # {2:6.2%} 是一个格式化字符串语法，表示将第三个参数格式化为一个百分数，并使用右对齐方式，并在左侧填充空格，总宽度为 6 个字符，保留两位小数。
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:6.2%} ,''Val Loss :{3:>5.2}, Val Acc: {4:>6.2%}'
                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc))
                model.train()
            total_batch +=1

            if total_batch - last_improve > config.require_improvement:
                # 验证集1oss超过1000 batch没下降，结束训练
                print("No optimization for a long time,auto-stopping...")
                flag = True
                break
        if flag:
            break

# 编写评价函数
def evaluate(config,model,data_iter,test=False):
    # 将模型切换到评估模式 在评估模式下，模型的行为与训练模式下略有不同。具体来说，评估模式下模型会关闭一些对训练过程的辅助功能，例如 dropout 和 batch normalization 等，并且不会对模型的参数进行更新。
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # 防止模型参数更新
    with torch.no_grad():

        for texts, labels in data_iter:
            outputs = model(texts)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失值累加
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            # labels_all 是所有样本的真实标签
            labels_all = np.append(labels_all, labels)
            # predict_all 是所有样本的预测标签
            predict_all = np.append(predict_all, predict)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            # config.class_list 是所有可能的类别列表
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            # 混淆矩阵
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)