import torch

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        # 批次大小（在config中定义）
        self.batch_size = batch_size
        # 数据
        self.batches = batches
        # //整除操作符  // 操作符会向下取整，舍弃余数。
        self.n_batches = len(batches) // batch_size
        self.residue = False # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 讲数据集转为tensor
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad 前的长度（超过pad_size的设为pad_size）
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        # 有剩余数据并且当前索引小于批次大小
        if self.residue and self.index < self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1

            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, predict):
    if predict is True:
        config.batch_size = 1
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter