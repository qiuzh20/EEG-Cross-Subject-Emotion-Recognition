import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import random

class Onesec_Dataset(Dataset):
    def __init__(self, 
                 data:torch.Tensor, 
                 label:torch.Tensor,
                 channel_wise_normalize:bool = False,
                 device='cpu'):
        data = data.to(device=device)
        label = label.to(device=device)
        if channel_wise_normalize:
            data = F.normalize(data, dim=1)
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        if self.data is None or self.labels is None:
            return 0
        else:
            return self.data.shape[0]

class Contrastive_Dataset(Dataset):
    """实现正负样本选择匹配的Dataset, 方便contrastive loss计算
    Args:
        data: 原始数据，形状为（被试数，trail，采样频率*每段总时间，通道数）
        seconds: 每段总时长，我们选取30s
        sample_interval: 选择用于contrastive learning的片段长度，默认为5s
        frequency: 采样频率，在本问题中为 125Hz
        neg_number: 在计算contrastive loss时选取的参考数，最后计算contrastive loss时分母有 2*neg_number+1 项
    """
    def __init__(self, 
                 data:torch.Tensor, # 80, 28, 3750, 32
                 seconds:int = 30,
                 sample_interval:int = 5,
                 frequency:int = 125,
                 neg_number:int = 10,
                 channel_wise_normalize:bool = False,
                 device='cpu'):
        assert neg_number < 28, "number of negtive references should be smaller than number of trails"
        assert sample_interval < 30, "length of reference should be smaller than trail length"
        self.seconds = seconds
        self.frequency = frequency
        self.sample_interval = sample_interval
        data = data.to(device)
        if channel_wise_normalize:
            data = F.normalize(data, dim=1)
        self.data = data
        self.total_trial = self.data.shape[1]
        self.neg_number = neg_number

    def __getitem__(self, index): # index 为样本A
        # 从总的 trail 数中 不重复地 选择neg_number+1个trial
        ref_trails = random.sample(range(self.total_trial), k=self.neg_number+1)
        # 将其中的1个作为正参考，如此实现避免了重复选择
        pos_index = ref_trails[0]
        # 随机在trial的30s中选取一段用于计算feature
        interval = random.randint(0, self.seconds-self.sample_interval-1)
        # 随机在所有被试样本中选取一个作为参考B
        ref_sample = random.randint(0, self.data.shape[0]-1)
        ref_sample = (ref_sample + 1)%self.data.shape[0] if ref_sample==index else ref_sample # 避免重复及索引超出范围
        # 将样本A与样本B的用于计算contrastive loss分母部分组合
        neg_samples = torch.cat((self.data[index, ref_trails[1:], interval*self.frequency:(interval+self.sample_interval)*self.frequency],\
            self.data[ref_sample, ref_trails, interval*self.frequency:(interval+self.sample_interval)*self.frequency]), dim=0)
        # 返回样本A，样本B中的pos reference及neg reference
        return self.data[index, pos_index, interval*self.frequency:(interval+self.sample_interval)*self.frequency],\
                self.data[ref_sample, pos_index, interval*self.frequency:(interval+self.sample_interval)*self.frequency],\
                    neg_samples

    def __len__(self):
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]