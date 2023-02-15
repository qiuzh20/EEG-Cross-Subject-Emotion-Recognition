import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# 参考 https://github.com/s4rduk4r/eegnet_pytorch 实现，采用了与Contrastive Learning of Subject-Invariant 
# EEG Representations for Cross-Subject Emotion Recognition 中类似的分别在通道和时间层面上做1维卷积的处理方式
class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y

class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 32, Samples: int = 128,
                 dropoutRate: float = 0.25, spacial_kernLength: int = 61, temporal_kernLength: int = 31,
                 F1:int = 8, D:int = 2, batch_norm=False):
        super().__init__()
        F2 = F1 * D
        # Make kernel size and odd number
        try:
            assert spacial_kernLength % 2 != 0 and temporal_kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: spacial and temporal kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, spacial_kernLength, padding=(spacial_kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) if batch_norm else nn.Identity() # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) if batch_norm else nn.Identity() # (B, F2, Samples - Chans + 1, 1)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1), 1)
        # Out: (B, F2, (Samples - Chans + 1), 1)
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=temporal_kernLength, padding=(temporal_kernLength // 2))
        self.bn3 = nn.BatchNorm1d(F2) if batch_norm else nn.Identity()
        # In: (B, F2, (Samples - Chans + 1), 1)
        # Out: (B, F2, (Samples - Chans + 1), 1)
        self.avg_pool2 = nn.AvgPool1d(30)
        # In: (B, F2 *  (Samples - Chans + 1) / 30)
        self.fc = nn.Linear(F2 * ((Samples - Chans + 1) // 30), nb_classes)

    def forward(self, x: torch.Tensor, output_feature=False):
        # Block 1
        # print(x.shape)
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = self.conv2(y1)
        # print(y1.shape)
        y1 = F.relu(self.bn2(y1))
        # y1 = self.avg_pool(y1)
        # print(y1.shape)
        y1 = self.dropout(y1)
        # Block 2
        y2 = self.conv3(y1)
        # print(y2.shape)
        y2 = F.relu(self.bn3(y2))
        y2 = self.avg_pool2(y2)
        y2 = self.dropout(y2)
        # print(y2.shape)
        y2 = torch.flatten(y2, 1)
        if output_feature:
            return y2
        else:
            return self.fc(y2)

class EEGNetv2(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 64, Samples: int = 128,
                 dropoutRate: float = 0.25, kernLength: int = 63,
                 F1:int = 8, D:int = 2, batch_norm=False):
        super().__init__()
        F2 = F1 * D
        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples)
        # Out: (B, F1, Samples)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) if batch_norm else nn.Identity() # (B, F1, Samples)
        # In: (B, F1, Samples)
        # Out: (B, F2, Samples - Chans + 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) if batch_norm else nn.Identity() # (B, F2, Samples - Chans + 1)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1))
        # Out: (B, F2, (Samples - Chans + 1))
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(F2) if batch_norm else nn.Identity()
        # self.avg_pool = nn.AvgPool1d(10, stride=1)
        # In: (B, F2)
        self.fc = nn.Linear(F2, nb_classes)

    def forward(self, x: torch.Tensor, output_feature=False):
        # Block 1
        # print(x.shape)
        y1 = self.conv1(x)
        y1 = F.relu(self.bn1(y1))
        # print(y1.shape)
        y1 = self.conv2(y1)
        # print(y1.shape)
        y1 = F.relu(self.bn2(y1))
        # y1 = self.avg_pool(y1)
        # print(y1.shape)
        y1 = self.dropout(y1)
        # Block 2
        y2 = self.conv3(y1)
        # print(y2.shape)
        y2 = F.relu(self.bn3(y2))
        # y2 = self.avg_pool2(y2)
        y2 = self.dropout(y2)
        # y2 = 0.5*torch.log(2*math.pi*math.e*torch.std(y2, dim=-1, unbiased=False))
        # y2 = self.avg_pool(y2.unsqueeze(1)).squeeze(1)
        y2 = torch.mean(y2, dim=-1)
        # print(y2.shape)
        # y2 = torch.flatten(y2, 1)
        if output_feature:
            return y2
        else:
            return self.fc(y2)


# 由 https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb 参考修改而来，使用原始EEG论文中的2D卷积

class EEGNet2D(nn.Module):
    def __init__(self, nb_classes: int=9, dropout=0.25, batch_norm=False):
        super(EEGNet2D, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(1, 64, (1, 32), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(64, False) if batch_norm else nn.Identity()
        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 64, (1, 60), stride=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(64, False) if batch_norm else nn.Identity()
        self.pooling2 = nn.AvgPool2d((1, 30), (1, 2))
        # Layer 3
        # self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(64, 3, (16, 6), (2, 1))
        self.batchnorm3 = nn.BatchNorm2d(3, False) if batch_norm else nn.Identity()
        self.pooling3 = nn.AvgPool2d((6, 3))
        self.fc1 = nn.Linear(48, nb_classes)

    def forward(self, x: torch.Tensor, output_feature=False):
        # Layer 1
        # print(x.shape)
        bs = x.shape[0]
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        x = x.permute(0, 3, 1, 2)
        # Layer 2
        # x = self.padding1(x)
        # print('layer2', x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)
        # print('after pooling', x.shape)
        # Layer 3
        # x = self.padding2(x)
        # print('layer3', x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)
        # print('after pooling', x.shape)
        x = x.reshape(bs, -1)
        if output_feature:
            return x
        else:
            return self.fc1(x)