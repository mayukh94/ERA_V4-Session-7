import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CIFARNet(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        # Conv1: Normal conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # Conv2: Depthwise Separable conv with stride=2
        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1)

        # Conv3: Dilated conv, dilation=2
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # Conv4: Normal conv, stride=2
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        # GAP + output
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=-1)
