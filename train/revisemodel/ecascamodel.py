"""
EEGNet v5 + ECA + Spatial Channel Attention（单点改动版）
在 conv_time → bn1 之后对 EEG 通道维 (C) 做 SE 式注意（与 ECA 的特征通道注意互补）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        padding = (k_size - 1) // 2
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=padding, bias=False)
    def forward(self, x):
        y = self.avg_pool(x); y = y.squeeze(-1).transpose(-1,-2)
        y = self.conv1d(y); y = y.transpose(-1,-2).unsqueeze(-1)
        y = torch.sigmoid(y); return x * y


class SpatialChannelSE(nn.Module):
    """
    对 EEG 通道维 (C) 的注意力：先对 (F1,T) 平均，得到 (B,C)，
    再 MLP 生成 C 维权重，Sigmoid 后回乘到原张量 (B,F1,C,T) 上。
    """
    def __init__(self, n_chans, reduction=4):
        super().__init__()
        hidden = max(1, n_chans // reduction)
        self.fc1 = nn.Linear(n_chans, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_chans, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
    def forward(self, x):
        B, F1, C, T = x.shape
        s = x.mean(dim=(1,3))           # (B,C)
        w = self.fc2(self.act(self.fc1(s)))  # (B,C)
        w = self.gate(w).view(B,1,C,1)       # (B,1,C,1)
        return x * w


class EEGNetECA_SCA(nn.Module):
    """EEGNet + ECA + 空间通道注意（对 EEG 通道维）"""
    def __init__(
        self,
        n_chans,
        n_outputs,
        input_window_samples,
        final_conv_length="auto",
        F1=8,
        D=2,
        F2=None,
        kernel_length=64,
        pool_mode="mean",
        dropout_rate=0.25,
        eca_k=3,
    ):
        super(EEGNetECA_SCA, self).__init__()
        if F2 is None: F2 = F1 * D

        # ===== Temporal Convolution =====
        self.conv_time = nn.Conv2d(1, F1, (1, kernel_length),
                                   stride=1, padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.sca = SpatialChannelSE(n_chans)   # ✨ 新增：EEG 通道注意

        # ===== Depthwise =====
        self.conv_depth = nn.Conv2d(F1, F1*D, (n_chans,1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool1 = nn.AvgPool2d((1,4)) if pool_mode=="mean" else nn.MaxPool2d((1,4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ===== Sep + ECA =====
        self.conv_sep = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1,16), groups=F1*D, padding=(0,8), bias=False),
            nn.Conv2d(F1*D, F2, 1, bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.eca = ECA(F2, k_size=eca_k)
        self.pool2 = nn.AvgPool2d((1,8)) if pool_mode=="mean" else nn.MaxPool2d((1,8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ===== Classifier =====
        self.final_conv_length = final_conv_length
        with torch.no_grad():
            dummy = torch.ones((1,1,n_chans,input_window_samples))
            out = self.forward_features(dummy)
            n_features = out.numel()
        self.classifier = nn.Linear(n_features, n_outputs)

    def forward_features(self, x):
        x = self.conv_time(x); x = self.bn1(x)
        x = self.sca(x)                 # ✨ 插入 SCA（作用于 EEG 通道维）
        x = self.conv_depth(x); x = self.bn2(x)
        x = F.elu(x); x = self.pool1(x); x = self.drop1(x)
        x = self.conv_sep(x); x = self.bn3(x); x = self.eca(x)
        x = F.elu(x); x = self.pool2(x); x = self.drop2(x)
        return x

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.forward_features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    n_chans, n_outputs, input_window_samples = 22, 4, 1125
    model = EEGNetECA_SCA(n_chans, n_outputs, input_window_samples, eca_k=5)
    x = torch.randn(2, n_chans, input_window_samples)
    y = model(x)
    print("输入尺寸:", x.shape); print("输出尺寸:", y.shape); print("✅ SCA Forward OK")
