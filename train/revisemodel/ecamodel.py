"""
EEGNet v5 + ECA（单点改动版，兼容 Python ≤3.9）
基于: Lawhern et al., 2018
改动点：在 conv_sep→bn3 之后加入 ECA 注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention: GAP → 1D Conv → Sigmoid"""
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        padding = (k_size - 1) // 2
        # 输入视为 (B, 1, C)，做局部跨通道卷积建模通道相关性
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=padding, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                    # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)     # (B, 1, C)
        y = self.conv1d(y)                      # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)   # (B, C, 1, 1)
        y = torch.sigmoid(y)
        return x * y


class EEGNetECA(nn.Module):
    def __init__(
        self,
        n_chans,
        n_outputs,
        input_window_samples,
        final_conv_length="auto",   
        F1=8,                       # Temporal filters
        D=2,                        # Depth multiplier
        F2=None,                    # 若为 None，则自动设为 F1*D
        kernel_length=64,
        pool_mode="mean",           # 'mean' or 'max'
        dropout_rate=0.25,
        eca_k=3,                    # ECA 1D卷积核大小(3/5 常用)
    ):
        super(EEGNetECA, self).__init__()

        if F2 is None:
            F2 = F1 * D
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # ======================
        # 1️⃣ Temporal Convolution
        # ======================
        self.conv_time = nn.Conv2d(
            1, F1, (1, kernel_length),
            stride=1, padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ======================
        # 2️⃣ Depthwise Convolution (Spatial Filtering)
        # ======================
        self.conv_depth = nn.Conv2d(
            F1, F1 * D, (n_chans, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        if pool_mode == "mean":
            self.pool1 = nn.AvgPool2d((1, 4))
        else:
            self.pool1 = nn.MaxPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ======================
        # 3️⃣ Separable Convolution
        # ======================
        self.conv_sep = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      groups=F1 * D, padding=(0, 8), bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)

        # === ✅ 唯一新增：ECA 注意力（插在 bn3 之后） ===
        self.eca = ECA(F2, k_size=eca_k)

        if pool_mode == "mean":
            self.pool2 = nn.AvgPool2d((1, 8))
        else:
            self.pool2 = nn.MaxPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ======================
        # 4️⃣ Classifier
        # ======================
        self.final_conv_length = final_conv_length

        # 计算卷积后 feature 尺寸
        with torch.no_grad():
            dummy = torch.ones((1, 1, n_chans, input_window_samples))
            out = self.forward_features(dummy)
            n_features = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Linear(n_features, n_outputs)

    def forward_features(self, x):
        x = self.conv_time(x)
        x = self.bn1(x)
        x = self.conv_depth(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_sep(x)
        x = self.bn3(x)
        x = self.eca(x)     # ← 唯一新增行：ECA 注意力
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        # 输入: (batch, chans, time)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (batch, 1, chans, time)
        x = self.forward_features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================
# ✅ 测试主程序
# ============================================================
if __name__ == "__main__":
    # 模拟输入 (batch, chans, time)
    n_chans = 22
    n_outputs = 4
    input_window_samples = 1125

    model = EEGNetECA(
        n_chans=n_chans,
        n_outputs=n_outputs,
        input_window_samples=input_window_samples,
        pool_mode="mean",
        eca_k = 5,    # 可调 3 或 5
    )

    print("✅ 模型结构:")
    print(model)
    print("=" * 80)

    # 随机输入测试
    x = torch.randn(2, n_chans, input_window_samples)
    y = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", y.shape)
    print("✅ 前向传播成功")
