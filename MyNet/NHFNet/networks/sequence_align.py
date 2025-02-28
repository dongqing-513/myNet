import torch
import torch.nn as nn

class SequenceAlignmentModule(nn.Module):
    """序列对齐模块，支持两种实现方式：
    1. 深度可分离卷积实现（use_lightweight=False）：
       - 使用深度可分离卷积进行降采样，参数量更少
       - 两层结构：1576->394->197
       
    2. 轻量级实现（use_lightweight=True）：
       - 使用特征增强、池化和残差连接
       - 包含通道注意力机制
    """
    def __init__(self, embed_dim, use_lightweight=False):
        super(SequenceAlignmentModule, self).__init__()
        self.embed_dim = embed_dim
        self.use_lightweight = use_lightweight

        # ====== 原始序列长度对齐模块 ======
        self.length_adapt = nn.ModuleList([
            nn.Sequential(
                # 第一层：1576 -> 394 使用深度可分离卷积
                # 步骤1：深度卷积 - 每个输入通道独立卷积
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    groups=self.embed_dim  # 使每个通道独立卷积
                ),
                # 步骤2：逐点卷积 - 1x1卷积实现通道混合
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                # 第二层：394 -> 197 使用深度可分离卷积
                # 步骤1：深度卷积
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    groups=self.embed_dim  # 使每个通道独立卷积
                ),
                # 步骤2：逐点卷积
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        ])

        # 原始特征归一化层
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        ])

        # ====== 轻量级序列长度对齐模块 ======
        if self.use_lightweight:
            # 1. 特征增强卷积
            self.feature_enhance = nn.ModuleList([
                nn.Sequential(
                    # 第一层：增强局部特征，不改变序列长度
                    nn.Conv1d(self.embed_dim, self.embed_dim,
                             kernel_size=3, stride=1, padding=1, groups=8),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ),
                nn.Sequential(
                    # 第二层：增强局部特征，不改变序列长度
                    nn.Conv1d(self.embed_dim, self.embed_dim,
                             kernel_size=3, stride=1, padding=1, groups=8),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            ])

            # 2. 高效池化层
            self.pool_layers = nn.ModuleList([
                nn.Sequential(
                    # 第一次降采样：1576 -> 394
                    nn.AvgPool1d(kernel_size=4, stride=4),
                    nn.BatchNorm1d(self.embed_dim)
                ),
                nn.Sequential(
                    # 第二次降采样：394 -> 197
                    nn.AvgPool1d(kernel_size=2, stride=2),
                    nn.BatchNorm1d(self.embed_dim)
                )
            ])

            # 3. 残差连接
            self.skip_connections = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(self.embed_dim, self.embed_dim,
                             kernel_size=1, stride=4),
                    nn.BatchNorm1d(self.embed_dim)
                ),
                nn.Sequential(
                    nn.Conv1d(self.embed_dim, self.embed_dim,
                             kernel_size=1, stride=2),
                    nn.BatchNorm1d(self.embed_dim)
                )
            ])

            # 4. 特征重校准
            self.channel_se = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(self.embed_dim, self.embed_dim // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.embed_dim // 4, self.embed_dim, 1),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(self.embed_dim, self.embed_dim // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.embed_dim // 4, self.embed_dim, 1),
                    nn.Sigmoid()
                )
            ])

    def forward(self, x):
        """
        输入: x shape [batch, seq_len, dim]
        输出: aligned_x shape [batch, new_seq_len, dim]
        """
        # 准备降采样：[batch, seq_len, dim] -> [batch, dim, seq_len]
        x = x.transpose(1, 2)

        if not self.use_lightweight:
            # ====== 深度可分离卷积 ======
            # 第一层降采样：1576 -> 394
            x = self.length_adapt[0](x)
            # [batch, dim, seq_len] -> [batch, seq_len, dim]
            x = x.transpose(1, 2)
            x = self.layer_norms[0](x)

            # [batch, seq_len, dim] -> [batch, dim, seq_len]
            x = x.transpose(1, 2)
            # 第二层降采样：394 -> 197
            x = self.length_adapt[1](x)
            # [batch, dim, seq_len] -> [batch, seq_len, dim]
            x = x.transpose(1, 2)
            x = self.layer_norms[1](x)

        else:
            # ====== 使用轻量级实现 ======
            for j in range(2):  # 两次降采样
                # 特征增强
                enhanced = self.feature_enhance[j](x)

                # 通道注意力
                se_weight = self.channel_se[j](enhanced)
                enhanced = enhanced * se_weight

                # 降采样
                pooled = self.pool_layers[j](enhanced)

                # 残差连接
                residual = self.skip_connections[j](x)
                x = pooled + residual

                # 转置回来进行归一化
                if j == 1 or (j == 0 and not self.training):
                    # [batch, dim, seq_len] -> [batch, seq_len, dim]
                    x = x.transpose(1, 2)
                    x = self.layer_norms[j](x)
                    if j < 1:  # 如果不是最后一层，转回去继续处理
                        # [batch, seq_len, dim] -> [batch, dim, seq_len]
                        x = x.transpose(1, 2)

        return x  # 返回 [batch, new_seq_len, dim]
