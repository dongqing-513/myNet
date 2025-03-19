import torch
import torch.nn as nn

class SequenceAlignmentModule(nn.Module):
    """序列对齐模块，支持三种实现方式：
    1. 深度可分离卷积模式（use_lightweight=False, use_hybrid=False）：
       - 优点：参数量少，计算效率高，直接通过可分离卷积降采样
       - 缺点：特征提取能力相对有限，缺乏特征自适应能力
       - 适用场景：资源受限、实时性要求高的环境
       - 两层结构：1576->394->197
        
    2. 特征增强模式（use_lightweight=True, use_hybrid=False）：
       - 优点：具有通道注意力机制，增强特征表达，保留重要信息
       - 缺点：计算复杂度较高，参数量增加，训练时间较长
       - 适用场景：需要高质量特征表示，对计算资源要求不严格的场合
       - 增强特征提取能力，同时使用残差连接保留原始信息
       
    3. 混合动态融合模式（use_hybrid=True）：
       - 优点：结合前两种方法的优势，动态平衡效率和准确率
       - 缺点：实现最复杂，需要额外的融合门控制参数
       - 适用场景：需要平衡准确率和效率的应用，具有充足计算资源
       - 使用可学习的融合门自适应调整不同路径的权重
    """
    def __init__(self, embed_dim, config):
        super(SequenceAlignmentModule, self).__init__()
        self.embed_dim = embed_dim
        self.config = config

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
        if self.config.get('use_lightweight', False):
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

        # ====== 混合模式序列长度对齐模块 ======
        if self.config.get('use_hybrid', False):
            # 深度可分离卷积路径
            self.depthwise_path = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=self.config.get('downsample_ratio', 4), 
                         stride=self.config.get('downsample_ratio', 4), groups=embed_dim),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU()
            )

            # 轻量级增强路径
            self.enhance_path = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=8),
                nn.GELU()
            )

            # 自适应池化
            self.adaptive_pool = nn.AdaptiveAvgPool1d(
                int(1 / self.config.get('downsample_ratio', 4) * 1000)  # 假设初始序列长度为1000
            )

            # 动态融合门
            self.fusion_gate = nn.Sequential(
                nn.Conv1d(2*embed_dim, embed_dim//2, 1),
                nn.ReLU(),
                nn.Conv1d(embed_dim//2, 1, 1),
                nn.Sigmoid()
            )

            # 残差连接
            self.skip_conv = nn.Conv1d(embed_dim, embed_dim, 
                                      kernel_size=1, stride=self.config.get('downsample_ratio', 4))

            # 归一化
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        输入: x shape [batch, seq_len, dim]
        输出: aligned_x shape [batch, new_seq_len, dim]
        """
        # 准备降采样：[batch, seq_len, dim] -> [batch, dim, seq_len]
        x = x.transpose(1, 2)

        if self.config.get('use_hybrid', False):
            # ====== 混合动态融合模式 ======
            # 深度可分离路径
            dw_path = self.depthwise_path(x)

            # 特征增强路径
            enh = self.enhance_path(x)
            lw_path = self.adaptive_pool(enh)

            # 动态融合
            concat = torch.cat([dw_path, lw_path], dim=1)
            alpha = self.fusion_gate(concat)
            fused = alpha * dw_path + (1 - alpha) * lw_path

            # 残差连接
            fused = fused + self.skip_conv(x)

            # 归一化
            fused = fused.transpose(1, 2)
            return self.norm(fused)
        
        elif self.config.get('use_lightweight', False):
            # ====== 特征增强模式 ======
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
        else:
            # ====== 深度可分离卷积模式 ======
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

        return x  # 返回 [batch, new_seq_len, dim]
