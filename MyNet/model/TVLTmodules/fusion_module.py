# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
import math
from model.TVLTmodules.position_embedding import SinusoidalPositionalEmbedding
from model.TVLTmodules.multihead_attention import MultiheadAttention

class LayerNorm(nn.Module):
    """Layer Normalization class for normalizing layer outputs"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FusionLayer(nn.Module):
    """渐进式特征融合层"""
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        attn_dropout=0.1,
        normalize_before=True
    ):
        super().__init__()
        self.normalize_before = normalize_before
        
        # 自注意力机制
        self.self_attn = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        self.self_attn_layer_norm = LayerNorm(hidden_size)
        
        # 跨模态注意力机制
        self.cross_attn = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        self.cross_attn_layer_norm = LayerNorm(hidden_size)
        
        # 前馈网络
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.final_layer_norm = LayerNorm(hidden_size)
        
        # Dropout settings
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

    def forward(
        self,
        x,                #query
        encoder_out=None, #keyvalue 对应MultiModalFusionEncoder传入的x_k
        encoder_mask=None,
        self_attn_mask=None,
        layer_idx=None
    ):
        """
        Forward pass for FusionLayer
        Args:
            x: Input features
            encoder_out: Encoder output for cross-modal attention
            encoder_mask: Encoder attention mask
            self_attn_mask: Self-attention mask
            layer_idx: Layer index for progressive fusion
        """
        residual = x
        
        # 1. 自注意力（同模态）
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        #  Q、K、V 都来自同一个模态，即输入 x。用于在同一模态内捕获特征之间的依赖关系
        x, _ = self.self_attn(query=x, key=x, value=x)  
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
            
        # 2. 跨模态注意力
        if encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.cross_attn_layer_norm(x)
            # Q、K、V 来自不同模态，即输入 encoder_out。用于在不同模态之间捕获特征之间的依赖关系
            x, _ = self.cross_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                # attn_mask=encoder_mask
            )  
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.cross_attn_layer_norm(x)
        
        # 3. 前馈网络
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
            
        return x

class MultiModalFusionEncoder(nn.Module):
    """改进的多模态融合编码器，支持渐进式特征融合和跨层跳跃链接"""
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        num_layers=12,
        fusion_layers=4,  # 控制开始融合的层数
        stride_layer=2,   # 控制跳跃连接的步长
        fusion_type='concat',
        dropout=0.1,
        relu_dropout=0.1,
        res_dropout=0.1,
        attn_dropout=0.1,
        embed_dropout=0.1,
        attn_mask=False,
        normalize_before=True
    ):
        super().__init__()
        # Initialize parameters
        self.hidden_size = hidden_size
        self.fusion_type = fusion_type
        self.num_heads = num_heads
        self.fusion_layers = fusion_layers
        self.stride_layer = stride_layer
        self.normalize_before = normalize_before
        
        # Embedding setup
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size)
        self.dropout = embed_dropout
        
        # Create progressive fusion layers
        self.layers = nn.ModuleList([
            FusionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_dropout=attn_dropout,
                normalize_before=normalize_before
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = LayerNorm(hidden_size)
        
        # Linear layer for feature fusion
        if fusion_type == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                LayerNorm(hidden_size)
            )

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Forward pass for MultiModalFusionEncoder
        Args:
            x_in: Main input features [batch_size, seq_len, hidden_size]
            x_in_k: Optional key input features
            x_in_v: Optional value input features
        """
        # 1. Positional encoding and embedding
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 2. Handle cross-modal inputs
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # 3. 渐进式特征融合
        # 访问之前所有层的中间表示
        # 灵活地选择如何融合这些历史信息
        # 在不影响自注意力处理的情况下实现特征重用
        start_fusion_layer = max(0, len(self.layers) - self.fusion_layers)
        cross_attn_states = []  # 存储交叉注意力的中间状态
        
        for i, layer in enumerate(self.layers):
            # 存储交叉注意力的中间状态（按照stride_layer的间隔）
            if i >= start_fusion_layer and (i - start_fusion_layer) % self.stride_layer == 0:
                # 在处理当前层之前存储，保存的是上一层的输出
                cross_attn_states.append(x)
            
            # 分情况处理：自注意力或交叉注意力
            if x_in_k is not None and x_in_v is not None and i >= start_fusion_layer:
                # 交叉注意力处理 attention -> dropout -> residual -> norm
                # 保留原始输入，在每个融合层和跳跃连接后，添加残差连接
                residual = x
                x = layer(x, x_k, x_v)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x
                x = self.layer_norm(x)
                
                # 应用跨层跳跃连接（按照stride_layer的间隔且有足够的历史状态）
                if len(cross_attn_states) > 1 and (i - start_fusion_layer + 1) % self.stride_layer == 0:
                    # 融合之前所有层的中间表示 当前状态 + 历史状态
                    if self.fusion_type == 'concat':
                        # 连接所有之前的状态 
                        all_states = torch.cat([x] + cross_attn_states[:-1], dim=-1)
                        x = self.fusion_layer(all_states)
                    elif self.fusion_type == 'add':
                        # 先计算历史状态的加权平均值，然后通过fusion_layer处理
                        num_states = len(cross_attn_states)
                        avg_state = sum(state * (1.0 / num_states) for state in cross_attn_states)
                        # 将当前状态和平均历史状态拼接，通过fusion_layer处理
                        fused = torch.cat([x, avg_state], dim=-1)
                        x = self.fusion_layer(fused)
                    else:  # gate
                        # 使用门控机制融合所有之前的状态
                        all_states = torch.cat([x] + cross_attn_states[:-1], dim=-1)
                        gate = torch.sigmoid(self.fusion_layer(all_states))
                        x = gate * x + (1 - gate) * torch.mean(torch.stack(cross_attn_states[:-1]), dim=0)
            else:
                # 自注意力处理（保持独立的逻辑）
                residual = x
                x = layer(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x
                x = self.layer_norm(x)
        
        # 4. Final layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
        return x