import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(
                ~mask[:, None, None, :].bool(), -1e12)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BottleAttentionNet(nn.Module):
    def __init__(self):
        super(BottleAttentionNet, self).__init__()
        # 基础参数设置
        self.embed_dim = 768  # 与输入维度匹配
        self.seq = 8  # fusion tokens数量
        self.layer_unimodal = 2  # 单模态transformer层数
        self.layer_multimodal = 2  # 多模态transformer层数

        # 投影层
        self.audio_linear = nn.Linear(768, self.embed_dim)
        self.visual_linear = nn.Linear(768, self.embed_dim)

        # MLP层
        self.audio_mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim * 4,
            out_features=self.embed_dim,
            drop=0.1
        )
        self.visual_mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim * 4,
            out_features=self.embed_dim,
            drop=0.1
        )

        # 可学习的fusion tokens
        self.fusion_tokens = nn.Parameter(torch.randn(self.seq, 1, self.embed_dim))

        # 创建单模态Transformer Block
        self.unimodal_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                norm_layer=nn.LayerNorm
            ) for _ in range(self.layer_unimodal)
        ])

        # 创建多模态Transformer Block
        self.multimodal_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                norm_layer=nn.LayerNorm
            ) for _ in range(self.layer_multimodal)
        ])

    def forward(self, audio, visual):
        # 预期输入尺寸：
        # audio: [1, 196, 768]
        # visual: [1, 1568, 768]

        # 1. 投影到embedding空间（可选，因为维度已经匹配）
        audio = self.audio_linear(audio).permute(1, 0, 2)  # [196, 1, 768]
        visual = self.visual_linear(visual).permute(1, 0, 2)  # [1568, 1, 768]

        # 2. 应用MLP层
        audio = self.audio_mlp(audio)  # [196, 1, 768]
        visual = self.visual_mlp(visual)  # [1568, 1, 768]

        # 3. 单模态特征提取
        for i in range(self.layer_unimodal):
            audio = self.unimodal_blocks[i](audio)
            visual = self.unimodal_blocks[i](visual)

        # 4. 扩展fusion tokens以匹配batch size
        fusion_tokens = self.fusion_tokens.expand(-1, audio.size(1), -1)  # [8, 1, 768]

        # 5. 第一阶段多模态融合：audio + fusion tokens
        x = torch.cat([audio, fusion_tokens], dim=0)  # [204, 1, 768] (196 + 8)

        # 6. 多模态特征融合
        for i in range(self.layer_multimodal):
            if i == 0:
                # 处理音频和fusion tokens
                x = self.multimodal_blocks[i](x)
                # 获取融合后的tokens并与视频特征拼接
                fused_tokens = x[audio.size(0):,:,:]  # [8, 1, 768]
                x = torch.cat([fused_tokens, visual], dim=0)  # [1576, 1, 768] (8 + 1568)
                x = self.multimodal_blocks[i](x)
            else:
                x = self.multimodal_blocks[i](x)

        # 7. 转换回原始形状 [1576, 1, 768] -> [1, 1576, 768]
        # batch size 序列长度（8个fusion tokens + 1568个视频特征）特征维度
        x = x.permute(1, 0, 2)

        return x  # 最终融合的音视频特征 [1, 1576, 768]
