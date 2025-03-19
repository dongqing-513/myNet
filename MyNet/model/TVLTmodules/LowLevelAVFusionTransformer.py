import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from model.TVLTmodules.multihead_attention import MultiheadAttention  # 导入优化的注意力机制
# 导入融合模块所需的组件
from model.TVLTmodules.fusion_module import MultiModalFusionEncoder, FusionLayer, LayerNorm
from model.TVLTmodules.position_embedding import SinusoidalPositionalEmbedding
import math
import importlib.util
import sys
sys.path.append('/home/mz/demo/MyNet')
from model.config import config as config_func
from types import SimpleNamespace
import os
# 更新导入路径
from tool.visualization import AttentionVisualizer

def get_default_config():
    """获取默认配置，并转换为SimpleNamespace对象"""
    config_dict = {}
    config_func(config_dict)
    return SimpleNamespace(**config_dict)

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



# 自定义的Block类，实际使用优化后的FusionLayer
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None):
        super().__init__()

        # 处理配置参数
        if config is None:
            # 如果没有传入配置，使用默认配置和传入的参数
            config_obj = get_default_config()
            config = {
                'hidden_size': dim,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'drop_rate': drop,
                'attention_dropout': attn_drop,
                'relu_dropout': getattr(config_obj, 'relu_dropout', 0.1),
                'res_dropout': getattr(config_obj, 'res_dropout', 0.1),
                'num_groups': getattr(config_obj, 'num_groups', 4),
                'reduction_ratio': getattr(config_obj, 'reduction_ratio', 16),
                'normalize_before': getattr(config_obj, 'normalize_before', True),
            }
        elif isinstance(config, dict):
            # 已经是字典形式，确保包含所有必要的键
            if 'hidden_size' not in config:
                config['hidden_size'] = dim
            if 'num_heads' not in config:
                config['num_heads'] = num_heads
            # 其他参数采用默认配置中的值
            config_obj = get_default_config()
            for key in ['relu_dropout', 'res_dropout', 'num_groups', 'reduction_ratio', 'normalize_before']:
                if key not in config:
                    config[key] = getattr(config_obj, key, 0.1)  # 默认值大多为0.1，除了normalize_before

        # 使用优化后的FusionLayer
        self.fusion_layer = FusionLayer(config=config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, encoder_out=None, mask=None, layer_idx=None):
        # 处理输入形状转换 [B, N, C] -> [N, B, C]
        x_shape = x.shape
        if len(x_shape) == 3 and x_shape[0] != x_shape[1]:  # 如果是[B, N, C]格式
            x = x.permute(1, 0, 2)
            if encoder_out is not None and len(encoder_out.shape) == 3:
                encoder_out = encoder_out.permute(1, 0, 2)

        # 调用优化后的FusionLayer
        x = self.fusion_layer(x, encoder_out, encoder_mask=mask, layer_idx=layer_idx)

        # 处理输出形状转换 [N, B, C] -> [B, N, C]
        if len(x_shape) == 3 and x_shape[0] != x_shape[1]:
            x = x.permute(1, 0, 2)

        return x


# 替换为优化后的自注意力机制适配类
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., config=None):
        super().__init__()

        # 处理配置参数
        if config is None:
            # 如果没有传入配置，使用默认配置和传入的参数
            config_obj = get_default_config()
            config = {
                'num_groups': getattr(config_obj, 'num_groups', 4),
                'reduction_ratio': getattr(config_obj, 'reduction_ratio', 16),
                'hidden_size': dim,
                'num_heads': num_heads,
                'drop_rate': attn_drop,
                'attention_dropout': attn_drop,
                'relu_dropout': getattr(config_obj, 'relu_dropout', 0.1),
                'res_dropout': getattr(config_obj, 'res_dropout', 0.1)
            }
        elif isinstance(config, dict):
            # 已经是字典形式，确保包含所有必要的键
            if 'hidden_size' not in config:
                config['hidden_size'] = dim
            if 'num_heads' not in config:
                config['num_heads'] = num_heads
            # 其他参数采用默认配置中的值
            config_obj = get_default_config()
            for key in ['relu_dropout', 'res_dropout', 'num_groups', 'reduction_ratio']:
                if key not in config:
                    config[key] = getattr(config_obj, key, 0.1)

        # 使用优化后的MultiheadAttention
        self.attn = MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_drop,
            bias=qkv_bias,
            use_optimized=True,            # 启用优化版本
            config=config
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # MultiheadAttention期望输入形状为[seq_len, batch_size, embed_dim]
        # 而当前输入为[batch_size, seq_len, embed_dim]，需要进行转置
        x = x.permute(1, 0, 2)  # [N, B, C]

        # 调用优化后的MultiheadAttention
        x, attn_weights = self.attn(query=x, key=x, value=x, attn_mask=mask, need_weights=self.save_attention)

        # 返回注意力权重用于可视化
        attention_output = (x, attn_weights if self.save_attention else None)

        # 转回原始形状
        x = x.permute(1, 0, 2)  # [B, N, C]
        x = self.proj_drop(x)
        return x, attn_weights if self.save_attention else None


class BottleAttentionNet(nn.Module):
    """
    瓶颈注意力网络(BottleAttentionNet)：使用少量可学习的融合令牌作为信息桥梁，实现高效多模态融合

    实现了MMFE(MultiModal Fusion Encoder)的设计理念，包括：
    1. 参数共享：各层Transformer复用相同参数，降低模型大小
    2. 渐进式融合：音频→融合令牌→视频的信息流动路径
    3. 瓶颈架构：通过少量令牌压缩跨模态信息，降低计算复杂度
    """
    def __init__(self, embed_dim=768, config=None, enable_visualization=False, visualizer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq = 8  # fusion tokens数量
        self.layer_unimodal = 4  # 单模态transformer层数
        self.layer_multimodal = 2  # 多模态transformer层数

        # 处理配置参数
        if config is None:
            # 如果没有传入配置，使用默认配置
            config_obj = get_default_config()
        elif isinstance(config, dict):
            # 如果传入的是字典，转换为SimpleNamespace
            config_obj = SimpleNamespace(**config)
        else:
            # 已经是对象形式
            config_obj = config

        # 获取配置参数值
        self.embed_dim = getattr(config_obj, 'hidden_size', embed_dim)

        # 初始化可视化工具
        vis_dir = getattr(config_obj, 'visualization_dir', 'visualizations/bottleneck')
        self.enable_visualization = enable_visualization
        self.visualizer = visualizer if self.enable_visualization else None

        # 将config转换为字典，方便后续处理
        if isinstance(config, SimpleNamespace):
            self.config = vars(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = vars(config_obj)

        # 投影层
        self.audio_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.visual_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # MLP层
        self.audio_mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim * 4,
            out_features=self.embed_dim,
            drop=getattr(config_obj, 'drop_rate', 0.1)
        )
        self.visual_mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim * 4,
            out_features=self.embed_dim,
            drop=getattr(config_obj, 'drop_rate', 0.1)
        )

        # 可学习的fusion tokens
        self.fusion_tokens = nn.Parameter(torch.randn(self.seq, 1, self.embed_dim))

        # 直接创建增强的单模态编码器，删除模态特定层
        # 增加编码器层数，从原来的layer_unimodal增加到layer_unimodal+2
        self.audioEncoder = MultiModalFusionEncoder(
            config=self.config,
            num_layers=self.layer_unimodal,  # 增加层数补偿删除的模态特定层
            fusion_layers=self.layer_unimodal,  # 所有层都设为融合层
            stride_layer=1,                     # 将步长设为1，允许更频繁的特征重用
            fusion_type='gate',                 # 使用门控融合机制增强特征交互
            normalize_before=self.config.get('normalize_before', True),
            share_parameters=True,              # 在单模态内部启用参数共享
            attn_mask=False,                    # 关闭注意力掩码，允许全部token交互
            embed_dropout=0.1,                  # 设置嵌入层dropout
            res_dropout=0.1                     # 设置残差连接dropout
        )
        self.visualEncoder = MultiModalFusionEncoder(
            config=self.config,
            num_layers=self.layer_unimodal,  # 增加层数补偿删除的模态特定层
            fusion_layers=self.layer_unimodal,  # 所有层都设为融合层
            stride_layer=1,  # 更频繁的特征重用
            fusion_type='gate',  # 使用门控融合机制增强特征交互
            normalize_before=self.config.get('normalize_before', True),
            share_parameters=True,              # 在单模态内部启用参数共享
            attn_mask=False,                    # 关闭注意力掩码，允许全部token交互
            embed_dropout=0.1,                  # 设置嵌入层dropout
            res_dropout=0.1                     # 设置残差连接dropout
        )

        # 多模态融合层 - 使用参数共享
        self.fusion_encoder = MultiModalFusionEncoder(
            config=self.config,
            num_layers=self.layer_multimodal,
            fusion_layers=self.layer_multimodal,  # 将所有层都设为融合层，增强特征交互
            stride_layer=1,                       # 将步长设为1，允许更频繁的特征重用
            fusion_type='gate',                   # 使用门控融合机制
            normalize_before=self.config.get('normalize_before', True),
            share_parameters=True,                # 启用参数共享以减少参数量
            attn_mask=False,                      # 关闭注意力掩码，允许全部token交互
            embed_dropout=0.1,                    # 设置嵌入层dropout
            res_dropout=0.1                       # 设置残差连接dropout
        )

    def forward(self, audio, visual, return_attention=False):
        # 1. 特征预处理
        # 输入尺寸：
        # audio: [B, A, 768] - 音频特征
        # visual: [B, V, 768] - 视频特征

        # 2. 特征投影和非线性变换
        # 2.1 线性投影
        audio = self.audio_linear(audio)  # [B, A, C]
        visual = self.visual_linear(visual)  # [B, V, C]

        # 如果启用可视化功能，则记录线性投影的结果
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.register_features('linear_audio', audio)
            self.visualizer.register_features('linear_visual', visual)

        # 2.2 MLP非线性变换
        audio = self.audio_mlp(audio)  # [B, A, C]
        visual = self.visual_mlp(visual)  # [B, V, C]

        # 如果启用可视化功能，则记录MLP的结果
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.register_features('mlp_audio', audio)
            self.visualizer.register_features('mlp_visual', visual)

        # 3. 单模态特征增强
        audio = self.audioEncoder(audio)  # [B, A, C]
        visual = self.visualEncoder(visual)  # [B, V, C]

        # 如果启用可视化功能，则记录单模态增强后的特征
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.register_features('encoder_audio', audio)
            self.visualizer.register_features('encoder_visual', visual)

        # 4. 准备fusion tokens
        batch_size = audio.shape[0]
        fusion_tokens = self.fusion_tokens.expand(-1, batch_size, -1)  # [F, B, C]

        # 5. 编排所有特征以准备特征融合
        # 所有模态数据在同一注意力空间中直接交互
        # 将音频、融合tokens、视频特征连接起来
        audio = audio.permute(1, 0, 2)       # [A, B, C]
        visual = visual.permute(1, 0, 2)      # [V, B, C]
        fusion_tokens = fusion_tokens.contiguous()  # [F, B, C]

        # 注意连接顺序：先fusion tokens，然后音频，最后视频
        # 这样fusion tokens可以同时与两种模态交互，成为桌面信息桥梁
        multimodal_input = torch.cat([fusion_tokens, audio, visual], dim=0)  # [F+A+V, B, C]

        # 如果启用可视化功能，则记录融合前的输入
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.register_features('multimodal_input', multimodal_input)

        # 6. 单步多模态融合
        # 直接在一个空间中融合所有模态，减少中间步骤
        if return_attention and self.enable_visualization:
            multimodal_output, attention_weights = self.fusion_encoder(multimodal_input, return_attention=True)

            # 注册注意力权重以供可视化
            if self.visualizer is not None:
                # 注册各层的注意力权重
                for layer_name, layer_weights in attention_weights.items():
                    if 'self_attn' in layer_weights:
                        self.visualizer.register_attention_weights(f'fusion_{layer_name}_self', layer_weights['self_attn'])
                    if 'cross_attn' in layer_weights:
                        self.visualizer.register_attention_weights(f'fusion_{layer_name}_cross', layer_weights['cross_attn'])

                # 提取和注册特定的跨模态注意力
                try:
                    # 获取最后一层的注意力
                    last_layer_name = f'layer_{self.layer_multimodal-1}'
                    if last_layer_name in attention_weights and 'self_attn' in attention_weights[last_layer_name]:
                        last_attn = attention_weights[last_layer_name]['self_attn']

                        # 计算各部分的序列长度
                        seq_len = self.seq  # fusion tokens长度
                        audio_len = audio.size(0)  # 音频序列长度
                        visual_len = visual.size(0)  # 视频序列长度

                        # 提取音频和视频之间的注意力
                        audio_start = seq_len
                        audio_end = seq_len + audio_len
                        visual_start = audio_end
                        visual_end = audio_end + visual_len

                        # 计算音频对视频的注意力
                        if last_attn.dim() >= 3 and audio_start < last_attn.size(1) and visual_start < last_attn.size(2):
                            audio_to_visual = last_attn[:, audio_start:audio_end, visual_start:visual_end]
                            self.visualizer.register_attention_weights('audio_to_visual', audio_to_visual)

                            # 计算视频对音频的注意力
                            visual_to_audio = last_attn[:, visual_start:visual_end, audio_start:audio_end]
                            self.visualizer.register_attention_weights('visual_to_audio', visual_to_audio)

                            # 计算融合tokens对音频和视频的注意力
                            fusion_to_audio = last_attn[:, :seq_len, audio_start:audio_end]
                            fusion_to_visual = last_attn[:, :seq_len, visual_start:visual_end]
                            self.visualizer.register_attention_weights('fusion_to_audio', fusion_to_audio)
                            self.visualizer.register_attention_weights('fusion_to_visual', fusion_to_visual)

                            # 计算音频和视频对融合tokens的注意力
                            audio_to_fusion = last_attn[:, audio_start:audio_end, :seq_len]
                            visual_to_fusion = last_attn[:, visual_start:visual_end, :seq_len]
                            self.visualizer.register_attention_weights('audio_to_fusion', audio_to_fusion)
                            self.visualizer.register_attention_weights('visual_to_fusion', visual_to_fusion)
                except Exception as e:
                    print(f"提取跨模态注意力时出错: {str(e)}")
        else:
            multimodal_output = self.fusion_encoder(multimodal_input)

        # 如果启用可视化功能，则记录融合后的输出
        if self.enable_visualization and self.visualizer is not None:
            self.visualizer.register_features('multimodal_output', multimodal_output)

        # 7. 直接使用完整的多模态输出作为结果，F+A+V 为8+196+1568=1772
        # 将结果转回原始形状 [F+A+V, B, C] -> [B, F+A+V, C]
        result = multimodal_output.permute(1, 0, 2)  # [B, F+A+V, C]

        # 如果启用可视化功能，则记录各部分特征
        if self.enable_visualization and self.visualizer is not None:
            fusion_part = multimodal_output[:self.seq]  # [F, B, C]
            audio_part = multimodal_output[self.seq:self.seq+audio.size(0)]  # [A, B, C]
            visual_part = multimodal_output[self.seq+audio.size(0):]  # [V, B, C]

            self.visualizer.register_features('fusion_tokens', fusion_part)
            self.visualizer.register_features('output_audio', audio_part)
            self.visualizer.register_features('output_visual', visual_part)

        return result  # 保持与编码器的输出形状一致
