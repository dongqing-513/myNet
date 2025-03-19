import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.TVLTmodules.text_lstm import BERTTextLSTMNet
# 移除旧的路径添加
# import sys
# sys.path.append('..')
# from MSAF import MSAF

from model.TVLTmodules.transformer import TransformerEncoder
from model.TVLTmodules.fusion_module import MultiModalFusionEncoder
from model.TVLTmodules.sequence_align import SequenceAlignmentModule
# 导入可视化工具
from tool.visualization import AttentionVisualizer

class MSAFLSTMNet(nn.Module):
    def __init__(self, _config):
        super(MSAFLSTMNet, self).__init__()

        self.max_feature_layers = 1  # number of layers in unimodal models before classifier

        # NFHNET语音和视频模型层 低阶视频音频模态聚合模块
        # self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = _config["hidden_size"]  # 使用_config获取参数
        # 交叉注意力

        # 确保配置中包含所有必要的参数
        config = {
            'hidden_size': _config['hidden_size'],
            'num_heads': _config['num_heads'],
            'num_layers': _config['num_layers'],
            'drop_rate': _config['drop_rate'],
            'relu_dropout': _config['drop_rate'],
            'res_dropout': _config['drop_rate'],
            'normalize_before': _config['normalize_before'],
            'num_groups': _config.get('num_groups', 4),  # 默认值改为4
            'reduction_ratio': _config.get('reduction_ratio', 12),  # 默认值改为12
            'attn_mask': False
        }

        # 新增：初始化可视化工具类
        vis_dir = _config.get('visualization_dir', 'visualizations/msaf')
        self.enable_visualization = _config.get('enable_visualization', False)
        self.visualizer = AttentionVisualizer(save_dir=vis_dir) if self.enable_visualization else None

        # 新增：获取是否启用参数共享的配置，默认为False
        use_shared_transformer = _config.get('use_shared_transformer', False)

        if use_shared_transformer:
            # 创建共享的transformer编码器
            self.shared_transformer = MultiModalFusionEncoder(
                config=config,
                num_layers=_config['num_layers'],
                fusion_layers=4,  # 控制开始融合的层数
                stride_layer=_config['skip_interval'],
                fusion_type=_config['fusion_type'],
                embed_dropout=_config['drop_rate'],
                share_parameters=True  # 启用参数共享
            )
            # 使用共享的transformer
            self.cross_transformer = self.shared_transformer
            self.classifcation = self.shared_transformer
        else:
            # 原有的实现方式
            self.cross_transformer = MultiModalFusionEncoder(
                config=config,
                num_layers=_config['num_layers'],
                fusion_layers=4,  # 控制开始融合的层数
                stride_layer=_config['skip_interval'],
                fusion_type=_config['fusion_type'],
                embed_dropout=_config['drop_rate']
            )
            """
            self.cross_transformer = TransformerEncoder(
                embed_dim=self.embed_dim,
                num_heads=8,
                layers=4,
                attn_dropout = 0.4,
                attn_mask=False
            )"""


            # 自注意力

            self.classifcation = MultiModalFusionEncoder(
                config=config,  # 直接传入配置对象
                num_layers=_config['num_layers'],
                fusion_layers=4,  # 控制开始融合的层数
                stride_layer=_config['skip_interval'],
                fusion_type=_config['fusion_type'],
                embed_dropout=_config['drop_rate']
            )
        """
        self.classifcation = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=8,
            layers=4,
            attn_mask=False
        )"""

        # 文本模型层
        if "bert" in _config:
            # 进行文本特征提取：Bert+优化的LSTM架构
            self.text_model = BERTTextLSTMNet(config=_config)
            # 填充文本id
            self.text_id = _config["bert"]["id"]

        # 定义visual_id和audio_id
        if "visual" in _config:
            self.visual_id = _config["visual"]["id"]
        if "audio" in _config:
            self.audio_id = _config["audio"]["id"]

        # 归一化
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # 序列长度对齐模块
        self.sequence_align = SequenceAlignmentModule(
            embed_dim=self.embed_dim,
            config=_config  # 传递整个配置对象
        )

        # 高效多视图多尺度池化 (EMMP) - 基于五篇顶级论文的集成设计
        self.pooling = nn.ModuleDict({
            # 特征转换层 - 降低维度减少参数量 (Vision Transformer for Small-Size Datasets, ICLR 2022)
            'transform': nn.Sequential(
                nn.LayerNorm(self.embed_dim),  # 首先标准化特征
                nn.Linear(self.embed_dim, self.embed_dim // 2),  # 降维50%
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            # 多尺度特征提取 - 金字塔特征提取 (Pyramid Vision Transformer, ICCV 2021)
            'pyramid': nn.ModuleDict({
                'level1': nn.Conv1d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=3, stride=1, padding=1),
                'level2': nn.Conv1d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=5, stride=1, padding=2),
                'level3': nn.Conv1d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=7, stride=1, padding=3),
                'level4': nn.Conv1d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=9, stride=1, padding=4),
            }),
            # 注意力池化 - 学习重要时间步 (Non-local Neural Networks, Wang et al., CVPR 2018)
            'attention': nn.Sequential(
                nn.Linear(self.embed_dim // 2, 64),  # 进一步降维
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)  # 对序列维度做归一化
            ),
            # 通道注意力 - 基于ECA-Net (Wang et al., CVPR 2020)
            'channel_attn': nn.Sequential(
                nn.Linear(self.embed_dim // 2, self.embed_dim // 8),
                nn.GELU(),
                nn.Linear(self.embed_dim // 8, self.embed_dim // 2),
                nn.Sigmoid()
            ),
            # 特征融合 - 基于ResMLPs (Touvron et al., NeurIPS 2021)
            'fusion': nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        })

    def forward(self, av, txt, attention_mask, return_attention=False):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                txt = self.text_model(txt, attention_mask)

            # 序列长度对齐，输入已经是 [batch, seq_len, dim]
            audio_visual_feature = self.sequence_align(av)

            # 最终特征归一化
            audio_visual_feature = self.layer_norm(audio_visual_feature)
            txt = self.layer_norm(txt)

            # 如果启用可视化功能，则记录原始特征
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.register_features('text_features', txt)
                self.visualizer.register_features('av_features', audio_visual_feature)

            # txt - self-attention torch.Size([1, 197, 768])
            result1 = self.classifcation(txt)
            # print("\nresult1",result1.shape)

            # 交叉注意力融合
            # T->(V,A)：文本引导的视听注意力
            l_av = self.cross_transformer(audio_visual_feature, result1, result1)
            # (V,A)->T：视听引导的文本注意力
            av_l = self.cross_transformer(result1, audio_visual_feature, audio_visual_feature)
            # print("==========cross_transformer2: (V,A)->T=========done",av_l.shape) # torch.Size([1, 512, 768])

            # 如果启用可视化功能，则记录交叉注意力结果
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.register_features('text_guided_av', l_av)
                self.visualizer.register_features('av_guided_text', av_l)

            # 融合特征的自注意力处理
            l_result = self.classifcation(av_l)
            # print("==========cross_transformer2 -> self-attention=========done",l_result.shape) # torch.Size([1, 512, 768])

            # cross_transformer1 -> self-attention
            av_result = self.classifcation(l_av)
            # print("==========cross_transformer1 -> self-attention========done",av_result.shape) # torch.Size([1, 512, 768])

            # 如果启用可视化功能，则记录自注意力结果
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.register_features('text_self_attention', l_result)
                self.visualizer.register_features('av_self_attention', av_result)

            all = result1 + audio_visual_feature + l_result + av_result
            #print("\ntest:",all.shape) # torch.Size([1, 512, 768])

            # 如果启用可视化功能，则记录融合后的特征
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.register_features('fused_features', all)

            # 1. 特征转换 - 降维
            transformed = self.pooling['transform'](all)  # [B, S, D/2]

            # 2. 多尺度特征提取 (Pyramid Vision Transformer)
            # 将特征转换为[B, D/2, S]以适用于卷积操作
            transformed_t = transformed.transpose(1, 2)  # [B, D/2, S]
            pyramid_features = []
            for level in ['level1', 'level2', 'level3', 'level4']:
                # 应用不同尺度的卷积捕获不同范围的上下文
                level_feat = self.pooling['pyramid'][level](transformed_t)  # [B, D/4, S]
                # 全局最大池化获取最显著的特征
                level_pooled = F.adaptive_max_pool1d(level_feat, 1).squeeze(-1)  # [B, D/4]
                pyramid_features.append(level_pooled)

            # 3. 注意力池化 - 学习重要时间步 (Non-local Neural Networks)
            attn_weights = self.pooling['attention'](transformed)  # [B, S, 1]
            attn_pooled = torch.bmm(attn_weights.transpose(1, 2), transformed).squeeze(1)  # [B, D/2]

            # 4. 全局统计池化 - 补充全局信息
            max_pooled = torch.max(transformed, dim=1)[0]  # [B, D/2]
            avg_pooled = torch.mean(transformed, dim=1)  # [B, D/2]

            # 5. 通道注意力增强 (ECA-Net)
            channel_weights = self.pooling['channel_attn'](avg_pooled)  # [B, D/2]
            enhanced_pooled = attn_pooled * channel_weights + 0.2 * (max_pooled + avg_pooled)  # [B, D/2]

            # 6. 特征集成 - 组合多尺度和注意力增强的特征
            multi_scale = torch.cat(pyramid_features, dim=1)  # [B, D]
            pooled_features = torch.cat([enhanced_pooled, multi_scale[:, :self.embed_dim//2]], dim=1)  # [B, D]

            # 7. 特征增强与残差连接 (ResMLPs)
            final_features = self.pooling['fusion'](pooled_features) + pooled_features  # [B, D]

            # 如果启用可视化功能，则记录最终特征
            if self.enable_visualization and self.visualizer is not None:
                self.visualizer.register_features('final_features', final_features)

            return final_features  # [B, D=768]
