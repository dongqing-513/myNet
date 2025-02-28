import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from NHFNet.networks.text_lstm import BERTTextLSTMNet
import sys
sys.path.append('..')
# from MSAF import MSAF

from NHFNet.modules.transformer import TransformerEncoder
from model.TVLTmodules.fusion_module import MultiModalFusionEncoder
from NHFNet.networks.sequence_align import SequenceAlignmentModule

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
            use_lightweight=_config.get('use_lightweight_align', False)  # 从配置中加载
        )

        # 多种池化策略
        self.global_pool = nn.AdaptiveAvgPool2d((1, 768))  # 全局平均池化
        self.attention_pool = nn.Sequential(
            nn.Linear(768, 128),  # 降维以减少参数
            #nn.LayerNorm(768 * 2),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 6分类任务 - 使用预训练权重
        self.classifier = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('dense', nn.Linear(self.embed_dim, self.embed_dim))
            ])),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.Identity(),
            nn.Linear(self.embed_dim * 2, 6)
        ])

        # 2分类任务 - 独立的分类器
        self.binary_classifier = nn.Sequential(
            # 特征转换层
            nn.Linear(1536, 1024),  # 输入是拼接的两种池化结果
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),

            # 中间层
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            # 输出层
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, av, txt, attention_mask):
        for i in range(self.max_feature_layers):
            if hasattr(self, "text_id"):
                txt = self.text_model(txt, attention_mask)

            # 序列长度对齐，输入已经是 [batch, seq_len, dim]
            audio_visual_feature = self.sequence_align(av)

            # 最终特征归一化
            audio_visual_feature = self.layer_norm(audio_visual_feature)
            txt = self.layer_norm(txt)

            # txt - self-attention torch.Size([1, 197, 768])
            result1 = self.classifcation(txt)
            # print("\nresult1",result1.shape)

            # 交叉注意力融合
            # T->(V,A)：文本引导的视听注意力
            l_av = self.cross_transformer(audio_visual_feature, result1, result1)
            # (V,A)->T：视听引导的文本注意力
            av_l = self.cross_transformer(result1, audio_visual_feature, audio_visual_feature)
            # print("==========cross_transformer2: (V,A)->T=========done",av_l.shape) # torch.Size([1, 512, 768])

            # 融合特征的自注意力处理
            l_result = self.classifcation(av_l)
            # print("==========cross_transformer2 -> self-attention=========done",l_result.shape) # torch.Size([1, 512, 768])

            # cross_transformer1 -> self-attention
            av_result = self.classifcation(l_av)
            # print("==========cross_transformer1 -> self-attention========done",av_result.shape) # torch.Size([1, 512, 768])

            # TODO: 现在直接用训练好的三维序列进行拼接result1 + result2 + l_result + av_result，直接输入liner进行分类
            # 在计算标号和损失之间的差异时，遇到了维度问题。需要查一下最新的如何将transformer结果进行预测和反向传播
            # 每个结果的最后一个时间步的输出
            # result1 = result1[-1]
            #print("result1",result1.shape)
            # torch.Size([512, 768]) 不取[-1]：torch.Size([1, 512, 768])

            all = result1 + audio_visual_feature + l_result + av_result
            #print("\ntest:",all.shape) # torch.Size([1, 512, 768])

            # 1. 注意力池化 - 学习每个时间步的重要性
            # [1, 220, 768] -> [1, 220, 1]
            attn_weights = self.attention_pool(all)
            # [1, 220, 1] x [1, 220, 768] -> [1, 768]
            attn_pooled = torch.bmm(attn_weights.transpose(1, 2), all).squeeze(1)

            # 全局平均池化
            avg_pooled = torch.mean(all, dim=1)  # [1, 768]

            # 二分类输出
            binary_output = torch.tanh(attn_pooled)

            return binary_output
