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


class MSAFLSTMNet(nn.Module):
    def __init__(self, _config):
        super(MSAFLSTMNet, self).__init__()

        self.max_feature_layers = 1  # number of layers in unimodal models before classifier
        
        # NFHNET语音和视频模型层 低阶视频音频模态聚合模块
        # self.audio_visual_model = BottleAttentionNet()

        self.embed_dim = _config["hidden_size"]  # 使用_config获取参数
        # 交叉注意力
        
        self.cross_transformer = MultiModalFusionEncoder(
            hidden_size=self.embed_dim,
            num_heads=_config['num_heads'],
            num_layers=_config['num_layers'],
            stride_layer=_config['skip_interval'],
            fusion_type=_config['fusion_type'],
            dropout=_config['drop_rate'],
            relu_dropout=_config['drop_rate'],
            res_dropout=_config['drop_rate'],
            attn_dropout=_config['drop_rate'],
            embed_dropout=_config['drop_rate'],
            attn_mask=False,
            normalize_before=_config['normalize_before']
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
            hidden_size=self.embed_dim,
            num_heads=_config['num_heads'],
            num_layers=_config['num_layers'],
            stride_layer=_config['skip_interval'],
            fusion_type=_config['fusion_type'],
            dropout=_config['drop_rate'],
            relu_dropout=_config['drop_rate'],
            res_dropout=_config['drop_rate'],
            attn_dropout=_config['drop_rate'],
            embed_dropout=_config['drop_rate'],
            attn_mask=False,
            normalize_before=_config['normalize_before']
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
            # 获取文本模型
            text_model = _config["bert"]["model"]
            # 进行文本特征提取：Bert+两层lstm 
            self.text_model = BERTTextLSTMNet()
            # 填充文本id
            self.text_id = _config["bert"]["id"]

        # 定义visual_id和audio_id
        if "visual" in _config:
            self.visual_id = _config["visual"]["id"]
        if "audio" in _config:
            self.audio_id = _config["audio"]["id"]

        # 归一化
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # 特征维度 [1, 220, 768]
        
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
        # txt已经通过BertTokenizer，,得到torch.Size([1, 512])
        # av已经通过一次transformer

        for i in range(self.max_feature_layers):
            # 处理文本数据，依次通过bert和两个lstm
            if hasattr(self, "text_id"):
                txt = self.text_model(txt,attention_mask)
                # print(f"\ntxt LSTM output sequence length: {txt.shape[1]}")
                # torch.Size([1, 512, 768])print(f"\nLSTM output dimension: {txt.shape}")
                
            # NFHNET语音和视频模型层
            # old：audio_visual_feature = self.audio_visual_model(x[self.audio_id], x[self.visual_id])
            audio_visual_feature = av
            # print("\nav.shape:",av.shape)
            # TODO: 交叉注意力需要多模态之间bath_size num_of_tokens dim维度一致，现在图片音频的融合维度远大于文本序列长度
            #       如何缩短av序列长度或增大文本序列长度            
            # av : ([1, 1801, 768]) bath_size num_of_tokens dim
            # 经过平均池化调整维度 
            # Selecting torch.Size([1, 512, 768])
            # audio_visual_feature = audio_visual_feature[:, :512, :]
            # av；Layer normalization
            # 处理音视频特征维度
            # 使用平均池化减少序列长度，从1801降到225（1801/8≈225）
            pool = nn.AvgPool1d(kernel_size=8, stride=8)
            # 转置以适应池化层的输入要求 [batch, seq_len, dim] -> [batch, dim, seq_len]
            audio_visual_feature = audio_visual_feature.transpose(1, 2)
            # print("\naudio_visual_feature:",audio_visual_feature.shape)
            # 应用池化
            audio_visual_feature = pool(audio_visual_feature)
            # 转置回原来的格式 [batch, dim, seq_len] -> [batch, seq_len, dim]
            audio_visual_feature = audio_visual_feature.transpose(1, 2)
            # print("\naudio_visual_feature:",audio_visual_feature.shape)
            
            # 对音视频特征进行归一化
            audio_visual_feature = self.layer_norm(audio_visual_feature)
            
            # 对文本归一化
            txt = self.layer_norm(txt)
            
            # txt - self-attention torch.Size([1, 512, 768])
            result1 = self.classifcation(txt)
            # print("\nresult1",result1.shape) 
            
            # cross_transformer1: T->(V,A)
            l_av = self.cross_transformer(audio_visual_feature, result1, result1)
            #print("==========cross_transformer1: T->(V,A)=========done",l_av.shape) # torch.Size([1, 512, 768])
            
            # cross_transformer2: (V,A)->T
            av_l = self.cross_transformer(result1, audio_visual_feature, audio_visual_feature)
            # print("==========cross_transformer2: (V,A)->T=========done",av_l.shape) # torch.Size([1, 512, 768])

            # cross_transformer2 -> self-attention 
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
            #print("\nattn_pooled",attn_pooled.shape)
            
            # 2. 全局平均池化
            avg_pooled = torch.mean(all, dim=1)  # [1, 768]
            #print("\navg_pooled",avg_pooled.shape)
            
            # 组合两种池化结果
            #pooled_result = torch.cat((attn_pooled, avg_pooled), dim=1)  # [1, 1536]
            """
            # 6分类任务 - 使用预训练权重的分类器
            x = self.classifier[0].dense(pooled_result)
            x = F.gelu(x)
            x = F.layer_norm(x, [self.embed_dim])
            
            x = self.classifier[1](x)
            x = F.gelu(x)
            
            x = self.classifier[2](x)
            x = F.gelu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
            emotion_output = self.classifier[4](x)  # [1, 6]
            emotion_probs = F.softmax(emotion_output, dim=-1)"""
            
            # 2分类任务 - 使用独立的分类器
            #binary_output = self.binary_classifier(pooled_result)  # [1, 1]
            binary_output = torch.tanh(attn_pooled)  # sigmoid更符合二分类任务,tanh限制在[-1,1]范围
            
            return binary_output
