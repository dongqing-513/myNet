import torch.nn as nn
from transformers import BertModel
'''
Input output shapes
audio (BERT): (50, 768)
label: (1) -> [sentiment]
'''


class BERTTextLSTMNet(nn.Module):
    def __init__(self, config=None, features_only=False):
        super(BERTTextLSTMNet, self).__init__()
        self.features_only = features_only
        
        # 直接使用bert_model配置
        self.bert = BertModel.from_pretrained(config.get("bert_model", "/home/mz/demo/MyNet/bert"))
        
        # 保存max_text_len用于维度检查
        self.max_text_len = config.get("max_text_len", 197)
        
        # 优化LSTM架构
        # 沙漏结构，压缩学习更紧凑的信息，恢复学习更丰富的特征
        hidden_sizes = [768, 384, 768]  # 渐进式的维度变化
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_sizes[i],
                hidden_size=hidden_sizes[i+1],
                num_layers=2,  # 增加层数以捕获更复杂的时序关系
                bidirectional=True,  # 双向LSTM可以捕获双向上下文
                batch_first=True,
                dropout=0.1  # 添加dropout防止过拟合
            ) for i in range(len(hidden_sizes)-1)
        ])
        
        # 添加层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(size) for size in hidden_sizes[1:]
        ])
        
        # 注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1
        )
        
        # 残差连接后的层归一化
        self.final_layer_norm = nn.LayerNorm(768)

    def forward(self, x, attention_mask):
        # 检查输入维度
        # x, attention_mask shape: [batch_size, seq_len]
        # - batch_size: 批次大小
        # - seq_len: 序列长度，最大为197（由max_text_len控制）
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_text_len, f"Input sequence length {seq_len} exceeds max_text_len {self.max_text_len}"
        
        # BERT编码
        # bert_output shape: [batch_size, seq_len, hidden_size=768]
        # - 768是BERT的隐藏层维度
        # - 每个token都被映射到768维的向量空间
        bert_output = self.bert(x, attention_mask=attention_mask)[0]
        
        # 通过LSTM层处理，包含残差连接
        hidden = bert_output
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            # 保存残差连接
            residual = hidden
            
            # LSTM处理
            lstm_out, _ = lstm(hidden)
            
            # 如果是双向LSTM，需要合并前向和后向的输出，
            # 第一层：[batch_size, seq_len, 384]
            # 第二层：[batch_size, seq_len, 768]
            if lstm.bidirectional:
                lstm_out = lstm_out.view(batch_size, seq_len, 2, -1).sum(dim=2)
            
            # 层归一化
            lstm_out = norm(lstm_out)
            
            # 残差连接（如果维度匹配）
            if lstm_out.shape[-1] == residual.shape[-1]:
                lstm_out = lstm_out + residual
            
            hidden = lstm_out
        
        # 自注意力机制
        # 转置以适应注意力层的输入要求， [seq_len, batch_size, 768]
        hidden = hidden.transpose(0, 1)
        attn_output, _ = self.self_attention(hidden, hidden, hidden, 
                                           key_padding_mask=~attention_mask.bool()) 
        # 转置回原始形状：[batch_size, seq_len, 768]
        hidden = hidden.transpose(0, 1)
        
        # 最终的残差连接和层归一化，[batch_size, seq_len, 768]
        output = self.final_layer_norm(hidden + attn_output.transpose(0, 1))
        
        return output
