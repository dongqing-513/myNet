import torch.nn as nn
from transformers import BertModel
import torch

class BERTTextLSTMNet(nn.Module):
    def __init__(self, config=None, features_only=False):
        super(BERTTextLSTMNet, self).__init__()
        self.features_only = features_only

        # BERT模型
        self.bert = BertModel.from_pretrained(config.get("bert_model", "/home/mz/demo/MyNet/bert"))

        # 保存max_text_len用于维度检查
        self.max_text_len = config.get("max_text_len", 221)  # 更新默认值以适应新的序列长度

        # 两层LSTM，使用中等隐藏维度以平衡性能和计算量
        hidden_size = 384  # 提升到BERT维度的一半，使双向LSTM输出维度与BERT匹配
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # 投影层，将LSTM输出映射回原始维度
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, 768),
            nn.Dropout(0.1)
        )

        # 层归一化
        self.norm = nn.LayerNorm(768)

    def forward(self, x, attention_mask):
        """
        Args:
            x: 输入的token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
                        1 表示真实token，0 表示padding token
                        用于BERT的self-attention，对LSTM可选
        """
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_text_len, f"Input sequence length {seq_len} exceeds max_text_len {self.max_text_len}"

        # BERT编码，使用attention_mask来忽略padding tokens
        bert_output = self.bert(x, attention_mask=attention_mask).last_hidden_state

        # 保存残差连接
        residual = bert_output

        # LSTM处理序列，不需要显式使用attention_mask
        # 因为：1. 序列是定长的(197)
        #      2. LSTM按序列顺序处理，对padding相对不敏感
        lstm_out, _ = self.lstm(bert_output)

        # 投影回原始维度 [batch_size, seq_len, 768]
        proj_out = self.proj(lstm_out)

        # 残差连接和归一化
        output = self.norm(proj_out + residual)

        return output
