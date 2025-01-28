import torch.nn as nn
from transformers import BertModel
'''
Input output shapes
audio (BERT): (50, 768)
label: (1) -> [sentiment]
'''


class BERTTextLSTMNet(nn.Module):
    def __init__(self, features_only=False):
        # 调用父类（nn.Module）的构造函数
        super(BERTTextLSTMNet, self).__init__()
        self.features_only = features_only
        #'bert-base-uncased'
        self.bert = BertModel.from_pretrained("/home/mz/demo/MyNet/bert")
        # 传入维度：输入大小为 768，表示输入序列中每个时间步的特征维度为 768（对应Bert的输出）
        # 传出维度：输出的隐藏状态维度为 128，即每个时间步的隐藏状态大小为 128。
        # 输出的形状为(batch_size, sequence_length, 128)，
        # 其中batch_size是批次大小，sequence_length是输入序列的长度。
        # 传入维度：接收上一个 LSTM 层的输出，即输入大小为 128。
        # 传出维度：输出的隐藏状态维度为 32，输出形状为(batch_size, sequence_length, 32)
        # self.lstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=768, num_layers=1, batch_first=True)    
        
        # 定义分类器部分，由一个 dropout 层和一个线性层组成
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            # 线性层的输入维度为 128，输出维度为 1，表示进行二分类或回归任务
            nn.Linear(128, 1),
        )

    def forward(self, x , attention_mask):
        # print("\nBERTTextLSTMNet",x.shape)#([1, 512])
        bert_output = self.bert(x,attention_mask=attention_mask)[0]
        # print("bertout",bert_output.shape)#([1, 512, 768])
        x, _ = self.lstm1(bert_output)
        # print("\nlstm1",x.shape)#([1, 512, 128])
        x, _ = self.lstm2(x)
        # print("\nlstm2",x.shape)#([1, 512, 768])
        # last = x[:, -1, :]([1, 768])
        
        return x    #self.classifier(last)
