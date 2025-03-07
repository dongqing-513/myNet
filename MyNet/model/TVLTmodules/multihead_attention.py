import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# device = torch.device('cuda:3')

# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False, use_optimized=False, config=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        self.use_optimized = use_optimized

        # 处理配置参数
        if config is None:
            config = {
                'num_groups': 8,
                'reduction_ratio': 8,
                'hidden_size': embed_dim,
                'drop_rate': attn_dropout
            }
        if isinstance(config, dict):
            from types import SimpleNamespace
            self.config = SimpleNamespace(**config)
        else:
            self.config = config

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        if not use_optimized:
            # 原始实现
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
            self.register_parameter('in_proj_bias', None)
            if bias:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            # 优化实现：使用分组线性变换减少参数量
            num_groups = getattr(self.config, 'num_groups', 2)  # 使用getattr提供默认值
            self.num_groups = num_groups
            self.qkv_proj = nn.Conv1d(embed_dim, embed_dim * 3, 1, groups=self.num_groups, bias=bias)
            self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1, groups=self.num_groups, bias=bias)

            # 可学习的相对位置编码
            self.max_seq_len = 197  # 固定序列长度
            self.rel_pos_embed = Parameter(torch.zeros(2 * self.max_seq_len - 1, self.head_dim))
            nn.init.trunc_normal_(self.rel_pos_embed, std=0.02)

            # 特征重校准
            reduction_ratio = getattr(self.config, 'reduction_ratio', 8)  # 使用getattr提供默认值
            self.feature_calibration = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // reduction_ratio),
                nn.GELU(),
                nn.Linear(embed_dim // reduction_ratio, embed_dim),
                nn.Sigmoid()
            )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def reset_parameters(self):
        if not self.use_optimized:
            # 原始参数初始化
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.in_proj_bias is not None:
                nn.init.constant_(self.in_proj_bias, 0.)
                nn.init.constant_(self.out_proj.bias, 0.)
        else:
            # 优化版本参数初始化
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.qkv_proj.bias is not None:
                nn.init.constant_(self.qkv_proj.bias, 0.)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _get_rel_pos_bias(self, seq_len):
        """增强的相对位置编码，实现解耦的注意力机制
        Args:
            seq_len: 序列长度
        Returns:
            相对位置偏置
        """
        if not self.use_optimized:
            return 0

        # 获取相对位置编码的相关部分
        start_pos = self.max_seq_len - seq_len
        end_pos = self.max_seq_len + seq_len - 1
        rel_pos_embed = self.rel_pos_embed[start_pos:end_pos]  # [2*seq_len-1, head_dim]

        # 为每个头复制相对位置编码
        rel_pos_embed = rel_pos_embed.unsqueeze(0)  # [1, 2*seq_len-1, head_dim]

        return rel_pos_embed

    def compute_position_attention(self, q, k, rel_pos_embed, seq_len):
        """计算解耦的位置注意力分数
        Args:
            q: 查询向量 [bsz * num_heads, seq_len, head_dim]
            k: 键向量 [bsz * num_heads, seq_len, head_dim]
            rel_pos_embed: 相对位置编码 [1, 2*seq_len-1, head_dim]
            seq_len: 序列长度
        Returns:
            位置注意力分数
        """
        bsz_num_heads = q.size(0)  # bsz * num_heads

        # 扩展相对位置编码以匹配batch size和head数量
        rel_pos_embed = rel_pos_embed.expand(bsz_num_heads, -1, -1)  # [bsz * num_heads, 2*seq_len-1, head_dim]

        # 内容-位置注意力
        # q: [bsz * num_heads, seq_len, head_dim]
        # rel_pos_embed.transpose(-2, -1): [bsz * num_heads, head_dim, 2*seq_len-1]
        content_pos = torch.matmul(q, rel_pos_embed.transpose(-2, -1))  # [bsz * num_heads, seq_len, 2*seq_len-1]

        # 位置-内容注意力
        # k: [bsz * num_heads, seq_len, head_dim]
        # 不需要转置k，因为我们需要它的原始形状
        pos_content = torch.matmul(rel_pos_embed, k.transpose(-2, -1))  # [bsz * num_heads, 2*seq_len-1, seq_len]

        # 只取需要的部分并组合
        content_pos = content_pos[:, :, seq_len-1:2*seq_len-1]  # [bsz * num_heads, seq_len, seq_len]
        pos_content = pos_content[:, seq_len-1:2*seq_len-1, :]  # [bsz * num_heads, seq_len, seq_len]

        return content_pos + pos_content

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        if not self.use_optimized:
            # 原始实现的forward逻辑
            qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
            kv_same = key.data_ptr() == value.data_ptr()

            tgt_len, bsz, embed_dim = query.size()
            assert embed_dim == self.embed_dim
            assert list(query.size()) == [tgt_len, bsz, embed_dim]
            assert key.size() == value.size()

            if qkv_same:
                # self-attention
                q, k, v = self.in_proj_qkv(query)
            elif kv_same:
                # encoder-decoder attention
                q = self.in_proj_q(query)
                if key is None:
                    assert value is None
                    k = v = None
                else:
                    k, v = self.in_proj_kv(key)
            else:
                q = self.in_proj_q(query)
                k = self.in_proj_k(key)
                v = self.in_proj_v(value)

            q = q * self.scaling

            if self.bias_k is not None:
                assert self.bias_v is not None
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

            q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            if k is not None:
                k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            if v is not None:
                v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            src_len = k.size(1)

            if self.add_zero_attn:
                src_len += 1
                k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
                v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
                if attn_mask is not None:
                    attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights += attn_mask.unsqueeze(0)

            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

            attn = torch.bmm(attn_weights, v)
            assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)

            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
            return attn, attn_weights

        else:
            # 优化实现的forward逻辑
            seq_len, bsz, embed_dim = query.size()
            scaling = float(self.head_dim) ** -0.5

            # 1. QKV投影
            x = query.permute(1, 2, 0)
            qkv = self.qkv_proj(x)  # [bsz, 3 * embed_dim, seq_len]
            qkv = self.channel_shuffle(qkv, self.num_groups)
            qkv = qkv.permute(2, 0, 1)
            q, k, v = qkv.chunk(3, dim=-1)  # [seq_len, bsz, embed_dim] (每个)

            # 2. 重塑为多头形式
            q = q.view(seq_len, bsz, self.num_heads, self.head_dim)
            k = k.view(seq_len, bsz, self.num_heads, self.head_dim)
            v = v.view(seq_len, bsz, self.num_heads, self.head_dim)

            # 3. 调整维度顺序用于注意力计算
            q = q.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, seq_len, self.head_dim)
            k = k.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, seq_len, self.head_dim)
            v = v.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, seq_len, self.head_dim)

            # 4. 注意力计算
            k = k.transpose(1, 2)
            q = q * scaling
            attn_weights = torch.bmm(q, k)  # [bsz * num_heads, seq_len, seq_len]

            # 5. 添加相对位置偏置
            rel_pos_bias = self._get_rel_pos_bias(seq_len)
            if isinstance(rel_pos_bias, torch.Tensor):
                k_for_pos = k.transpose(-2, -1)
                pos_attn = self.compute_position_attention(q, k_for_pos, rel_pos_bias, seq_len)
                attn_weights = attn_weights + pos_attn

            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1)
                attn_weights = attn_weights + attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

            # 6. 计算注意力输出
            attn = torch.bmm(attn_weights, v)  # [bsz * num_heads, seq_len, head_dim]

            # 7. 重塑输出
            attn = attn.view(bsz, self.num_heads, seq_len, self.head_dim)
            attn = attn.permute(2, 0, 1, 3).contiguous()
            attn = attn.view(seq_len, bsz, embed_dim)

            # 8. 特征重校准
            attn_reshaped = attn.permute(1, 2, 0)
            attn_reshaped = attn_reshaped.transpose(1, 2)
            scale = self.feature_calibration(attn_reshaped)
            scale = scale.transpose(1, 2)
            attn = (attn_reshaped.transpose(1, 2) * scale).permute(2, 0, 1)

            # 9. 输出投影
            attn = attn.permute(1, 2, 0)
            attn = self.out_proj(attn)
            attn = self.channel_shuffle(attn, self.num_groups)
            attn = attn.permute(2, 0, 1)

            # 10. 计算平均注意力权重
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len).mean(dim=1)

            return attn, attn_weights

    def channel_shuffle(self, x, groups):
        """实现通道重排，增强不同组之间的信息交流
        Args:
            x: 输入张量，形状为 [bsz, channels, seq_len]
            groups: 分组数
        Returns:
            重排后的张量，形状保持不变
        """
        bsz, channels, seq_len = x.size()
        channels_per_group = channels // groups

        # 重塑为 [bsz, groups, channels_per_group, seq_len]
        x = x.view(bsz, groups, channels_per_group, seq_len)
        # 交换 groups 和 channels_per_group 维度
        x = x.transpose(1, 2).contiguous()
        # 重塑回原始形状
        x = x.view(bsz, channels, seq_len)
        return x

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
