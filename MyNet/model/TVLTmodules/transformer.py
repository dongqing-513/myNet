import torch
from torch import nn
import torch.nn.functional as F
from model.TVLTmodules.position_embedding import SinusoidalPositionalEmbedding
from model.TVLTmodules.multihead_attention import MultiheadAttention
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        layers: Number of layers
        attn_dropout: Dropout applied on the attention weights
        relu_dropout: Dropout applied on the first layer of the residual block
        res_dropout: Dropout applied on the residual block
        attn_mask: Boolean indicating whether to apply mask or not
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        
        # 创建基本配置
        config = {
            'hidden_size': embed_dim,
            'num_heads': num_heads,
            'drop_rate': attn_dropout,
            'relu_dropout': relu_dropout,
            'res_dropout': res_dropout,
            'normalize_before': True,
            'attn_mask': attn_mask,
            'num_groups': 2,  # 默认值
            'reduction_ratio': 8  # 默认值
        }

        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(TransformerEncoderLayer(config))

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        self.layer_norm = LayerNorm(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        
        self.attn_mask = attn_mask

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                # 1
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def training_step(self, batch, batch_idx):
        ret = self(batch)
        
        # 记录损失组件和批次统计信息
        if hasattr(self, 'metrics_logger'):
            if 'loss_components' in ret:
                self.metrics_logger.log_loss_components(
                    ret['loss_components'],
                    self.global_step,
                    prefix='train'
                )
            if 'batch_stats' in ret:
                self.metrics_logger.log_batch_stats(
                    ret['batch_stats'],
                    self.global_step,
                    prefix='train'
                )
        
        return ret
        
    def validation_step(self, batch, batch_idx):
        ret = self(batch)
        
        # 记录验证集的损失组件和批次统计信息
        if hasattr(self, 'metrics_logger'):
            if 'loss_components' in ret:
                self.metrics_logger.log_loss_components(
                    ret['loss_components'],
                    self.global_step,
                    prefix='val'
                )
            if 'batch_stats' in ret:
                self.metrics_logger.log_batch_stats(
                    ret['batch_stats'],
                    self.global_step,
                    prefix='val'
                )
        
        return ret


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            from types import SimpleNamespace
            self.config = SimpleNamespace(**config)
        else:
            self.config = config

        self.normalize_before = self.config.normalize_before

        self.self_attn = MultiheadAttention(
            self.config.hidden_size,
            self.config.num_heads,
            self.config.drop_rate,
            use_optimized=True,
            config=self.config
        )
        self.attn_mask = self.config.attn_mask

        self.relu_dropout = self.config.relu_dropout
        self.res_dropout = self.config.res_dropout

         # Memory and Compound control
        self.mem_proj = nn.Sequential(
            nn.Linear(2*self.config.hidden_size, self.config.hidden_size),
            nn.Sigmoid()
        )
        self.att_proj = nn.Sequential(
            nn.Linear(2*self.config.hidden_size, self.config.hidden_size),
            nn.Sigmoid()           
        )

        self.fc1 = Linear(self.config.hidden_size, 4*self.config.hidden_size)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.config.hidden_size, self.config.hidden_size)
        self.layer_norms = nn.ModuleList([LayerNorm(self.config.hidden_size) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        # 残差连接
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            # 使用 x 同时作为查询（query）、键（key）和值（value）进行自注意力计算，并传入掩码 mask
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            # 分别对 x_k 和 x_v 进行可能的层归一化处理。
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            # 进行交叉自注意力计算
            # 2
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        # 加入残差连接，将输入和自注意力的输出相加
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    class Config:
        def __init__(self):
            self.hidden_size = 300
            self.num_heads = 4
            self.drop_rate = 0.1
            self.relu_dropout = 0.1
            self.res_dropout = 0.1
            self.normalize_before = True
            self.attn_mask = False

    config = Config()
    encoder = TransformerEncoder(config.hidden_size, config.num_heads, 2)
    x = torch.tensor(torch.rand(20, 2, config.hidden_size))
    print(encoder(x).shape)
