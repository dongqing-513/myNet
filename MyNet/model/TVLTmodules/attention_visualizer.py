import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch.nn.functional as F

class AttentionVisualizer:
    """
    用于注册和可视化注意力权重和特征的工具类
    """
    def __init__(self, save_dir='visualizations', max_frames=10):
        """
        初始化可视化工具
        
        Args:
            save_dir: 保存可视化结果的目录
            max_frames: 每个批次中要可视化的最大帧数
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_frames = max_frames
        
        # 存储特征和注意力权重的字典
        self.features = {}
        self.attention_weights = {}
        self.step_count = 0
        
        # 注册的句柄集合，用于确定哪些需要可视化
        self.registered_features = set()
        self.registered_attentions = set()

    def register_features(self, name, features):
        """
        注册特征进行可视化
        
        Args:
            name: 特征的名称
            features: 特征张量 [batch, seq, dim] 或 [seq, batch, dim]
        """
        if features is None:
            return
            
        # 确保特征是分离的并转到CPU
        if torch.is_tensor(features):
            features = features.detach().cpu()
            
        # 记录特征
        self.features[name] = features
        self.registered_features.add(name)

    def register_attention_weights(self, name, weights):
        """
        注册注意力权重进行可视化
        
        Args:
            name: 注意力权重的名称
            weights: 注意力权重张量 [batch, heads, seq_q, seq_k] 或其他形式
        """
        if weights is None:
            return
            
        # 确保权重是分离的并转到CPU
        if torch.is_tensor(weights):
            weights = weights.detach().cpu()
            
        # 记录权重
        self.attention_weights[name] = weights
        self.registered_attentions.add(name)
    
    def visualize_feature_maps(self, step=None):
        """
        可视化所有注册的特征图
        
        Args:
            step: 当前训练步骤，用于文件名
        """
        if not self.features:
            return
            
        step_str = f"step_{step}" if step is not None else f"step_{self.step_count}"
        feature_dir = self.save_dir / "features" / step_str
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化每个特征
        for name, feature in self.features.items():
            try:
                # 确定特征的形状
                if feature.dim() == 3:
                    # 如果是 [batch, seq, dim] 格式，转换为 [batch, dim, seq] 以便于可视化
                    if feature.shape[0] < feature.shape[1]:
                        # 可能是 [seq, batch, dim] 格式
                        feature = feature.permute(1, 2, 0)
                    else:
                        # 应该是 [batch, seq, dim] 格式
                        feature = feature.permute(0, 2, 1)
                
                # 限制批次大小，避免生成太多图像
                batch_size = min(feature.shape[0], self.max_frames)
                
                for b in range(batch_size):
                    # 获取当前批次的特征
                    feat = feature[b]
                    
                    # 计算特征的平均值和标准差，用于归一化
                    feat_mean = feat.mean()
                    feat_std = feat.std()
                    normalized_feat = (feat - feat_mean) / (feat_std + 1e-8)
                    
                    # 创建热图
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(normalized_feat.numpy(), cmap='viridis')
                    plt.title(f"{name} - Batch {b}")
                    plt.tight_layout()
                    
                    # 保存图像
                    plt.savefig(feature_dir / f"{name}_batch_{b}.png")
                    plt.close()
            except Exception as e:
                print(f"可视化特征 {name} 时出错: {str(e)}")
    
    def visualize_attention(self, step=None):
        """
        可视化所有注册的注意力权重
        
        Args:
            step: 当前训练步骤，用于文件名
        """
        if not self.attention_weights:
            return
            
        step_str = f"step_{step}" if step is not None else f"step_{self.step_count}"
        attn_dir = self.save_dir / "attention" / step_str
        attn_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化每个注意力权重
        for name, attn in self.attention_weights.items():
            try:
                # 检查形状
                if attn.dim() < 3:
                    continue
                    
                # 根据维度调整
                if attn.dim() == 4:  # [batch, heads, seq_q, seq_k]
                    batch_size = min(attn.shape[0], self.max_frames)
                    num_heads = min(attn.shape[1], 4)  # 限制要可视化的头数
                    
                    for b in range(batch_size):
                        for h in range(num_heads):
                            attn_map = attn[b, h]
                            
                            # 创建热图
                            plt.figure(figsize=(10, 8))
                            sns.heatmap(attn_map.numpy(), cmap='viridis')
                            plt.title(f"{name} - Batch {b}, Head {h}")
                            plt.tight_layout()
                            
                            # 保存图像
                            plt.savefig(attn_dir / f"{name}_batch_{b}_head_{h}.png")
                            plt.close()
                elif attn.dim() == 3:  # [batch, seq_q, seq_k] 或 [heads, seq_q, seq_k]
                    # 假设是 [batch, seq_q, seq_k]
                    batch_size = min(attn.shape[0], self.max_frames)
                    
                    for b in range(batch_size):
                        attn_map = attn[b]
                        
                        # 创建热图
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(attn_map.numpy(), cmap='viridis')
                        plt.title(f"{name} - Batch {b}")
                        plt.tight_layout()
                        
                        # 保存图像
                        plt.savefig(attn_dir / f"{name}_batch_{b}.png")
                        plt.close()
            except Exception as e:
                print(f"可视化注意力 {name} 时出错: {str(e)}")
    
    def save_tensors(self, step=None):
        """
        保存所有注册的特征和注意力权重为numpy数组
        
        Args:
            step: 当前训练步骤，用于文件名
        """
        if not (self.features or self.attention_weights):
            return
            
        step_str = f"step_{step}" if step is not None else f"step_{self.step_count}"
        tensor_dir = self.save_dir / "tensors" / step_str
        tensor_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存特征
        for name, feature in self.features.items():
            try:
                if torch.is_tensor(feature):
                    np.save(tensor_dir / f"{name}.npy", feature.numpy())
            except Exception as e:
                print(f"保存特征 {name} 时出错: {str(e)}")
        
        # 保存注意力权重
        for name, attn in self.attention_weights.items():
            try:
                if torch.is_tensor(attn):
                    np.save(tensor_dir / f"{name}.npy", attn.numpy())
            except Exception as e:
                print(f"保存注意力权重 {name} 时出错: {str(e)}")
    
    def clear(self):
        """
        清除当前存储的特征和注意力权重，但保留注册信息
        """
        self.features = {}
        self.attention_weights = {}
    
    def visualize_all(self, step=None, save_tensors=True):
        """
        可视化所有注册的特征和注意力权重，并选择性地保存原始张量
        
        Args:
            step: 当前训练步骤
            save_tensors: 是否保存原始张量
        """
        if step is not None:
            self.step_count = step
        
        # 可视化特征
        self.visualize_feature_maps(self.step_count)
        
        # 可视化注意力权重
        self.visualize_attention(self.step_count)
        
        # 保存原始张量
        if save_tensors:
            self.save_tensors(self.step_count)
        
        # 更新步数并清理
        self.step_count += 1
        self.clear()
