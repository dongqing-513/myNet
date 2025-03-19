import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

class AttentionVisualizer:
    """
    通用的注意力可视化工具类，可用于不同模块的注意力和特征可视化
    """
    def __init__(self, save_dir: str = "visualizations"):
        """
        初始化可视化工具类
        
        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.attention_weights = {}
        self.features = {}
        
    def register_attention(self, name: str, attention_weights: torch.Tensor):
        """
        注册注意力权重用于可视化
        
        Args:
            name: 注意力层的标识名称
            attention_weights: 注意力权重张量
        """
        if attention_weights is not None:
            self.attention_weights[name] = attention_weights.detach().cpu()
    
    def register_features(self, name: str, features: torch.Tensor):
        """
        注册特征向量用于可视化
        
        Args:
            name: 特征的标识名称
            features: 特征张量
        """
        if features is not None:
            self.features[name] = features.detach().cpu()
    
    def plot_attention_map(self, name: Optional[str] = None, 
                           title: Optional[str] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'viridis',
                           show_axis_labels: bool = False,
                           axis_labels: Optional[Dict[str, List[str]]] = None):
        """
        绘制注意力权重热力图
        
        Args:
            name: 要可视化的注意力层名称，如果为None则可视化所有注册的注意力
            title: 图表标题
            save_path: 保存路径，如果为None则自动生成
            figsize: 图表大小
            cmap: 颜色映射
            show_axis_labels: 是否显示坐标轴标签
            axis_labels: 坐标轴标签字典，格式为{'x': [...], 'y': [...]}
        """
        if not self.attention_weights:
            print("没有注册的注意力权重用于可视化")
            return
        
        if name is not None:
            if name not in self.attention_weights:
                print(f"未找到名为'{name}'的注意力权重")
                return
            attention_to_plot = {name: self.attention_weights[name]}
        else:
            attention_to_plot = self.attention_weights
        
        for attn_name, attn_weights in attention_to_plot.items():
            plt.figure(figsize=figsize)
            
            # 处理注意力权重维度
            attn_weights_np = attn_weights.numpy()
            
            # 如果有多个注意力头，或多个批次，取平均
            if len(attn_weights_np.shape) > 2:
                # 常见形状: [batch_size, num_heads, seq_len_q, seq_len_k]
                # 或者: [num_heads, batch_size, seq_len_q, seq_len_k]
                if len(attn_weights_np.shape) == 4:
                    if attn_weights_np.shape[0] > attn_weights_np.shape[1]:
                        # [batch_size, num_heads, seq_len_q, seq_len_k]
                        attn_weights_np = attn_weights_np.mean(axis=(0, 1))
                    else:
                        # [num_heads, batch_size, seq_len_q, seq_len_k]
                        attn_weights_np = attn_weights_np.mean(axis=(0, 1))
                else:
                    # 其他情况，假设前两个维度是batch和head
                    attn_weights_np = attn_weights_np.mean(axis=tuple(range(len(attn_weights_np.shape)-2)))
            
            # 绘制热力图
            sns.heatmap(attn_weights_np, cmap=cmap, annot=False, square=True)
            
            # 设置坐标轴标签
            if show_axis_labels and axis_labels is not None:
                if 'x' in axis_labels and len(axis_labels['x']) == attn_weights_np.shape[1]:
                    plt.xticks(np.arange(len(axis_labels['x'])) + 0.5, axis_labels['x'], rotation=45)
                if 'y' in axis_labels and len(axis_labels['y']) == attn_weights_np.shape[0]:
                    plt.yticks(np.arange(len(axis_labels['y'])) + 0.5, axis_labels['y'], rotation=0)
            
            # 设置标题和标签
            plt_title = title if title else f"{attn_name} Attention Map"
            plt.title(plt_title)
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            
            # 设置保存路径
            if save_path:
                plt_save_path = save_path
            else:
                plt_save_path = os.path.join(self.save_dir, f"{attn_name}_attention_map.png")
                
            plt.tight_layout()
            plt.savefig(plt_save_path)
            plt.close()
            print(f"Attention map saved to {plt_save_path}")
    
    def visualize_all_attention_maps(self, prefix: str = "", suffix: str = ""):
        """
        可视化所有注册的注意力层
        
        Args:
            prefix: 文件名前缀
            suffix: 文件名后缀
        """
        for name in self.attention_weights.keys():
            save_path = os.path.join(self.save_dir, f"{prefix}{name}{suffix}_attention_map.png")
            self.plot_attention_map(name=name, title=f"{name} Attention Map", save_path=save_path)
    
    def plot_feature_distribution(self, 
                                method: str = 'tsne', 
                                names: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (10, 8),
                                sample_ratio: float = 1.0,
                                random_state: int = 42,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None):
        """
        使用降维方法可视化特征分布
        
        Args:
            method: 使用的降维方法，'tsne'或'pca'
            names: 要可视化的特征名称列表，如果为None则使用所有注册的特征
            figsize: 图表大小
            sample_ratio: 对每个特征的采样比例，以减少计算量
            random_state: 随机种子
            title: 图表标题
            save_path: 保存路径，如果为None则自动生成
        """
        if not self.features:
            print("没有注册的特征用于可视化")
            return
        
        # 确定要可视化的特征
        if names is not None:
            features_to_plot = {k: self.features[k] for k in names if k in self.features}
            if not features_to_plot:
                print(f"未找到指定名称的特征: {names}")
                return
        else:
            features_to_plot = self.features
        
        # 准备数据
        all_features = []
        labels = []
        
        for name, feature in features_to_plot.items():
            # 确保特征是二维的 [samples, features]
            feature_np = feature.numpy()
            orig_shape = feature_np.shape
            
            if len(orig_shape) > 2:
                # 将高维特征展平为二维 [samples, features]
                feature_np = feature_np.reshape(-1, orig_shape[-1])
            
            # 采样以减少计算量
            if sample_ratio < 1.0:
                num_samples = max(1, int(feature_np.shape[0] * sample_ratio))
                indices = np.random.choice(feature_np.shape[0], num_samples, replace=False)
                feature_np = feature_np[indices]
            
            all_features.append(feature_np)
            labels.extend([name] * feature_np.shape[0])
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        
        # 降维
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=random_state)
            method_name = "t-SNE"
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
            method_name = "PCA"
            
        reduced_features = reducer.fit_transform(all_features)
        
        # 绘制
        plt.figure(figsize=figsize)
        
        # 设置颜色
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                        color=colors[i], label=label, alpha=0.7)
        
        plt.legend()
        plt.title(title if title else f"{method_name} Feature Distribution")
        
        # 设置保存路径
        if save_path:
            plt_save_path = save_path
        else:
            plt_save_path = os.path.join(self.save_dir, f"{method.lower()}_feature_distribution.png")
            
        plt.tight_layout()
        plt.savefig(plt_save_path)
        plt.close()
        print(f"Feature distribution visualization saved to {plt_save_path}")
    
    def compare_feature_distributions(self, method: str = 'tsne'):
        """
        比较使用t-SNE和PCA的特征分布可视化结果
        
        Args:
            method: 使用的降维方法，'both', 'tsne'或'pca'
        """
        if method.lower() == 'both':
            self.plot_feature_distribution(method='tsne')
            self.plot_feature_distribution(method='pca')
        else:
            self.plot_feature_distribution(method=method)
    
    def clear(self):
        """清除所有注册的注意力权重和特征"""
        self.attention_weights.clear()
        self.features.clear()
        print("已清除所有注册的注意力权重和特征")
        
    def visualize_all_attention_maps(self, prefix: str = "", save_path: Optional[str] = None):
        """
        一次性可视化所有注册的注意力图
        
        Args:
            prefix: 图表标题前缀
            save_path: 可选的保存路径，如果为None则使用默认路径
        """
        if not self.attention_weights:
            print("没有注册的注意力权重可供可视化")
            return
            
        for name, weights in self.attention_weights.items():
            title = f"{prefix}{name}" if prefix else name
            custom_save_path = None
            if save_path:
                filename = f"{name.replace(' ', '_').lower()}_attention.png"
                custom_save_path = os.path.join(save_path, filename)
            self.plot_attention_map(name, title=title, save_path=custom_save_path)
        
        print(f"已可视化 {len(self.attention_weights)} 个注意力图")
        
    def visualize_attention_between_modalities(self, modality1: str, modality2: str, 
                                              title: str = "Cross-Modal Attention", 
                                              save_path: Optional[str] = None):
        """
        可视化两个模态之间的注意力权重
        
        Args:
            modality1: 第一个模态的名称
            modality2: 第二个模态的名称
            title: 图表标题
            save_path: 可选的保存路径，如果为None则使用默认路径
        """
        key_name = f"{modality1}_{modality2}_attention"
        if key_name in self.attention_weights:
            self.plot_attention_map(key_name, title=title, save_path=save_path)
        else:
            # 尝试其他可能的名称组合
            found = False
            for name in self.attention_weights.keys():
                if modality1 in name.lower() and modality2 in name.lower():
                    self.plot_attention_map(name, title=title, save_path=save_path)
                    found = True
                    break
            
            if not found:
                print(f"未找到 {modality1} 和 {modality2} 之间的注意力权重")
