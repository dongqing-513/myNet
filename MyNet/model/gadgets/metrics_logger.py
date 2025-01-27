import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import torch
from torch.utils.tensorboard import SummaryWriter

class MetricsLogger:
    """用于记录和可视化训练过程中的各种指标"""
    
    def __init__(self, log_dir):
        """初始化日志记录器
        
        Args:
            log_dir: TensorBoard日志目录
        """
        self.writer = SummaryWriter(log_dir)
        self.metrics_history = defaultdict(list)
        
    def log_loss_components(self, loss_components, global_step, prefix='train'):
        """记录损失组件
        
        Args:
            loss_components: 包含各损失组件的字典
            global_step: 全局训练步数
            prefix: 指标前缀（train/val/test）
        """
        for name, value in loss_components.items():
            self.writer.add_scalar(f'{prefix}/losses/{name}', value, global_step)
            self.metrics_history[f'{prefix}/losses/{name}'].append(value)
            
    def log_batch_stats(self, batch_stats, global_step, prefix='train'):
        """记录批次统计信息
        
        Args:
            batch_stats: 包含批次统计信息的字典
            global_step: 全局训练步数
            prefix: 指标前缀
        """
        for name, value in batch_stats.items():
            self.writer.add_scalar(f'{prefix}/batch_stats/{name}', value, global_step)
            self.metrics_history[f'{prefix}/batch_stats/{name}'].append(value)
            
    def plot_loss_components(self, prefix='train'):
        """绘制损失组件趋势图
        
        Args:
            prefix: 指标前缀
        Returns:
            fig: matplotlib图像对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in self.metrics_history.keys():
            if name.startswith(f'{prefix}/losses/'):
                values = self.metrics_history[name]
                ax.plot(values, label=name.split('/')[-1])
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss Value')
        ax.set_title(f'{prefix.capitalize()} Loss Components')
        ax.legend()
        
        return fig
        
    def plot_batch_stats(self, prefix='train'):
        """绘制批次统计信息趋势图
        
        Args:
            prefix: 指标前缀
        Returns:
            fig: matplotlib图像对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for name in self.metrics_history.keys():
            if name.startswith(f'{prefix}/batch_stats/'):
                values = self.metrics_history[name]
                ax.plot(values, label=name.split('/')[-1])
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.set_title(f'{prefix.capitalize()} Batch Statistics')
        ax.legend()
        
        return fig
        
    def log_figures(self, global_step, prefix='train'):
        """将图像记录到TensorBoard
        
        Args:
            global_step: 全局训练步数
            prefix: 指标前缀
        """
        # 记录损失组件图
        loss_fig = self.plot_loss_components(prefix)
        self.writer.add_figure(f'{prefix}/loss_components', loss_fig, global_step)
        plt.close(loss_fig)
        
        # 记录批次统计图
        stats_fig = self.plot_batch_stats(prefix)
        self.writer.add_figure(f'{prefix}/batch_stats', stats_fig, global_step)
        plt.close(stats_fig)
        
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()
