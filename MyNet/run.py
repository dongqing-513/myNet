import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import json
import copy
import pytorch_lightning as pl

# 修改临时文件目录到项目内部，解决/tmp目录问题
tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = tmp_dir
os.environ['TEMP'] = tmp_dir
os.environ['TMP'] = tmp_dir

from model.config import ex
from model.TVLTmodules import Transformer
from model.data.datamodules.multitask_datamodule import MTDataModule
import torch
from sacred.observers import FileStorageObserver
from model.gadgets.metrics_logger import MetricsLogger

warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.")

import torch.multiprocessing as mp

# 设置多进程启动方式为spawn，解决进程间通信问题
mp.set_start_method('spawn', force=True)

# 添加 Sacred 观察器
ex.observers.append(FileStorageObserver('sacred_runs'))

#  @ex.automain装饰器表明这是一个主要的执行入口点，接受一个_config参数，配置字典。
@ex.automain
def main(_config):
    #  深度复制配置
    _config = copy.deepcopy(_config)
    # 设置随机种子
    pl.seed_everything(_config["seed"])

    # 数据模块传入配置参数
    # TODO: 读取NFHnet的文本数据
    dm = MTDataModule(_config, dist=True)

    # TODO: 添加NHFNet 到Transformer内部的forward
    model = Transformer(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,                # 保存最好的 k 个模型
        verbose=True,                # 是否打印日志
        #monitor="mosei/val/f1",     # 监控验证集上的 F1 分数
        monitor="val/the_metric",    # 要监控的指标名称，模型会根据这个指标在验证集上的表现来决定是否保存权重等操作
        mode="max",                  # 确定监控指标的优化方向
        save_last=True,              # 额外保存训练最后一个 epoch 对应的模型权重文件
        filename='{epoch}-{the_metric:.2f}'  # 模型权重文件的文件名格式
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_from_{_config["load_local_path"].split("/")[-1][:-5]}_{_config["model_type"]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # callbacks = [checkpoint_callback, lr_callback]
    # 创建指标记录器
    metrics_logger = MetricsLogger(_config["log_dir"])
    
    # 将metrics_logger添加到模型中
    model.metrics_logger = metrics_logger
    
    # 添加回调以定期记录图表
    class MetricsPlottingCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if hasattr(pl_module, 'metrics_logger'):
                pl_module.metrics_logger.log_figures(trainer.global_step, prefix='train')
                
        def on_validation_epoch_end(self, trainer, pl_module):
            if hasattr(pl_module, 'metrics_logger'):
                pl_module.metrics_logger.log_figures(trainer.global_step, prefix='val')
    
    class BestModelVisualizationCallback(pl.Callback):
        """在最佳模型被保存时生成可视化图表"""
        
        def __init__(self, monitor='val/the_metric', mode='max'):
            super().__init__()
            self.monitor = monitor
            self.mode = mode
            self.best_score = float('-inf') if mode == 'max' else float('inf')
            self.best_epoch = 0
            
        def on_validation_epoch_end(self, trainer, pl_module):
            # 获取当前监控指标的值
            current_score = trainer.callback_metrics.get(self.monitor)
            current_epoch = trainer.current_epoch
            
            if current_score is None:
                return
                
            # 检查是否是更好的得分
            if (self.mode == 'max' and current_score > self.best_score) or \
               (self.mode == 'min' and current_score < self.best_score):
                
                # 更新最佳分数和轮次
                old_score = self.best_score
                self.best_score = current_score
                self.best_epoch = current_epoch
                
                # 生成并保存可视化图表
                print(f"\n生成最佳模型的可视化图表 (epoch {current_epoch}): {self.monitor} 从 {old_score:.4f} 提升到 {current_score:.4f}")
                
                # 创建特定于此最佳模型的可视化目录
                vis_dir = os.path.join(pl_module.hparams.config["visualization_dir"], f"epoch_{current_epoch}")
                os.makedirs(vis_dir, exist_ok=True)
                
                # 临时更新可视化器的保存目录
                self._update_visualizer_dir(pl_module, vis_dir)
                
                # 1. 处理 MSAFLSTMNet 的可视化
                if hasattr(pl_module, 'msaf') and pl_module.msaf is not None:
                    self._generate_msaf_visualizations(pl_module.msaf)
                
                # 2. 处理 BottleAttentionNet 的可视化
                if hasattr(pl_module, 'transformer') and hasattr(pl_module.transformer, 'transformer'):
                    self._generate_bottle_attention_visualizations(pl_module.transformer.transformer)
                
                # 恢复原始保存目录
                self._restore_visualizer_dir(pl_module)
                
                print(f"可视化图表已保存到: {vis_dir}")
        
        def _update_visualizer_dir(self, pl_module, new_dir):
            """临时更新所有可视化器的保存目录"""
            self.original_dirs = {}
            
            # 保存并更新 MSAF 的可视化器
            if hasattr(pl_module, 'msaf') and hasattr(pl_module.msaf, 'visualizer') and pl_module.msaf.visualizer is not None:
                self.original_dirs['msaf'] = pl_module.msaf.visualizer.save_dir
                pl_module.msaf.visualizer.save_dir = new_dir
            
            # 保存并更新 BottleAttentionNet 的可视化器
            if (hasattr(pl_module, 'transformer') and 
                hasattr(pl_module.transformer, 'transformer') and 
                hasattr(pl_module.transformer.transformer, 'visualizer') and 
                pl_module.transformer.transformer.visualizer is not None):
                self.original_dirs['transformer'] = pl_module.transformer.transformer.visualizer.save_dir
                pl_module.transformer.transformer.visualizer.save_dir = new_dir
        
        def _restore_visualizer_dir(self, pl_module):
            """恢复所有可视化器的原始保存目录"""
            # 恢复 MSAF 的可视化器
            if 'msaf' in self.original_dirs and hasattr(pl_module, 'msaf') and hasattr(pl_module.msaf, 'visualizer'):
                pl_module.msaf.visualizer.save_dir = self.original_dirs['msaf']
            
            # 恢复 BottleAttentionNet 的可视化器
            if ('transformer' in self.original_dirs and 
                hasattr(pl_module, 'transformer') and 
                hasattr(pl_module.transformer, 'transformer') and 
                hasattr(pl_module.transformer.transformer, 'visualizer')):
                pl_module.transformer.transformer.visualizer.save_dir = self.original_dirs['transformer']
        
        def _generate_msaf_visualizations(self, msaf):
            """生成 MSAF 模型的所有可视化图表"""
            if not hasattr(msaf, 'visualizer') or msaf.visualizer is None:
                return
            
            try:
                # 生成注意力图
                msaf.visualizer.visualize_all_attention_maps(prefix="MSAF_Best_")
                
                # 生成特征分布图 (t-SNE)
                msaf.visualizer.compare_feature_distributions(method='tsne')
                
                # 生成特征分布图 (PCA)
                msaf.visualizer.compare_feature_distributions(method='pca')
                
                print("已生成 MSAF 模型的注意力图和特征分布图")
            except Exception as e:
                print(f"生成 MSAF 可视化时出错: {str(e)}")
        
        def _generate_bottle_attention_visualizations(self, bottle_attention):
            """生成 BottleAttentionNet 模型的所有可视化图表"""
            if not hasattr(bottle_attention, 'visualizer') or bottle_attention.visualizer is None:
                return
            
            try:
                # 生成注意力图
                bottle_attention.visualizer.visualize_all_attention_maps(prefix="BottleAttention_Best_")
                
                # 生成模态间注意力图
                bottle_attention.visualizer.visualize_attention_between_modalities(
                    modality1='audio', 
                    modality2='visual', 
                    title="Best_Cross-Modal_Attention"
                )
                
                # 生成特征分布图 (t-SNE)
                bottle_attention.visualizer.compare_feature_distributions(method='tsne')
                
                # 生成特征分布图 (PCA)
                bottle_attention.visualizer.compare_feature_distributions(method='pca')
                
                print("已生成 BottleAttention 模型的注意力图和特征分布图")
            except Exception as e:
                print(f"生成 BottleAttention 可视化时出错: {str(e)}")
    
    # 添加最佳模型可视化回调
    best_viz_callback = BestModelVisualizationCallback(monitor="val/the_metric", mode="max")
    callbacks = [checkpoint_callback, lr_callback, MetricsPlottingCallback(), best_viz_callback]
    
    num_gpus = (
        _config["gpus"]
        if isinstance(_config["gpus"], int)
        else len(_config["gpus"])
    )

    total_bs = _config.get("per_gpu_batchsize", 0) * num_gpus * _config.get("num_nodes", 1)

    grad_steps = max(_config.get("batch_size", 0) // total_bs, 1)
    
    
    trainer = pl.Trainer(
        devices=_config["gpus"],               # 'auto'每个节点应用的要训练的 GPU 数量（int）或要训练的 GPU（列表或 str）
        # auto_select_gpus=True,               # 自动选择所有可用的gpu
        num_nodes=_config["num_nodes"],        # 分布式训练的GPU节点数。
        accelerator="gpu",                     # 加速器类型
        strategy="ddp_find_unused_parameters_true",  # 现在并不是所有的参数都参与到损失计算中了！！！
        benchmark=True,
        deterministic=False,
        accumulate_grad_batches=grad_steps,    # 每 k 个批次或按照字典中的设置累积梯度
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        # replace_sampler_ddp=True,           # 显式启用或禁用采样器替换。 如果没有指定这个使用 DDP 时会自动切换。
        log_every_n_steps=10,   
        # flush_logs_every_n_steps=10,         # 将日志刷新到磁盘的频率（默认为每 100 步）
        # weights_summary="top",               # 训练开始时打印权重摘要
        fast_dev_run=_config["fast_dev_run"],  # 如果设置为n（int），则运行 n，否则如果设置为True，则运行 1 批处理训练、验证和测试来查找任何错误
        val_check_interval=_config["val_check_interval"],
    )

    # 决定是进行模型训练还是测试
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
