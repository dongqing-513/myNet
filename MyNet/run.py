import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import json
import copy
import pytorch_lightning as pl

from model.config import ex
from model.TVLTmodules import Transformer
from model.data.datamodules.multitask_datamodule import MTDataModule
import torch
from sacred.observers import FileStorageObserver
from model.gadgets.metrics_logger import MetricsLogger

warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.")

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
    
    callbacks = [checkpoint_callback, lr_callback, MetricsPlottingCallback()]
    
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
