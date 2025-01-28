import torch
import random

from model.gadgets.my_metrics import Accuracy, Scalar, F1Score
from model.TVLTmodules.objectives import compute_vrar_recall

def set_metrics(pl_module):
    """设置模型的评估指标
    
    根据配置文件中的 loss_names 为不同任务设置相应的评估指标：
    - mae_audio/mae_video: 设置损失值指标
    - mosei: 设置准确率和损失值指标
    - moseiemo: 设置6种情绪(愤怒、厌恶、恐惧、快乐、悲伤、惊讶)的准确率和损失值指标
    - 其他任务: 设置准确率和损失值指标
    
    Args:
        pl_module: PyTorch Lightning 模块实例
    """
    for split in ["train", "val"]:  # 分别为训练集和验证集设置指标
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:  # 跳过权重小于1的任务
                continue
            if k=="mae_audio" or k=="mae_video":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=="mosei":
                setattr(pl_module, f"{split}_{k}_accuracy2", Accuracy())
                setattr(pl_module, f"{split}_{k}_f1score", F1Score())  # 添加F1 Score评估
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k=="moseiemo":
                # 为6种情绪分别设置准确率指标
                setattr(pl_module, f"{split}_{k}_angry", Accuracy())
                setattr(pl_module, f"{split}_{k}_disgust", Accuracy())
                setattr(pl_module, f"{split}_{k}_fear", Accuracy())
                setattr(pl_module, f"{split}_{k}_happy", Accuracy())
                setattr(pl_module, f"{split}_{k}_sad", Accuracy())
                setattr(pl_module, f"{split}_{k}_surprise", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
def epoch_wrapup(pl_module):
    """在每个训练/验证epoch结束时进行指标统计和日志记录
    
    主要功能：
    1. 如果启用了视频-音频召回指标，计算并记录相关指标
    2. 根据不同任务类型，计算并记录相应的指标：
       - mlm任务：记录准确率和损失
       - mae任务：记录损失
       - mosei任务：记录准确率和损失
       - moseiemo任务：记录6种情绪的准确率和损失
    3. 计算并记录总体指标(the_metric)
    
    Args:
        pl_module: PyTorch Lightning 模块实例
    """
    torch.distributed.barrier()  # 等待所有进程同步
    phase = "train" if pl_module.training else "val"
    the_metric = 0  # 用于累积所有指标的总和
    print("")
    print("=================================================")

    # 如果配置了视频-音频召回指标且在验证阶段
    if pl_module.hparams.config["get_va_recall_metric"] and not pl_module.training:
        (vr_r1, vr_r5, vr_r10, ar_r1, ar_r5, ar_r10) = compute_vrar_recall(pl_module)
        print((vr_r1, vr_r5, vr_r10, ar_r1, ar_r5, ar_r10), pl_module.global_step)
        # 记录各种召回率指标
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r1", vr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r5", vr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/vr_r10", vr_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r1", ar_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r5", ar_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ar_r10", ar_r10, pl_module.global_step
        )
        the_metric += vr_r1.item() + ar_r1.item()
        
    # 遍历所有任务，计算并记录指标
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:  # 跳过权重小于1的任务
            continue

        value = 0        
        
        if loss_name == "mlm":  # 掩码语言模型任务
            # 计算并记录准确率
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            print(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            
            # 计算并记录损失值
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "mae_audio" or loss_name == "mae_video":  # MAE任务
            # 计算并记录损失值（取负值因为是最小化目标）
            value = - getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        elif loss_name == "mosei":  # MOSEI情感分类任务
            # 计算并记录准确率
            # 'mosei/val/accuracy2' 或 'mosei/train/accuracy2'
            value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy2").compute()
            f1_metrics = getattr(pl_module, f"{phase}_{loss_name}_f1score").compute()
            
            # 分别记录F1相关指标
            pl_module.log(f"{loss_name}/{phase}/accuracy2_epoch", value2)
            pl_module.log(f"{loss_name}/{phase}/f1_epoch", f1_metrics['f1'])
            pl_module.log(f"{loss_name}/{phase}/precision_epoch", f1_metrics['precision'])
            pl_module.log(f"{loss_name}/{phase}/recall_epoch", f1_metrics['recall'])
            
            print(f"{loss_name}/{phase}/accuracy2_epoch", value2)
            print(f"{loss_name}/{phase}/f1_epoch", f1_metrics['f1'])
            print(f"{loss_name}/{phase}/precision_epoch", f1_metrics['precision'])
            print(f"{loss_name}/{phase}/recall_epoch", f1_metrics['recall'])
            
            getattr(pl_module, f"{phase}_{loss_name}_accuracy2").reset()
            getattr(pl_module, f"{phase}_{loss_name}_f1score").reset()
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            the_metric += value2
            
        elif loss_name == "moseiemo":  # MOSEI多情绪分类任务
            # 计算6种情绪的准确率
            happy = getattr(pl_module, f"{phase}_{loss_name}_happy").compute()
            sad = getattr(pl_module, f"{phase}_{loss_name}_sad").compute()
            angry = getattr(pl_module, f"{phase}_{loss_name}_angry").compute()
            fear = getattr(pl_module, f"{phase}_{loss_name}_fear").compute()
            disgust = getattr(pl_module, f"{phase}_{loss_name}_disgust").compute()
            surprise = getattr(pl_module, f"{phase}_{loss_name}_surprise").compute()
            
            # 记录各种情绪的准确率
            pl_module.log(f"{loss_name}/{phase}/happy_epoch", happy)
            pl_module.log(f"{loss_name}/{phase}/sad_epoch", sad)
            pl_module.log(f"{loss_name}/{phase}/angry_epoch", angry)
            pl_module.log(f"{loss_name}/{phase}/fear_epoch", fear)
            pl_module.log(f"{loss_name}/{phase}/disgust_epoch", disgust)            
            pl_module.log(f"{loss_name}/{phase}/surprise_epoch", surprise)
            
            print(f"{loss_name}/{phase}/happy_epoch", happy)
            print(f"{loss_name}/{phase}/sad_epoch", sad)
            print(f"{loss_name}/{phase}/angry_epoch", angry)
            print(f"{loss_name}/{phase}/fear_epoch", fear)
            print(f"{loss_name}/{phase}/disgust_epoch", disgust)            
            print(f"{loss_name}/{phase}/surprise_epoch", surprise)
            
            # 重置所有情绪指标
            getattr(pl_module, f"{phase}_{loss_name}_happy").reset()
            getattr(pl_module, f"{phase}_{loss_name}_sad").reset()
            getattr(pl_module, f"{phase}_{loss_name}_angry").reset()
            getattr(pl_module, f"{phase}_{loss_name}_fear").reset()
            getattr(pl_module, f"{phase}_{loss_name}_disgust").reset()            
            getattr(pl_module, f"{phase}_{loss_name}_surprise").reset()
            
            # 计算并记录损失值
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            print(f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            # 累加所有情绪的准确率到总指标
            the_metric += angry + disgust + fear + happy + sad + surprise
            
        the_metric += value

    # 记录总体指标 所有激活任务的指标总和
    pl_module.log(f"{phase}/the_metric", the_metric)
    print("=================================================")
    torch.distributed.barrier()  # 等待所有进程同步
    
def check_non_acc_grad(pl_module):
    """检查非累积梯度
    
    检查token_type_embeddings的梯度是否为空或全零
    
    Args:
        pl_module: PyTorch Lightning 模块实例
    
    Returns:
        bool: 如果梯度为空或全零返回True，否则返回False
    """
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()

def set_task(pl_module):
    """设置当前活动任务
    
    从配置文件中获取权重>=1的任务作为当前活动任务
    
    Args:
        pl_module: PyTorch Lightning 模块实例
    """
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return
