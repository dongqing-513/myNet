import os
import glob
import json
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from model.TVLTmodules.dist_utils import all_gather

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def get_mask_from_lengths(lengths, max_len=None, inv=True):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, dtype=torch.int32, device=lengths.device)
    if inv:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
    else:
        mask = ids.unsqueeze(0).expand(lengths.size(0), -1) < lengths.unsqueeze(1)
    return mask



def compute_mlm(pl_module, batch):
    # MLM (Masked Language Modeling) 损失
    # 用于预测被遮蔽的文本标记，帮助模型理解文本语义
    # 建议：考虑增加难例挖掘，关注预测错误较多的标记类型

    infer = pl_module.infer(batch, mask_text=True, mask_visual=False)
    mlm_logits = pl_module.transformer.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_mae_video(pl_module, batch, patch_size=16, num_patches=14):
    
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    pred = pl_module.transformer.mae_score_video(infer["video_feats"])    
    target = patchify_video(infer['video'], patch_size)
    mask = infer['video_masks']
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    ret = {
        "mae_video_loss": loss*0.3,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mae_video_loss")(ret["mae_video_loss"])

    pl_module.log(f"mae_video/{phase}/loss", loss)

    return ret


def patchify_video(vids, p=16):
    """
    imgs: (N, T, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    t = vids.shape[1]
    h = vids.shape[3] // p
    w = vids.shape[4] // p
    x = vids.reshape(shape=(vids.shape[0], t, vids.shape[2], h, p, w, p))
    x = torch.einsum('ntchpwq->nthwpqc', x)
    x = x.reshape(shape=(vids.shape[0], h * w * t, p**2 * vids.shape[2]))
    return x
    
    
def compute_mae_audio(pl_module, batch, audio_patch_size=[2,128]):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    
    pred = pl_module.transformer.mae_score_audio(infer["audio_feats"])   
    target = patchify_audio(infer['audio'], audio_patch_size[0], audio_patch_size[1])
    mask = infer['audio_masks']
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mae_audio_loss": loss*3.0,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mae_audio_loss")(ret["mae_audio_loss"])

    pl_module.log(f"mae_audio/{phase}/loss", loss)

    return ret


def patchify_audio(audios, p1=16, p2=16):
    """
    audios: (N, 1, H, W)
    x: (N, L, patch_size**2 *3)
    """
    h = audios.shape[2] // p1
    w = audios.shape[3] // p2
    x = audios.reshape(shape=(audios.shape[0], audios.shape[1], h, p1, w, p2))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(audios.shape[0], h * w, p1 * p2 * audios.shape[1]))
    return x
    
    
def denormalize(x):
    return (np.clip(x, -1.0, 1.0) + 1.0)/2.0


def compute_mae_joint(pl_module, batch, patch_size=16, audio_patch_size=[2,128]):
    infer = pl_module.infer(batch, mask_text=False, mask_visual=True, use_mae=True)
    
    pred_a = pl_module.transformer.mae_score_audio(infer["audio_feats"])   
    target_a = patchify_audio(infer['audio'], audio_patch_size[0], audio_patch_size[1])
    mask_a = infer['audio_masks']
    loss_a = (pred_a - target_a) ** 2
    loss_a = loss_a.mean(dim=-1)  # [N, L], mean loss per patch
    loss_a = (loss_a * mask_a).sum() / mask_a.sum()   # mean loss on removed patches

    pred_v = pl_module.transformer.mae_score_video(infer["video_feats"])    
    target_v = patchify_video(infer['video'], patch_size)
    mask_v = infer['video_masks']    
    loss_v = (pred_v - target_v) ** 2
    loss_v = loss_v.mean(dim=-1)  # [N, L], mean loss per patch
    loss_v = (loss_v * mask_v).sum() / mask_v.sum()   # mean loss on removed patches  

    
    ret = {
        "mae_audio_loss": loss_a,
        "mae_video_loss": loss_v
    }

    phase = "train" if pl_module.training else "val"
    loss_a = getattr(pl_module, f"{phase}_mae_audio_loss")(ret["mae_audio_loss"])
    loss_v = getattr(pl_module, f"{phase}_mae_video_loss")(ret["mae_video_loss"])

    pl_module.log(f"mae_audio/{phase}/loss_a", loss_a)
    pl_module.log(f"mae_video/{phase}/loss_v", loss_v)

    return ret


def compute_vam(pl_module, batch):
    # VAM (Video-Audio Matching) 损失
    # 用于对齐视频和音频的多模态表示
    # 建议：
    # 1. 添加时序一致性约束
    # 2. 考虑引入多尺度对比学习

    if len(batch["audio_data"]) > 1:
        pos_len = len(batch["audio_data"]) // 2
        neg_len = len(batch["audio_data"]) - pos_len
    else:
        pos_len = 1 if np.random.rand() < 0.5 else 0
        neg_len = 1 - pos_len
    vam_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vam_labels = vam_labels[torch.randperm(vam_labels.size(0))]
    vam_videos = torch.stack(
            [
                ti if vam_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(batch["video_data"], batch["false_video_0"]))
            ]
        )    
    batch = {k: v for k, v in batch.items()}
    batch["video_data"] = vam_videos

    infer = pl_module.infer(batch, mask_text=False, mask_visual=False, use_mae=False)
    vam_logits = pl_module.transformer.matching_score(infer["cls_feats"])
    vam_loss = F.binary_cross_entropy_with_logits(vam_logits.squeeze(), vam_labels.squeeze())
    ret = {
        "vam_loss": vam_loss,
        "vam_logits": vam_logits,
        "vam_labels": vam_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vam_loss")(ret["vam_loss"])
    
    acc = getattr(pl_module, f"{phase}_vam_accuracy")(
        ret["vam_logits"], ret["vam_labels"]
    )
    pl_module.log(f"vam/{phase}/loss", loss)
    pl_module.log(f"vam/{phase}/accuracy", acc)

    return ret


def compute_vtm(pl_module, batch):
    # VTM (Video-Text Matching) 损失
    # 用于对齐视频和文本的语义表示
    # 建议：
    # 1. 添加硬负样本挖掘
    # 2. 考虑使用 InfoNCE 损失替代现有的对比损失

    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    vtm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    vtm_labels = vtm_labels[torch.randperm(vtm_labels.size(0))]
    vtm_videos = torch.stack(
            [
                ti if vtm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(batch["video_data"], batch["false_video_0"]))
            ]
        )    
    batch = {k: v for k, v in batch.items()}
    batch["video_data"] = vtm_videos

    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    
    vtm_logits = pl_module.transformer.matching_score(infer["cls_feats"])
    vtm_loss = F.binary_cross_entropy_with_logits(vtm_logits, vtm_labels.unsqueeze(1))
    ret = {
        "vtm_loss": vtm_loss,
        "vtm_logits": vtm_logits,
        "vtm_labels": vtm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vtm_loss")(ret["vtm_loss"])
    acc = getattr(pl_module, f"{phase}_vtm_accuracy")(
        ret["vtm_logits"], ret["vtm_labels"]
    )
    pl_module.log(f"vtm/{phase}/loss", loss)
    pl_module.log(f"vtm/{phase}/accuracy", acc)

    return ret

# 将特征值从一个连续的数值范围离散化为两个类别，二值化处理               
def a2_parse(a):
    if a < 0:
            res = 0
    else: 
            res = 1
    return res

# 将输入的一组数值转换为对应的二维独热编码（one-hot encoding）表示
def get_logits_a2(score):
    eyes = torch.eye(2).to(score.device)
    score = torch.tensor([a2_parse(item) for item in score]).to(score.device)
    return eyes[score]


# 六分类任务 快乐、悲伤、愤怒、恐惧、厌恶、惊讶
# mosei_score是一个形状为torch.Size([4, 6])的张量
def compute_moseiemo(pl_module, batch):
    # MOSEI 情绪分类任务的损失
    # 建议：
    # 1. 使用加权交叉熵损失处理类别不平衡
    # 2. 考虑多任务学习框架

    # 模型推理
    # 进行推理，传入批次数据、不掩码文本和视觉内容的标志。这一步提取出文本、音频和视频的特征等信息
    infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    # print(f"After infer, cls_feats size: {infer['cls_feats'].size()}")  # torch.Size([4, 1577, 768])
    # 使用pl_module的transformer对象的分类器对推理得到的特征（cls_feats）进行处理，得到情感分数预测。
    mosei_score = pl_module.transformer.classifier(infer["cls_feats"])
    # print(f"After classifier, mosei_score size: {mosei_score.size()}")  #  torch.Size([4, 6]) 有 4 个样本，每个样本对应 6 个类别
    
    # 计算损失
    # 从批次数据中提取情感标签列表，并转换为张量并移动到与模型相同的设备上。
    labels = torch.tensor(batch["emolist"]).to(pl_module.device).float()
    # print(f"labels size: {labels.size()}") #  torch.Size([4, 0])
    # 使用二元交叉熵损失函数计算预测分数和真实标签之间的损失。
    # 输入张量 目标张量
    mosei_loss = F.binary_cross_entropy_with_logits(mosei_score, labels)
    # print(f"mosei_loss size: {mosei_loss.size()}")
    
    # 构建返回字典
    # 创建一个包含损失、预测分数和真实标签的字典，用于后续的处理和返回
    ret = {
        "moseiemo_loss": mosei_loss,
        "moseiemo_score": mosei_score,
        "moseiemo_labels": labels
    }

    # 根据训练阶段计算特定损失和准确率
    # 确定当前是训练阶段还是验证阶段。
    phase = "train" if pl_module.training else "val"
    # 获取当前阶段的损失处理方法，并传入计算得到的损失值
    loss = getattr(pl_module, f"{phase}_moseiemo_loss")(ret["moseiemo_loss"])
    
    # 分别针对不同的情感类别（快乐、悲伤、愤怒、恐惧、厌恶、惊讶）计算准确率
    happy = getattr(pl_module, f"{phase}_moseiemo_happy")(mosei_score[:, 0:1], labels[:, 0:1])
    sad = getattr(pl_module, f"{phase}_moseiemo_sad")(mosei_score[:, 1:2], labels[:, 1:2])
    angry = getattr(pl_module, f"{phase}_moseiemo_angry")(mosei_score[:, 2:3], labels[:, 2:3])
    fear = getattr(pl_module, f"{phase}_moseiemo_fear")(mosei_score[:, 3:4], labels[:, 3:4])
    disgust = getattr(pl_module, f"{phase}_moseiemo_disgust")(mosei_score[:, 4:5], labels[:, 4:5])
    surprise = getattr(pl_module, f"{phase}_moseiemo_surprise")(mosei_score[:, 5:6], labels[:, 5:6])
    
    # 记录不同阶段和不同情感类别的损失和准确率到日志中，以便在训练过程中进行监控和分析。
    pl_module.log(f"moseiemo/{phase}/loss", loss)
    pl_module.log(f"moseiemo/{phase}/happy", happy)
    pl_module.log(f"moseiemo/{phase}/sad", sad)
    pl_module.log(f"moseiemo/{phase}/angry", angry)
    pl_module.log(f"moseiemo/{phase}/fear", fear)
    pl_module.log(f"moseiemo/{phase}/disgust", disgust)
    pl_module.log(f"moseiemo/{phase}/surprise", surprise)

    return ret

"""
def compute_mosei(pl_module, batch):
    mosei_score = pl_module.infer(batch, mask_text=False, mask_visual=False)
    # print("mosei_score.key",mosei_score.keys()) infer函数返回的ret
    # print("mosei_score['hidden_size']",mosei_score["hidden_size"].size())
    # msaf_mosei的返回，torch.Size([1, 1]) mosei_score：情感分数预测
    # mosei_score = mosei_score["hidden_size"]
    # print("moser_score",mosei_score) # tensor([[-2.1468]], device='cuda:0', grad_fn=<MeanBackward1>)
    # infer = pl_module.infer(batch, mask_text=False, mask_visual=False)
    # print("\nmosei_score[hidden_size]",mosei_score["hidden_size"].shape)
    mosei_score = pl_module.transformer.classifier(mosei_score["hidden_size"])
    
    score_label = torch.tensor(batch["score"]).to(pl_module.device)
    # 将二分类问题视为一个回归问题，将模型的输出视为一个连续的预测得分，目标是使这个预测得分尽可能接近真实标签
    mosei_loss = F.mse_loss(mosei_score.squeeze(), score_label.squeeze(), size_average=None, reduce=None, reduction='mean') 
    ret = {
        "mosei_loss": mosei_loss,
        "mosei_score": mosei_score,
        "mosei_labels2": torch.tensor(batch["label2"]).to(pl_module.device),
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mosei_loss")(ret["mosei_loss"])
    
    acc2 = getattr(pl_module, f"{phase}_mosei_accuracy2")(
        get_logits_a2(mosei_score), ret["mosei_labels2"]
    )
    
    # 计算F1 Score及其相关指标
    f1_metrics = getattr(pl_module, f"{phase}_mosei_f1score")(
        get_logits_a2(mosei_score), ret["mosei_labels2"]
    )
    
    # 记录当前阶段的损失和准确率到日志中，以便在训练过程中进行监控和分析。
    pl_module.log(f"mosei/{phase}/loss", loss)
    pl_module.log(f"mosei2/{phase}/accuracy2", acc2)
    pl_module.log(f"mosei/{phase}/f1", f1_metrics["f1"])
    pl_module.log(f"mosei/{phase}/precision", f1_metrics["precision"])
    pl_module.log(f"mosei/{phase}/recall", f1_metrics["recall"])
    #pl_module.log(f"mosei/{phase}/logits_mean", logits.mean())
    #pl_module.log(f"mosei/{phase}/logits_std", logits.std())

    return ret
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


def a2_parse(item):
    # 这里根据你的实际需求修改解析逻辑
    if item >= 0:
        return 1
    else:
        return 0


# 将输入的一组数值转换为对应的二维独热编码（one-hot encoding）表示
def get_logits_a2(score):
    eyes = torch.eye(2).to(score.device)
    score = torch.tensor([a2_parse(item) for item in score]).to(score.device)
    return eyes[score]


def compute_mosei(pl_module, batch):
    # MOSEI 情感分析任务的损失
    # 处理了类别不平衡问题
    # 关注难分类样本
    # 特别处理边界情况
    # 防止过拟合和过度自信

    # 模型推理
    # 进行推理，传入批次、不掩码文本和视觉内容的标志，提取出文本、音频和视频的特征
    mosei_score = pl_module.infer(batch, mask_text=False, mask_visual=False)
    # print(f"mosei_score keys: {mosei_score.keys()}, hidden_size: {mosei_score['hidden_size']}")
    # msaf_mosei的返回，torch.Size([1, 1]) mosei_score：情感分数预测
    # 使用pl_module的transformer对象的分类器对推理得到的特征（cls_feats）进行处理，得到情感分数预测。
    mosei_score = pl_module.transformer.classifier(mosei_score["hidden_size"])
    # print(f"mosei_score after classifier: {mosei_score.size()}")
    # tensor([[-2.1468]],[1, 1] device='cuda:0', grad_fn=<MeanBackward1>)

    # 获取原始分数和二分类标签
    score_label = torch.tensor(batch["score"]).to(pl_module.device)
    # print(f"score_label size: {score_label.size()}") # ([1, 2])
    # binary_labels = (score_label >= 0).float()  # 转换为二分类标签
    binary_labels = (score_label >= 0).float()  # 转换为二分类标签

    # 将 binary_labels 扩展为 [1, 2] 的独热编码形式
    # binary_labels = torch.nn.functional.one_hot(binary_labels.long(), num_classes=2).float()
    # print("binary_labels: ", binary_labels.size())  # 确保形状为 [1, 2]

    # 标签平滑处理，减少过拟合
    smoothing = 0.1
    binary_labels = binary_labels * (1 - smoothing) + smoothing / 2

    # 由于已经在数据层面实现了平衡采样，这里使用更温和的类别权重
    pos_count = (binary_labels > 0.5).sum()
    neg_count = (binary_labels <= 0.5).sum()
    total = pos_count + neg_count
    
    # 使用batch内的实际比例计算权重，但限制权重范围避免过度补偿
    weight_scale = min(max(pos_count / neg_count, 0.5), 2.0)
    pos_weight = 1.0
    neg_weight = weight_scale
    
    sample_weights = torch.where(binary_labels > 0.5, 
                               pos_weight * torch.ones_like(binary_labels),
                               neg_weight * torch.ones_like(binary_labels))

    # 主分类损失（BCE损失）
    logits = torch.sigmoid(mosei_score.squeeze())
    bce_loss = F.binary_cross_entropy(
        logits,
        binary_labels.squeeze(),
        reduction='none'
    )
    weighted_bce_loss = (bce_loss * sample_weights.squeeze()).mean()

    # 改进的Focal Loss - 使用更小的gamma值，因为数据已经平衡
    gamma = 2.0  # 对所有样本使用相同的gamma
    pt = torch.where(binary_labels > 0.5, logits, 1 - logits)
    focal_weight = (1 - pt) ** gamma
    focal_loss = -(focal_weight * torch.log(pt + 1e-7)).mean()  # 移除sample_weights

    # 置信度正则化损失
    confidence_reg = -torch.mean(torch.log(torch.clamp(torch.abs(logits - 0.5), min=1e-7)))

    # 边界样本辅助损失 - 保持不变，因为这与类别平衡无关
    boundary_threshold = 0.33
    boundary_mask = (torch.abs(score_label) < boundary_threshold).float()
    boundary_loss = F.binary_cross_entropy(
        logits,
        binary_labels.squeeze(),
        reduction='none'
    ) * boundary_mask.squeeze()
    boundary_loss = boundary_loss.mean()

    # 组合所有损失组件
    loss_components = {
        "bce_loss": weighted_bce_loss.item(),
        "focal_loss": focal_loss.item(),
        "confidence_reg": confidence_reg.item(),
        "boundary_loss": boundary_loss.item()
    }
    
    # 计算总损失
    loss = (
        weighted_bce_loss * 1.0 +  # 主损失
        focal_loss * 0.5 +         # 降低focal loss的权重
        confidence_reg * 0.1 +     # 保持小的正则化权重
        boundary_loss * 0.3        # 保持边界损失权重
    )

    # 计算当前batch的类别分布
    batch_stats = {
        "pos_ratio": (binary_labels > 0.5).float().mean().item(),
        "neg_ratio": (binary_labels <= 0.5).float().mean().item(),
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
        "boundary_ratio": boundary_mask.float().mean().item()
    }

    ret = {
        "mosei_loss": loss,
        "mosei_score": mosei_score,
        "mosei_labels2": torch.tensor(batch["label2"]).to(pl_module.device),
        # 添加详细的损失组件和统计信息
        "loss_components": loss_components,
        "batch_stats": batch_stats
    }

    # 根据训练阶段计算特定损失和准确率
    # 确定当前是训练阶段还是验证阶段。
    phase = "train" if pl_module.training else "val"
    # 获取当前阶段的损失处理方法，并传入计算得到的损失值
    loss = getattr(pl_module, f"{phase}_mosei_loss")(ret["mosei_loss"])

    acc2 = getattr(pl_module, f"{phase}_mosei_accuracy2")(
        get_logits_a2(mosei_score), ret["mosei_labels2"]
    )

    # 计算F1 Score及其相关指标
    f1_metrics = getattr(pl_module, f"{phase}_mosei_f1score")(
        get_logits_a2(mosei_score), ret["mosei_labels2"]
    )

    # 记录当前阶段的损失和准确率到日志中，以便在训练过程中进行监控和分析。
    pl_module.log(f"mosei/{phase}/loss", loss)
    pl_module.log(f"mosei2/{phase}/accuracy2", acc2)
    pl_module.log(f"mosei/{phase}/f1", f1_metrics["f1"])
    pl_module.log(f"mosei/{phase}/precision", f1_metrics["precision"])
    pl_module.log(f"mosei/{phase}/recall", f1_metrics["recall"])
    pl_module.log(f"mosei/{phase}/logits_mean", logits.mean())
    pl_module.log(f"mosei/{phase}/logits_std", logits.std())

    # 返回包含损失、预测分数和标签的字典
    return ret


@torch.no_grad()
def compute_vrar_recall(pl_module):
    audio_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(audio_only=True)
    audio_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    audio_loader = torch.utils.data.DataLoader(
        audio_dset,
        batch_size=4,
        shuffle=False,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            audio_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    video_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        video_only=True
    )
    video_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(video_dset, shuffle=False)
    video_loader = torch.utils.data.DataLoader(
        video_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        sampler=dist_sampler,
        collate_fn=functools.partial(
            video_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    audio_preload = list()
    for _b in tqdm.tqdm(audio_loader, desc="audio prefetch loop"):
        audio_preload.append(
            [_b["audio_data"].to(pl_module.device),
             _b["a_index"],
            ],
        )
        
    aids = list()
    for pre in audio_preload:
        aids += pre[1]
    aids = torch.tensor(aids)

    video_preload = list()
    for _b in tqdm.tqdm(video_loader, desc="video prefetch loop"):
        video_preload.append(
            [_b["video_data"].to(pl_module.device),
             _b["v_index"],
            ],
        )
        
    rank_scores = list()
    rank_iids = list()

    num_samples = 10
    count = 0
    for video_batch in tqdm.tqdm(video_preload, desc="rank loop"):
        count += 1
        _ve, _vid = video_batch
        _, l, c, h, w = _ve.shape

        video_batch_score = list()
        for audio_batch in audio_preload:
            fblen = len(audio_batch[0])
            ve = _ve.expand(fblen, l, c, h, w)
            
            with torch.cuda.amp.autocast():
                score = pl_module.transformer.matching_score(
                    pl_module.infer(
                        {
                        "audio_data": audio_batch[0],
                        "video_data": ve,
                        }
                    )["cls_feats"]
                )[:, 0]
                score = F.sigmoid(score)
            video_batch_score.append(score)
    
        video_batch_score = torch.cat(video_batch_score)
        rank_scores.append(video_batch_score.cpu().tolist()) 
        rank_iids.append(_vid)

        if count == num_samples:
            break
            
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = aids[topk10.indices]
    topk5_iids = aids[topk5.indices]
    topk1_iids = aids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (aids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (aids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (aids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


    
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
