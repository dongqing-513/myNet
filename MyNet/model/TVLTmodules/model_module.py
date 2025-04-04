import copy
import json
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import AdamW
from transformers import get_cosine_schedule_with_warmup
from model.TVLTmodules import heads, objectives, model_utils, msaf_mosei, text_lstm
import model.TVLTmodules.tvlt as tvlt

from huggingface_sb3 import load_from_hub

import model.TVLTmodules.msaf_mosei as msaf
import model.TVLTmodules.text_lstm as text_lstm


# PyTorch Lightning 框架下进行深度学习模型的训练和评估
class Transformer(pl.LightningModule):
    # config字典作为参数，用于配置模型的各种超参数
    def __init__(self, config, model_type='transformer'):
        super().__init__()

        self.model_type = model_type
        self.learning_rate = config["learning_rate"] # 学习率
        self.weight_decay = config["weight_decay"]   # 权重衰减
        self.patch_size = config["patch_size"]       # 图像补丁大小
        self.audio_patch_size = config["audio_patch_size"] # 音频补丁大小
        self.warmup_steps = config["warmup_steps"]   # 预热步数
        self.max_epoch = config["max_epoch"]         # 最大训练轮次
        self.max_steps = config["max_steps"]         # 最大步数
        self.hidden_size= config['hidden_size']       # 隐藏层大小
        self.num_heads= config['num_heads']       # 多头注意力头数
        self.num_layers= config['num_layers']       # 多头注意力层数
        self.skip_interval= config['skip_interval']       # 跳跃间隔
        self.fusion_type= config['fusion_type']       # 融合类型
        self.drop_rate= config['drop_rate']       # dropout率
        self.normalize_before= config['normalize_before']       # 是否在前面进行归一化

        # 特征网络：使用tvlt的读取音视频， 并通过一个encoder
        # model_type='mae_vit_base_patch16_dec512d8b'
        self.transformer = getattr(tvlt, config["model_type"])(config=config)
        # self.transformer = TVLT(
        #         patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        #         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        #         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # 特征网络：文本
        modalities = ('bert', 'visual', 'audio')
        # 加载bert和两层lstm的模型到model
        model = text_lstm.BERTTextLSTMNet(config=config)
        model_param = {
            'bert': {
                'model': model,
                'id': modalities.index('bert')
            },
            'visual': {
                'id': modalities.index('visual')
            },
            'audio': {
                'id': modalities.index('audio')
            },
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'skip_interval': self.skip_interval,
            'fusion_type': self.fusion_type,
            'drop_rate': self.drop_rate,
            'normalize_before': self.normalize_before
        }

        # 文本特征 + 融合网络
        self.msaf = msaf.MSAFLSTMNet(model_param)


        # 保存超参数
        self.save_hyperparameters()
        # 设置模型的评估指标
        model_utils.set_metrics(self)
        self.current_tasks = list()
        # 应用权重初始化函数到当前实例和模型
        self.apply(objectives.init_weights)
        self.transformer.init_weights()

        # ===================== load checkpoints ======================

        # 从本地加载模型检查点
        if config["load_local_path"]:
            print(f"Loading checkpoint from {config['load_local_path']}")
            state_dict = torch.load(config["load_local_path"], map_location="cpu", weights_only=True)

            # 打印权重文件的键
            # print("Keys in checkpoint:", state_dict.keys())

            if "model" in state_dict.keys():
                # 是否严格加载，要求预训练模型和当前模型的结构完全一致
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict.keys():
                state_dict = state_dict['state_dict']

            # 检查关键层的权重是否存在
            key_layers = ['classifier.0.dense.weight', 'classifier.1.weight', 'classifier.2.weight', 'classifier.4.weight']
            for key in key_layers:
                if key in state_dict:
                    print(f"Found {key} in checkpoint, shape:", state_dict[key].shape)
                else:
                    print(f"Warning: {key} not found in checkpoint")

            # 加载权重并捕获任何不匹配
            try:
                self.load_state_dict(state_dict, strict=config['strict_load'])
                print("Successfully loaded checkpoint")
            except Exception as e:
                print("Error loading checkpoint:", str(e))

        if config["load_hub_path"]:
            ckpt_path = load_from_hub(repo_id="TVLT/models", filename=config["load_hub_path"])
            self.transformer.load_state_dict(torch.load(ckpt_path), strict=config['strict_load'])

        # 添加损失值监控 检查损失值并调整学习率 当损失值上升时，将学习率乘以0.95
        # self.previous_loss = float('inf')
        # self.lr_decay = 0.95  # 学习率衰减因子

    def infer(
        self,
        batch,
        audio_embeds=None,     # 输入的批次数据，包含文本、音频和视频数据。
        audio_masks=None,
        video_embeds=None,
        video_masks=None,
        audio_token_type_idx=1, # 音频标记类型索引
        video_token_type_idx=2,
        mask_text=False,
        mask_visual=False,
        use_mae=False        # 是否使用掩码自编码器
    ):
        """
        推断函数，根据输入的批次数据进行推理。

        处理流程：
        1. 提取文本、音频、视频特征
        2. 使用transformer处理音视频特征
        3. 使用msaf进行多模态融合

        返回值：
        - ret：包含推理结果的字典
        """

        # 根据 mask_text 的值确定是否在文本键后添加 "_mlm"
        do_mlm = "_mlm" if mask_text else ""
        videokey = "video_data"
        audiokey = "audio_data"
        txtkey   = "txt_data"   # our_00
        textkey = "text_ids"+do_mlm

        # 判断批次数据中是否包含音频和视频键
        use_audio = audiokey in list(batch.keys())
        use_video = videokey in list(batch.keys())
        has_text = textkey in list(batch.keys())

        # 如果有文本数据
        if has_text:
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            # 如果键存在且 mask_text 为 True，则获取文本标签掩码
            text_labels_mlm = batch[f"text_labels_mlm"] if f"text_labels_mlm" in batch.keys() and mask_text else None
        else:
            text_ids = None
            text_labels = None
            text_masks = None
            text_embeds = None
            text_labels_mlm = None

        # 如果有音频数据，则从批次中获取音频数据，否则设置为 None
        if use_audio:
            audio = batch[audiokey]
        else:
            audio = None

        # 如果有视频数据，则从批次中获取视频数据，否则设置为 None
        if use_video:
            video = batch[videokey]
        else:
            video = None

        text_feats, audio_feats, video_feats = None, None, None
        audio_labels_mlm = video_labels_mlm = None

        # 1. 使用transformer提取音视频特征
        av = self.transformer(text_ids=text_ids, text_masks=text_masks, audio=audio,
                             audio_masks=audio_masks, video=video, video_masks=video_masks,
                             mask_visual=mask_visual, use_mae=use_mae)

        # 2. 使用msaf进行多模态融合
        # msaf包含：
        # - BERT+LSTM处理文本
        # - 多模态特征融合
        text_tokens = batch[txtkey][0]  # 原始文本tokens
        attention_mask = batch['attention_mask'][0]  # 文本的attention mask，需要取[0]因为是batch数据
        hidden_size = self.msaf(av, text_tokens, attention_mask, return_attention=False)

        # 3. 返回所有需要的特征和输出
        ret = {
            "text_feats": text_feats,
            "audio_feats": audio_feats,
            "video_feats": video_feats,
            "video_masks": video_masks,
            "video": video,
            "audio_masks": audio_masks,
            "audio": audio,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            # add
            "hidden_size":hidden_size
        }

        return ret

    def forward(self, batch):
        ret = dict()
        # 如果当前任务列表为空，调用 infer 函数进行推理，并更新结果字典
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # 根据当前任务列表进行不同的计算和更新结果字典

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "mae_audio" in self.current_tasks and "mae_video" in self.current_tasks:
            ret.update(objectives.compute_mae_joint(self, batch, self.patch_size, self.audio_patch_size))

        # Masked Patch Prediction
        elif "mae_audio" in self.current_tasks:
            ret.update(objectives.compute_mae_audio(self, batch, self.audio_patch_size))

        elif "mae_video" in self.current_tasks:
            ret.update(objectives.compute_mae_video(self, batch, self.patch_size))

        # 会用到，在每次训练开始前自动加载dataloader
        if "mosei" in self.current_tasks:
            ret.update(objectives.compute_mosei(self, batch))

        if "moseiemo" in self.current_tasks:
            ret.update(objectives.compute_moseiemo(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        """训练步骤

        职责：
        1. 获取模型输出
        2. 记录日志
        3. 返回损失（损失计算在objectives.py中）

        参数：
            batch: 输入的数据批次
            batch_idx: 批次索引

        返回：
            total_loss: 计算得到的总损失
        """
        # 1. 获取模型输出
        model_utils.set_task(self)
        output = self(batch)  # 这会调用forward，进而调用objectives.compute_mosei

        # 2. 获取损失
        total_loss = sum([v for k, v in output.items() if "loss" in k and isinstance(v, torch.Tensor)])

        # 3. 记录日志
        self.log("train/total_loss", total_loss)
        """
        # 检查损失值并调整学习率 当损失值上升时，将学习率乘以0.95
        if hasattr(self, 'previous_loss') and total_loss > self.previous_loss:
            current_lr = self.learning_rate * self.lr_decay
            self.learning_rate = current_lr
            # 更新优化器的学习率
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = current_lr
            self.log("train/learning_rate", current_lr)

        self.previous_loss = total_loss
        """
        return total_loss

    # 用于在每个 epoch 结束时进行一些清理、统计或其他与模型训练相关的操作。可能包括更新指标、保存中间结果、重置某些状态变量等。
    # def training_epoch_end(self, outs):
    def on_train_epoch_end(self):
        model_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        model_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        model_utils.epoch_wrapup(self)  # 在验证步骤结束时调用 epoch_wrapup

    def test_step(self, batch, batch_idx):
        model_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        model_utils.epoch_wrapup(self)


    # 配置模型的优化器和学习率调度器。
    def configure_optimizers(self):
        # 为不同组件创建不同的参数组
        no_decay = ["bias", "LayerNorm.weight"]

        # Transformer参数组
        transformer_params = [
            {
                "params": [p for n, p in self.transformer.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate
            },
            {
                "params": [p for n, p in self.transformer.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.learning_rate
            }
        ]

        # MSAF参数组 - 使用较小的学习率
        msaf_params = [
            {
                "params": [p for n, p in self.msaf.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate * 0.1  # MSAF使用较小的学习率
            },
            {
                "params": [p for n, p in self.msaf.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.learning_rate * 0.1
            }
        ]

        # 合并所有参数组
        optimizer_grouped_parameters = transformer_params + msaf_params

        # 使用AdamW优化器，添加梯度裁剪
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )

        # 计算训练步数
        num_training_steps = self.max_steps
        warmup_steps = self.warmup_steps

        # 使用分段余弦学习率调度器
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=1.0  # 添加半个余弦周期
        )

        # 返回优化器和调度器配置
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss",  # 监控验证损失
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # 对不同组件分别进行梯度裁剪
        # transformer部分
        transformer_params = [p for name, p in self.transformer.named_parameters() if p.requires_grad]
        if transformer_params:
            torch.nn.utils.clip_grad_norm_(transformer_params, max_norm=0.5)

        # msaf部分
        msaf_params = [p for name, p in self.msaf.named_parameters() if p.requires_grad]
        if msaf_params:
            torch.nn.utils.clip_grad_norm_(msaf_params, max_norm=0.3)  # msaf使用更小的裁剪阈值

        # 记录梯度信息
        if self.training:
            # 计算梯度范数
            grad_norm_transformer = torch.nn.utils.clip_grad_norm_(transformer_params, float('inf'))
            grad_norm_msaf = torch.nn.utils.clip_grad_norm_(msaf_params, float('inf'))

            # 记录梯度范数
            self.log('grad/norm_transformer', grad_norm_transformer, on_step=True, on_epoch=True)
            self.log('grad/norm_msaf', grad_norm_msaf, on_step=True, on_epoch=True)
