import random
import torch
import io
import os
import glob
import json
import re

import pandas as pd
import numpy as np

from model.data.datasets.rawvideo_utils import RawVideoExtractor
from .base_video_dataset import BaseVideoDataset
    
def a2_parse(a):
    if a < 0:
        res = 0
    else: 
        res = 1
    return res

class BalancedBatchSampler(torch.utils.data.Sampler):
    """实现类别平衡的批次采样器"""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 确保数据集有必要的属性
        if not hasattr(dataset, 'labels') or not hasattr(dataset, 'pos_indices') or not hasattr(dataset, 'neg_indices'):
            raise ValueError("Dataset must have labels and indices attributes")
            
        self.pos_indices = dataset.pos_indices
        self.neg_indices = dataset.neg_indices
        
        # 计算每个类别在批次中的样本数
        self.samples_per_class = batch_size // 2
        
        # 确保每个类别至少有一个样本
        if self.samples_per_class == 0:
            self.samples_per_class = 1
            
        # 计算总批次数
        self.num_batches = min(
            len(self.pos_indices) // self.samples_per_class,
            len(self.neg_indices) // self.samples_per_class
        )
        
    def __iter__(self):
        # 打乱每个类别的索引
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        random.shuffle(pos_indices)
        random.shuffle(neg_indices)
        
        # 生成平衡的批次
        for i in range(self.num_batches):
            batch_indices = []
            
            # 添加正样本
            start_idx = i * self.samples_per_class
            end_idx = start_idx + self.samples_per_class
            batch_indices.extend(pos_indices[start_idx:end_idx])
            
            # 添加负样本
            batch_indices.extend(neg_indices[start_idx:end_idx])
            
            # 打乱批次内的顺序
            random.shuffle(batch_indices)
            yield from batch_indices
            
    def __len__(self):
        return self.num_batches * self.batch_size

class MOSEIDataset(BaseVideoDataset):
    def __init__(self, *args, split="", use_balanced_sampler=True, **kwargs):
        self.split = split
        self.use_balanced_sampler = use_balanced_sampler
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__(*args, **kwargs)

    def get_suite(self, index):
        """获取数据样本，包含视频、音频和文本的增强
        Args:
            index: 样本索引
        Returns:
            dict: 处理后的数据样本
        """
        # 构建文件路径
        video_path = self.metadata_dir+'video/'+self.keys[index]+'.mp4'
        audio_path = self.metadata_dir+'audio/'+self.keys[index]+'.wav'
        txt_path = self.metadata_dir+'text/'+self.keys[index]+'.txt'

        ret = {}  # 初始化返回字典
            
        # 1. 加载视频数据
        video_data = self._get_video(index, video_path)
        if video_data is None:
            return None
        ret.update(video_data)
            
        # 视频数据增强
        if self.split == 'train':
            n_video_augments = self.draw_false_video
            if hasattr(self, 'labels'):
                # 对少数类样本进行更多的增强
                if self.labels[index] == (0 if self.pos_count > self.neg_count else 1):
                    n_video_augments = min(n_video_augments * 2, 5)  # 最多增强5次
                
            for i in range(n_video_augments):
                try:
                    # 从同类别中采样
                    if hasattr(self, 'labels'):
                        candidate_indices = self.pos_indices if self.labels[index] == 1 else self.neg_indices
                        random_index = random.choice(candidate_indices)
                    else:
                        random_index = random.randint(0, len(self.keys) - 1)
                    
                    video_path_f = self.metadata_dir+'video/'+self.keys[random_index]+'.mp4'
                    if os.path.exists(video_path_f):
                        ret.update(self._get_false_video(i, video_path_f))
                except Exception as e:
                    print(f"Warning: Failed to load false video {i}: {str(e)}")
                    continue
        
        # 2. 加载音频数据
        if self.use_audio:
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            audio_data = self._get_audio(index, audio_path)
            if audio_data is None:
                return None
            ret.update(audio_data)
            
            # 音频数据增强
            if self.split == 'train':
                n_audio_augments = self.draw_false_audio
                if hasattr(self, 'labels'):
                    if self.labels[index] == (0 if self.pos_count > self.neg_count else 1):
                        n_audio_augments = min(n_audio_augments * 2, 5)
                
                for i in range(n_audio_augments):
                    try:
                        if hasattr(self, 'labels'):
                            candidate_indices = self.pos_indices if self.labels[index] == 1 else self.neg_indices
                            random_index = random.choice(candidate_indices)
                        else:
                            random_index = random.randint(0, len(self.keys) - 1)
                        
                        audio_path_f = self.metadata_dir+'audio/'+self.keys[random_index]+'.wav'
                        if os.path.exists(audio_path_f):
                            ret.update(self._get_false_audio(i, audio_path_f))
                    except Exception as e:
                        print(f"Warning: Failed to load false audio {i}: {str(e)}")
                        continue
            
        # 3. 加载文本数据
        if self.use_text:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                pattern = r"<s>(.*?)</s>"
                match = re.search(pattern, content)
                if match:
                    text = self._clean_text(match.group(1))
                    text_data = self._get_text(index, text)
                    ret.update(text_data)
                    
                    # 文本数据增强
                    if self.split == 'train':
                        n_text_augments = self.draw_false_text
                        if hasattr(self, 'labels'):
                            if self.labels[index] == (0 if self.pos_count > self.neg_count else 1):
                                n_text_augments = min(n_text_augments * 2, 5)
                        
                        for i in range(n_text_augments):
                            try:
                                if hasattr(self, 'labels'):
                                    candidate_indices = self.pos_indices if self.labels[index] == 1 else self.neg_indices
                                    random_index = random.choice(candidate_indices)
                                else:
                                    random_index = random.randint(0, len(self.keys) - 1)
                                
                                txt_path_f = self.metadata_dir+'text/'+self.keys[random_index]+'.txt'
                                with open(txt_path_f, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                match = re.search(pattern, content)
                                if match:
                                    false_text = self._clean_text(match.group(1))
                                    ret.update(self._get_false_text(i, false_text))
                            except Exception as e:
                                print(f"Warning: Failed to load false text {i}: {str(e)}")
                                continue
                else:
                    return None
            except Exception:
                return None
        
        # 4. 添加标签和权重
        score = float(self.labels_score[self.labels_score['FileName']==self.keys[index]]['sentiment_score'])
        label2 = a2_parse(score)
        ret.update({
            "label2": label2,
            "score": score,
            "index": index
        })
        
        # 添加样本权重（如果在训练集中）
        if self.split == 'train' and hasattr(self, 'sample_weights'):
            ret["sample_weight"] = self.sample_weights[index]
        
        return ret

    def get_sampler(self, batch_size):
        """获取数据采样器，支持平衡采样
        Args:
            batch_size: 批次大小
        Returns:
            sampler: 数据采样器
        """
        if self.split == 'train' and self.use_balanced_sampler:
            return BalancedBatchSampler(self, batch_size)
        return torch.utils.data.RandomSampler(self)

    def _clean_text(self, text):
        """清理和规范化文本"""
        if not text:
            return ""
            
        # 基本清理
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.lower()  # 转小写
        
        # 移除特殊字符，但保留基本标点
        text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
        
        return text

    def _load_metadata(self):
        """加载元数据并计算类别统计"""
        if self.split == 'train':
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_train.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'train/'
            
            # 计算类别统计信息
            scores = self.labels_score['sentiment_score'].values
            self.labels = [a2_parse(score) for score in scores]
            self.labels2 = [a2_parse(score) for score in scores]
            
            # 计算正负样本索引和数量
            self.pos_indices = [i for i, label in enumerate(self.labels) if label == 1]
            self.neg_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.pos_count = len(self.pos_indices)
            self.neg_count = len(self.neg_indices)
            
            # 计算样本权重
            total = self.pos_count + self.neg_count
            self.pos_weight = total / (2 * self.pos_count) if self.pos_count > 0 else 1.0
            self.neg_weight = total / (2 * self.neg_count) if self.neg_count > 0 else 1.0
            self.sample_weights = [self.pos_weight if label == 1 else self.neg_weight for label in self.labels]
            
            print(f"\nDataset Statistics:")
            print(f"Total samples: {total}")
            print(f"Positive samples: {self.pos_count}, Negative samples: {self.neg_count}")
            print(f"Pos/Neg ratio: {self.pos_count/self.neg_count:.2f}")
            print(f"Sample weights - Positive: {self.pos_weight:.2f}, Negative: {self.neg_weight:.2f}")
            
        elif self.split == 'val':
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_valid.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'valid/'
            scores = self.labels_score['sentiment_score'].values
            self.labels = [a2_parse(score) for score in scores]
            self.labels2 = [a2_parse(score) for score in scores]
            
        elif self.split == 'test':
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_test.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'test/'
            scores = self.labels_score['sentiment_score'].values
            self.labels = [a2_parse(score) for score in scores]
            self.labels2 = [a2_parse(score) for score in scores]

    def __len__(self):
        return len(self.keys)