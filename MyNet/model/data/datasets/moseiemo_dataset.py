
import random
import torch
import io
import os
import glob
from tqdm import tqdm
import json
import re

import pandas as pd
import numpy as np
from model.data.datasets.rawvideo_utils import RawVideoExtractor
from .base_video_dataset import BaseVideoDataset


class MOSEIEMODataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split

        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__(*args, **kwargs,)        
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
   
    def _load_metadata(self):
        if self.split=='train':
                
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_train.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'train/'       
            
        elif self.split=='val':
            
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_valid.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'valid/'
            
        elif self.split=='test':
             
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_test.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'test/'           
       

    # 从给定的数据源中提取和整合不同类型的数据，包括视频、音频和文本信息，并生成一个包含这些数据以及相关情感列表的字典对象
    def get_suite(self, index):
        result = None
        video_path = self.metadata_dir+'video/'+self.keys[index]+'.mp4'
        # audio_wav 或者 audio?
        audio_path = self.metadata_dir+'audio/'+self.keys[index]+'.wav'
        txt_path = self.metadata_dir+'text/'+self.keys[index]+'.txt'
        
        ret = dict()  
        # 调用_get_video方法获取视频数据，并将结果更新到ret字典中。          
        ret.update(self._get_video(index, video_path, rand_sample=True))
        
        # # 获取样本中的音頻内容(如果需要)，并调用self._get_audio(index, audio_path)获取音頻数据，更新到ret字典中
        if self.use_audio:
            ret.update(self._get_audio(index, audio_path)) 

        # 获取样本中的文本内容(如果需要)，并调用self._get_text(index, text)获取文本数据，更新到ret字典中
        if self.use_text:
            text = ''
            with open(txt_path, 'r') as file:
                content = file.read()
                pattern = r"<s>(.*?)</s>"
                match = re.search(pattern, content)
                if match:
                    text = match.group(1)
                    
            ret.update(self._get_text(index, text))  
            # 进入循环                         
            for i in range(self.draw_false_text):
                # 随机生成索引
                random_index = random.randint(0, len(self.index_mapper) - 1)
                # 获取随机索引对应的样本内容
                sample_f = self.metadata[self.keys[random_index]]
                text_f = sample_f['text']
                # 获取假视频数据，更新到ret字典中
                ret.update(self._get_false_text(i, text_f))
        # 进入循环
        for i in range(self.draw_false_video):
            # 随机生成索引
            random_index = random.randint(0, len(self.index_mapper) - 1)
            # 获取随机索引对应的路径
            video_path_f = self.metadata_dir+'video/'+self.keys[random_index]+'.mp4'
            # 获取假视频数据，更新到ret字典中。
            ret.update(self._get_false_video(i, video_path_f, rand_sample=True))

        # 从self.labels数据中筛选出特定行的数据，并将其第 3 列及之后的列的值大于 0.0 的结果转换为 NumPy 数组，存储在emolist中
        emolist = np.array(self.labels[self.labels['FileName']==self.keys[index]].iloc[0, 3:] > 0.0)
        # 最后，将emolist更新到ret字典中
        ret.update({"emolist": emolist})

        return ret
