import math
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer

from PIL import Image
import torchaudio
from decord import VideoReader, AudioReader
from decord import cpu, gpu
import librosa
import audiosegment
from moviepy.editor import AudioFileClip
import ffmpeg


def time_to_indices(video_reader, time):
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)

def preprocess_audio(audio, sr, target_shape=(1, 224, 224)):
    """
    将音频数据处理为指定维度的梅尔频谱图。

    参数:
    audio: 输入的音频数据
    sr: 音频的采样率
    target_shape: 目标频谱图形状，支持(1, 224, 224)和(1, 464, 128)

    返回:
    torch.Tensor: 形状为target_shape的梅尔频谱图
    """
    # 对音频进行均值归一化处理
    audio = audio - audio.mean()
    
    # 根据目标形状选择合适的参数
    if target_shape == (1, 224, 224):
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        n_mels = 224
    else:  # (1, 464, 128)
        n_fft = 1024
        hop_length = 220
        win_length = 1024
        n_mels = 128
    
    # 生成梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0
    )
    
    # 转换为分贝刻度并归一化
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min()) * 2 - 1
    
    # 转换为PyTorch张量
    mel_spec = torch.from_numpy(mel_spec).float()
    
    if target_shape == (1, 224, 224):
        # 先调整为(224, 224)
        mel_spec = F.interpolate(
            mel_spec.unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        # 添加batch维度
        mel_spec = mel_spec.unsqueeze(0)
    else:  # (1, 464, 128)
        # 调整时间维度到464
        mel_spec = mel_spec.transpose(0, 1)  # (freq, time) -> (time, freq)
        if mel_spec.shape[0] < 464:
            # 如果太短，进行填充
            pad_size = 464 - mel_spec.shape[0]
            mel_spec = F.pad(mel_spec, (0, 0, 0, pad_size), "constant", 0)
        elif mel_spec.shape[0] > 464:
            # 如果太长，进行裁剪
            mel_spec = mel_spec[:464, :]
        # 添加batch维度
        mel_spec = mel_spec.unsqueeze(0)
    
    # 确保输出形状正确
    assert mel_spec.shape == target_shape, \
        f"Expected shape {target_shape}, got {mel_spec.shape}"
    
    return mel_spec

def crop_image_only_outside(img, tol=30.0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    mask = mask.all(0).all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[:, row_start:row_end,col_start:col_end]


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[2]
    height = img.shape[1]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    if len(img.shape) == 3:
        center_cropped_img = img[:, top:bottom, left:right]
    else:
        center_cropped_img = img[:, top:bottom, left:right, ...]

    return center_cropped_img


import librosa
import torch
import torch.nn.functional as F


def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=Image.BICUBIC),])


class RawVideoExtractor():
    
    def __init__(self, centercrop=True, audio_size=464, video_size=224, framerate=1, num_frames=8):
        self.centercrop = centercrop
        self.audio_size = audio_size
        self.video_size = video_size
        self.framerate = framerate
        self.max_frames = num_frames
        self.transform_video = self._transform(self.video_size)
        self.sr = 44100
        self.print_error = False
        if not self.print_error:
            import warnings
            warnings.filterwarnings("ignore")
        
    def _transform(self, n_px):
        # 图像尺寸调整 resize,将图像调整为边长为n_px的正方形大小，使用双三次插值（Image.BICUBIC）算法进行插值计算，以获得更平滑的图像缩放效果。
        return Compose([
            Resize([n_px, n_px], interpolation=Image.BICUBIC),])

    def _transform_audio(self, n_px):
        return Normalize(mean=[0.5], std=[0.5])  

    def audio_to_tensor(self, path, timestamp=None):
        """
        将音频文件转换为张量。
        
        参数:
        path: 音频文件路径
        timestamp: 时间戳

        返回:
        torch.Tensor: 形状为(1, 224, 224)或(1, 464, 128)的梅尔频谱图
        """
        # 根据audio_size确定目标形状
        target_shape = (1, 224, 224) #(1, 464, 128)

        try:
            if path.endswith(('mp3', 'wav', 'flac')):
                # 加载音频文件
                audio, org_sr = torchaudio.load(path)
                if org_sr != self.sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
                audio = audio.mean(0).numpy()
                
                if timestamp is not None:
                    start, end = int(self.sr * timestamp[0]), int(self.sr * timestamp[1])
                    audio = audio[start: end]
                
                # 生成梅尔频谱图
                audio = preprocess_audio(audio, sr=self.sr, target_shape=target_shape)

            elif path.endswith(('avi', 'mp4')):
                audio = AudioFileClip(path)
                org_sr = audio.fps
                if timestamp is not None:
                    audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=org_sr).mean(1)
                else:
                    audio = audio.to_soundarray(fps=org_sr).mean(1)
                
                if org_sr != self.sr:
                    audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=org_sr, new_freq=self.sr).numpy()
                
                # 生成梅尔频谱图
                audio = preprocess_audio(audio, sr=self.sr, target_shape=target_shape)

            else:
                # 处理其他类型的文件（如图像文件）
                if path.endswith('jpg'):
                    audio = np.array(Image.open(path))/255.0
                    audio = torch.from_numpy(audio*2.0-1.0).unsqueeze(0)
                    if target_shape == (1, 224, 224):
                        audio = audio[:, :, :224].transpose(1,2)
                        audio = F.interpolate(audio.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        audio = audio[:, :, :464].transpose(1,2)
                        audio = torch.cat([audio, torch.ones_like(audio[:, :, :32])*-0.794754], -1)
                else:
                    start, end = int(self.sr * timestamp[0]/511.99058), int(self.sr * timestamp[1]/511.99058)
                    index1, index2 = int(start // 65500), int(end // 65500)
                    
                    if index1 == index2:
                        audio = np.array(Image.open(f'{path}_num{str(index1)}.jpg'))/255.0
                        audio = audio[:, int(start%65500): int(end%65500)]
                    else:
                        audio = np.array(Image.open(f'{path}_num{str(index1)}.jpg'))/255.0
                        audio_ = np.array(Image.open(f'{path}_num{str(index2)}.jpg'))/255.0
                        audio = np.concatenate([audio[:, int(start%65500):], audio_[:, :int(end%65500)]], -1)
                    
                    audio = torch.from_numpy(audio*2.0-1.0).unsqueeze(0)
                    if target_shape == (1, 224, 224):
                        audio = audio[:, :, :224].transpose(1,2)
                        audio = F.interpolate(audio.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        audio = audio[:, :, :464].transpose(1,2)
                        audio = torch.cat([audio, torch.ones_like(audio[:, :, :32])*-0.794754], -1)
        
        except Exception as e:
            if self.print_error:
                print(e)
            # 错误情况下返回全-1的张量
            audio = torch.ones(target_shape) * -1
        
        # 确保输出形状正确
        assert audio.shape == target_shape, \
            f"Expected shape {target_shape}, got {audio.shape}"
        
        return audio
    
    # 将给定的视频文件路径转换为张量表示，同时支持随机采样和指定时间戳的功能，以及进行一些视频预处理操作    
    def video_to_tensor(self, path, timestamp=None, get_video=True, get_audio=True, rand_sample=False):

        try:
            # 获取视频文件，获取视频对象
            video = VideoReader(path)
            # 获取视频的平均帧率
            framerate = video.get_avg_fps()
            # 计算视频的时长
            video_len = len(video)/framerate
            
            # 随机采样处理：
            if rand_sample and timestamp is None:
                # 设置视频剪辑长度为 15.0 秒。
                video_clip_length = 15.0
                # 如果视频时长大于等于剪辑长度
                if video_len >= video_clip_length:
                    # 随机生成一个起始时间start，范围是在视频时长减去剪辑长度的范围内
                    start = (np.random.rand(1)*(video_len-(video_clip_length-0.1)))[0]
                    # 设置时间戳
                    timestamp = [start, start + (video_clip_length-0.1)]
                # 如果视频时长小于剪辑长度
                else:
                    # 设置时间戳
                    timestamp = [0, video_len-0.1]

            # 指定时间戳
            if timestamp is not None:
                # 将时间戳转换为视频帧的索引范围
                start, end = time_to_indices(video, timestamp)
                # 对索引范围进行限制，确保起始索引和结束索引在合法范围内
                end = min(len(video)-1, end)
                start = min(start, end-1)
                # 使用线性插值生成采样索引，用于从视频中抽取特定的帧
                downsamlp_indices = np.linspace(start, end, self.max_frames, endpoint=False).astype(np.int)

            else: 
                # 直接线性插值生成从视频开头到结尾的采样索引                      
                downsamlp_indices = np.linspace(0, len(video), self.max_frames, endpoint=False).astype(np.int)

            # 根据采样索引从视频中获取特定的帧，并转换为 NumPy 数组。
            video = video.get_batch(downsamlp_indices).asnumpy()
            # 对视频帧进行裁剪，去除外部的部分
            video = crop_image_only_outside(video)
            # 找到视频帧的最小边长
            min_shape = min(video.shape[1:3])
            # 对视频帧进行中心裁剪，裁剪为边长为min_shape的正方形。
            video = center_crop(video, min_shape, min_shape)
            # 将 NumPy 数组转换为 PyTorch 张量，并调整维度顺序。
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            # 图像尺寸调整 resize,将图像调整为边长为n_px的正方形大小，使用双三次插值（Image.BICUBIC）算法进行插值计算，以获得更平滑的图像缩放效果。
            video = self.transform_video(video)
            # 归一化
            video = (video/255.0-0.5)/0.5
        except Exception as e:
            if self.print_error:
                print(e)
            video = torch.ones([self.max_frames, 3, self.video_size, self.video_size]) * -1

        return video
    
    
    def video_audio_to_tensor(self, path, timestamp=None, rand_sample=False):

        try:
            video = VideoReader(path)
            framerate = video.get_avg_fps()
            video_len = len(video)/video.get_avg_fps()
            
            if rand_sample and timestamp is None:
                video_clip_length = 15.0
                if video_len >= video_clip_length:
                    start = (np.random.rand(1)*(video_len-(video_clip_length-0.1)))[0]
                    timestamp = [start, start + (video_clip_length-0.1)]
                else:
                    timestamp = [0, video_len-0.1]

            if timestamp is not None:
                start, end = time_to_indices(video, timestamp)            
                end = min(len(video)-1, end)
                start = min(start, end-1)
                downsamlp_indices = np.linspace(start, end, self.max_frames, endpoint=False).astype(np.int)
            else:            
                downsamlp_indices = np.linspace(0, len(video), self.max_frames, endpoint=False).astype(np.int)

            video = video.get_batch(downsamlp_indices).asnumpy()
            video = crop_image_only_outside(video)
            min_shape = min(video.shape[1:3])
            video = center_crop(video, min_shape, min_shape)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            video = self.transform_video(video)       
            video = (video/255.0-0.5)/0.5

            audio = AudioFileClip(path)   
            sr = audio.fps
            if timestamp is not None:
                audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=sr).mean(1)
            else:
                audio = audio.to_soundarray(fps=sr).mean(1)
            audio = preprocess_audio(audio, sr=sr, target_shape=(1, self.audio_size, 128))[:, :self.audio_size]
        except Exception as e:
            print(e)
            audio = torch.zeros([1, 16, 128])
            video = torch.zeros([self.max_frames, 3, self.video_size, self.video_size])
        return video, audio



def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=Image.BICUBIC),])
    

def load_audio(path, sr=44100, timestamp=None):
    audio, org_sr = torchaudio.load(path)
    if org_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=sr)
    audio = audio.mean(0).numpy()      
    if timestamp is not None:
        start, end = int(sr * timestamp[0]), int(sr * timestamp[1])
        audio = audio[start: end]
    audio = preprocess_audio(audio, sr=sr, target_shape=(1, 1024, 128))
    audio = audio[:, :1024]
    return audio.unsqueeze(0).float()

    

def load_video(path, num_frames=8, timestamp=None):
    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/framerate

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)
        end = min(len(video)-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)

    else:                       
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(224)(video)
    video = (video/255.0-0.5)/0.5
    return video.unsqueeze(0).float()

    
def load_video_raw(path, num_frames=8, timestamp=None):
    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/framerate

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)
        end = min(len(video)-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)

    else:                       
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    return video
    
    
def load_video_audio(path, num_frames=8, sr=44100, timestamp=None):
    # 没用到这个函数
    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/video.get_avg_fps()

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)            
        end = min(len(video)-1, end)
        start = min(start, end-1)

        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)
    else:            
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(224)(video)       
    video = (video/255.0-0.5)/0.5

    # 音频文件读取与加载，并获取基本属性
    audio = AudioFileClip(path)   
    org_sr = audio.fps
    
    if timestamp is not None:
        audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=org_sr).mean(1)
    else:
        audio = audio.to_soundarray(fps=sr).mean(1)
    if org_sr != sr:
        audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=org_sr, new_freq=sr).numpy()
    audio = preprocess_audio(audio, sr=sr, target_shape=(1, 224, 224))
    return video.unsqueeze(0).float(), audio.unsqueeze(0).float()
    

def load_text(path):
    # BertTokenizer是与这个特定模型（bert-base-uncased）配套使用的分词工具。
    # 它知道如何将自然语言文本分割成适合输入到 “bert-base-uncased” 模型的格式，
    # 包括如何处理单词、标点符号，以及如何生成特殊标记（如 [CLS]、[SEP] 等
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with open(path) as f:
        text = f.readline()
    encoding = tokenizer(
                text,
                # 如果文本长度小于指定的最大长度（这里是 40），则在文本末尾添加填充标记，使长度达到最大长度。
                padding="max_length",
                # 如果文本长度超过最大长度，则截断文本
                truncation=True,
                max_length=40,
                # 返回一个特殊标记掩码，如填充标记、分隔标记等。
                return_special_tokens_mask=True,
            )
    # input_ids：这是一个整数列表，表示文本经过分词后每个词对应的在词汇表中的索引。例如：[101, 2023, 2003, 1996, 2171, 3231, 102]，其中 101 和 102 通常是特殊标记 [CLS] 和 [SEP]
    # attention_mask：这是一个与input_ids长度相同的整数列表，用于指示哪些位置的输入是有效的（值为 1），哪些位置是填充的（值为 0）。例如：[1, 1, 1, 1, 1, 1, 1] 表示所有位置都是有效的输入。
    # token_type_ids（可选）：在一些任务中，用于区分不同的文本片段，例如在句子对任务中区分两个句子。通常也是一个整数列表。
    
    # 返回原始文本、编码后的输入 ID 张量（通过将字典中的 input_ids 转换为张量并添加一个维度）以及注意力掩码张量（同样转换并添加一个维度）。
    # 预训练模型在其预训练阶段已经学习了大量的语言知识，但对于新的特定任务或新的数据集，通常还需要进行进一步的微调（fine-tuning）。在微调过程中，模型会根据新的任务和数据调整其参数，以更好地适应特定的问题。
    return text, torch.tensor(encoding['input_ids']).unsqueeze(0), torch.tensor(encoding['attention_mask']).unsqueeze(0), 

def load_image(path, image_size=224):
    
    image = np.array(Image.open(path).convert("RGB").resize((image_size, image_size), Image.ANTIALIAS))
    image = image/255.0*2.0-1.0
    image = torch.tensor(image[np.newaxis, ...]).permute(0, 3, 1, 2)
    return image    
