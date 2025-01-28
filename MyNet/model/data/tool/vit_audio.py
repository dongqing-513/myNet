import os
import torch
import torchaudio
import librosa
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from decord import VideoReader
from scipy.ndimage import zoom

def read_video_frame(video_path):
    pass

def crop_center(image, crop_size=224):
    """
    Center crop an image to specified size
    
    Args:
        image (numpy.ndarray): Input image
        crop_size (int): Size to crop to (default 224)
    
    Returns:
        numpy.ndarray: Center cropped image
    """
    h, w = image.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    
    return image[start_h:start_h+crop_size, start_w:start_w+crop_size]

def split_image_to_patches(image, patch_size=74):
    """
    Split an image into patches
    
    Args:
        image (numpy.ndarray): Input image
        patch_size (int): Size of each patch (default 16)
    
    Returns:
        list: List of image patches
    """
    # Ensure image is in correct format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Resize to ensure divisibility
    image = image.resize((224, 224))
    
    patches = []
    for i in range(0, 224, patch_size):
        for j in range(0, 224, patch_size):
            patch = image.crop((j, i, j+patch_size, i+patch_size))
            patches.append(patch)
    
    return patches

def process_video_to_patches(video_path, output_dir=''):
    """
    Process a video into patches
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save patches
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read frame
    frame = read_video_frame(video_path)
    
    # Crop frame
    cropped_frame = crop_center(frame)
    
    # Split into patches
    patches = split_image_to_patches(cropped_frame)
    
    # Save patches
    for i, patch in enumerate(patches):
        patch.save(os.path.join(output_dir, f'patch_{i}.jpg'))
    
    print(f"Processed {len(patches)} patches from {video_path}")

def audio_to_spectrogram(audio_path):
    """
    Convert audio file to 224x224 mel spectrogram
    """
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=224,
        normalized=True
    )
    
    # Generate mel spectrogram
    mel_spect = mel_transform(waveform)
    
    # Convert to decibels
    mel_spect_db = torchaudio.transforms.AmplitudeToDB()(mel_spect)
    
    # Convert to numpy array and remove batch dimension
    mel_spect_db = mel_spect_db.squeeze(0).numpy()
    
    # 确保频谱图大小为224x224
    if mel_spect_db.shape[1] != 224:
        # 如果长度不是224，则调整大小
        scale_factor = 224 / mel_spect_db.shape[1]
        mel_spect_db = zoom(mel_spect_db, [1, scale_factor])
    
    # 如果高度不是224，也进行调整
    if mel_spect_db.shape[0] != 224:
        scale_factor = 224 / mel_spect_db.shape[0]
        mel_spect_db = zoom(mel_spect_db, [scale_factor, 1])
    
    print(f"Final spectrogram shape after resizing: {mel_spect_db.shape}")
    
    # Normalize to [0, 1] range
    mel_spect_db = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())
    
    return mel_spect_db

def split_spectrogram_to_patches(spectrogram, patch_size=74):
    """
    Split spectrogram into patches of size patch_size x patch_size with overlap
    """
    height, width = spectrogram.shape
    patches = []
    positions = []
    
    print(f"Original spectrogram shape: {height}x{width}")
    print(f"Patch size: {patch_size}x{patch_size}")
    
    # 计算步长，使得能够得到3x3=9个patch
    stride_h = (height - patch_size) // 2
    stride_w = (width - patch_size) // 2
    
    # 生成3x3的网格
    for i in range(3):
        for j in range(3):
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = start_h + patch_size
            end_w = start_w + patch_size
            
            print(f"Extracting patch at position ({start_h}:{end_h}, {start_w}:{end_w})")
            patch = spectrogram[start_h:end_h, start_w:end_w]
            print(f"Patch shape: {patch.shape}")
            
            patches.append(patch)
            positions.append((start_h, start_w))
    
    print(f"Total patches generated: {len(patches)}")
    return patches, positions

def process_audio_file(audio_path, output_dir):
    """
    Process a single audio file:
    1. Convert to spectrogram
    2. Split into patches
    3. Save patches
    """
    # Create output directory if it doesn't exist
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Convert to spectrogram
    spectrogram = audio_to_spectrogram(audio_path)
    
    # Save full spectrogram
    plt.figure(figsize=(10, 10))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mel_spectrogram.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Split into patches and save them
    patches, positions = split_spectrogram_to_patches(spectrogram, patch_size=75)
    
    print(f"Splitting spectrogram into patches...")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Number of patches: {len(patches)}")
    
    for idx, (patch, pos) in enumerate(zip(patches, positions)):
        patch_filename = os.path.join(patches_dir, f'patch_{idx:02d}_pos_{pos[0]:03d}_{pos[1]:03d}.png')
        
        plt.figure(figsize=(3, 3))
        plt.imshow(patch, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(patch_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved patch {idx+1}/{len(patches)}: {patch_filename}")
    
    return spectrogram, patches

def main():
    # Define paths
    audio_dir = "/home/mz/demo/TVLT/Dataset/cmumosei/train/audio"
    output_dir = "/home/mz/demo/MyNet/model/data/tool/audio_patches"
    
    # Get first audio file
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
    if not audio_files:
        print("No audio files found in the directory")
        return
    
    first_audio = os.path.join(audio_dir, audio_files[0])
    print(f"Processing audio file: {first_audio}")
    
    # Process the audio file
    spectrogram, patches = process_audio_file(first_audio, output_dir)

if __name__ == "__main__":
    main()