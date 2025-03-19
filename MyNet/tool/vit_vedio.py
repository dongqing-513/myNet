import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
import numpy as np

def read_video_frame(video_path):
    """
    Read a single frame from a video file
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        numpy.ndarray: A single video frame
    """
    # Read the video
    video = VideoReader(video_path)
    
    # Select the middle frame
    mid_frame_index = len(video) // 2
    frame = video[mid_frame_index].asnumpy()
    
    return frame

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

def split_image_to_patches(image, patch_size=75):
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

def process_video_to_patches(video_path, output_dir='patches'):
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

# Example usage
if __name__ == '__main__':
    # Select the first video from the directory
    video_dir = '/home/mz/demo/TVLT/Dataset/cmumosei/train/video'
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if video_files:
        video_path = os.path.join(video_dir, video_files[0])
        process_video_to_patches(video_path)
    else:
        print("No video files found in the directory.")