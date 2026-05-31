"""
Video Dataset for MANTA Few-Shot Action Recognition.

Paper Section 4.1 — Experimental Setup:
    "We uniformly sample F frames from each video to form sub-sequences.
     All frames are resized to 256×256 and randomly cropped to 224×224
     during training, with center crop during testing."

Frame Sampling Strategy:
    Given a video with T total frames and target F frames:
    - If T >= F: uniformly sample F frames (temporal stride = T/F)
    - If T < F:  repeat frames to reach F (loop padding)

Data Organization:
    split.txt format (one line per video):
        /path/to/video_frames_dir class_name
    
    Frame directory structure:
        video_dir/
            frame_000000.jpg
            frame_000001.jpg
            ...
"""

import os
import random
from typing import List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VideoDataset(Dataset):
    """
    Dataset for loading video frames organized by class directories.
    
    Supports the standard FSAR benchmark format where each video is stored
    as a directory of pre-extracted JPEG frames.
    
    Args:
        split_file: Path to split file (train.txt / val.txt / test.txt)
        num_frames: Number of frames F to sample per video (default 8)
        image_size: Crop size for frames (default 224)
        resize_size: Resize frames to this before cropping (default 256)
        use_flip: Whether to apply random horizontal flip (default True for training)
        mode: 'train' or 'test' (affects augmentation)
    """

    def __init__(
        self,
        split_file: str,
        num_frames: int = 8,
        image_size: int = 224,
        resize_size: int = 256,
        use_flip: bool = True,
        mode: str = 'train',
    ):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode

        # Parse split file
        self.samples: List[Tuple[str, str]] = []  # (video_path, class_name)
        self.class_to_videos: dict = defaultdict(list)

        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)  # Split from right to handle paths with spaces
                if len(parts) == 2:
                    video_path, class_name = parts
                    self.samples.append((video_path, class_name))
                    self.class_to_videos[class_name].append(video_path)

        self.classes = sorted(list(self.class_to_videos.keys()))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Data augmentation transforms
        if mode == 'train':
            transform_list = [
                transforms.Resize(resize_size),
                transforms.RandomCrop(image_size),
            ]
            if use_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            transform_list = [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def get_num_classes(self) -> int:
        """Return total number of classes in this split."""
        return len(self.classes)

    def get_class_videos(self, class_name: str) -> List[str]:
        """Get all video paths for a given class."""
        return self.class_to_videos[class_name]

    def _get_frame_paths(self, video_dir: str) -> List[str]:
        """
        Get sorted list of frame file paths in a video directory.
        
        Supports common frame naming conventions:
            frame_000000.jpg, img_00001.jpg, 0001.jpg, etc.
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        frames = []
        
        if not os.path.isdir(video_dir):
            return frames

        for fname in sorted(os.listdir(video_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_extensions:
                frames.append(os.path.join(video_dir, fname))

        return frames

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Uniformly sample F frame indices from a video with T total frames.
        
        Strategy (per paper):
            - If T >= F: uniform temporal sampling with stride T/F
            - If T < F:  loop/repeat frames to fill F positions
        
        Args:
            total_frames: Total number of available frames T
            
        Returns:
            List of F frame indices (0-indexed)
        """
        F = self.num_frames

        if total_frames >= F:
            # Uniform temporal sampling
            if self.mode == 'train':
                # During training: add random jitter within each segment
                segment_size = total_frames / F
                indices = []
                for i in range(F):
                    start = int(i * segment_size)
                    end = int((i + 1) * segment_size)
                    idx = random.randint(start, min(end - 1, total_frames - 1))
                    indices.append(idx)
            else:
                # During testing: deterministic center of each segment
                indices = np.linspace(0, total_frames - 1, F, dtype=int).tolist()
        else:
            # Loop padding: repeat frames to reach F
            indices = list(range(total_frames))
            while len(indices) < F:
                indices.extend(list(range(total_frames)))
            indices = indices[:F]

        return indices

    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess F frames from a video directory.
        
        Args:
            video_path: Path to directory containing video frames
            
        Returns:
            tensor: [F, C, H, W] — F preprocessed frames
        """
        frame_paths = self._get_frame_paths(video_path)

        if len(frame_paths) == 0:
            # Return zero tensor if no frames found (graceful fallback)
            return torch.zeros(self.num_frames, 3, 224, 224)

        # Sample frame indices
        indices = self._sample_frame_indices(len(frame_paths))

        # Load and transform frames
        frames = []
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert('RGB')
            img_tensor = self.transform(img)  # [C, H, W]
            frames.append(img_tensor)

        return torch.stack(frames, dim=0)  # [F, C, H, W]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single video sample.
        
        Args:
            index: Sample index
            
        Returns:
            tuple of (video_tensor [F, C, H, W], class_index int)
        """
        video_path, class_name = self.samples[index]
        class_idx = self.class_to_idx[class_name]
        video_tensor = self.load_video(video_path)
        return video_tensor, class_idx
