"""
MANTA Dataset Components
=========================

Provides video data loading and episodic sampling for few-shot action recognition.

Modules:
    - video_dataset.py:   VideoDataset class for loading pre-extracted frames
    - episode_sampler.py: EpisodicSampler for N-way K-shot task construction
"""

from datasets.video_dataset import VideoDataset
from datasets.episode_sampler import EpisodicSampler

__all__ = ['VideoDataset', 'EpisodicSampler']
