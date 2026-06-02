"""
Episodic Sampler for N-way K-shot Few-Shot Action Recognition.

Paper Section 4.1:
    "We follow the standard episodic training paradigm. Each training episode
     (task) consists of:
        - Support set S: N classes × K shots = N*K videos
        - Query set Q: N classes × n_query queries = N*n_query videos
     
     Classes in each episode are randomly sampled from the training split.
     Support and query videos within each class are disjoint."

Episode Structure:
    Given N-way K-shot with n_query queries per class:
        Support: [N, K, F, C, H, W] — N classes, K videos each, F frames
        Query:   [N*n_query, F, C, H, W] — n_query videos per class
        Support labels: [N*K] — class indices (0 to N-1)
        Query labels:   [N*n_query] — class indices (0 to N-1)
"""

import random
from typing import Tuple

import torch

from datasets.video_dataset import VideoDataset


class EpisodicSampler:
    """
    Sampler that constructs N-way K-shot episodes from a VideoDataset.
    
    Each episode randomly selects N classes and samples K+n_query videos
    per class (K for support, n_query for query).
    
    The sampler ensures:
        1. Classes are randomly selected from available classes
        2. Support and query videos are DISJOINT within each class
        3. Labels are relative (0 to N-1) within each episode
    """

    @staticmethod
    def sample_episode(
        dataset: VideoDataset,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 5,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a single N-way K-shot episode from the dataset.

        Args:
            dataset: VideoDataset instance containing the data split
            n_way: Number of classes N per episode
            k_shot: Number of support shots K per class
            n_query: Number of query samples per class
            device: Target device for tensors (default: CPU)

        Returns:
            tuple of:
                support_videos: [N, K, F, C, H, W] — support set
                query_videos:   [N*n_query, F, C, H, W] — query set
                support_labels: [N*K] — support class indices (0..N-1)
                query_labels:   [N*n_query] — query class indices (0..N-1)
        """
        if device is None:
            device = torch.device('cpu')

        # Step 1: Randomly select N classes
        available_classes = dataset.classes
        assert len(available_classes) >= n_way, (
            f"Dataset has {len(available_classes)} classes but episode needs {n_way}"
        )
        selected_classes = random.sample(available_classes, n_way)

        support_videos = []
        query_videos = []
        support_labels = []
        query_labels = []

        for class_idx, class_name in enumerate(selected_classes):
            # Step 2: Get all videos for this class
            class_video_paths = dataset.get_class_videos(class_name)
            total_needed = k_shot + n_query

            assert len(class_video_paths) >= total_needed, (
                f"Class '{class_name}' has {len(class_video_paths)} videos "
                f"but needs {total_needed} (K={k_shot} + n_query={n_query})"
            )

            # Step 3: Randomly select K+n_query disjoint videos
            selected_videos = random.sample(class_video_paths, total_needed)

            # First K videos → support, remaining → query
            support_paths = selected_videos[:k_shot]
            query_paths = selected_videos[k_shot:]

            # Step 4: Load video frames
            for vpath in support_paths:
                video_tensor = dataset.load_video(vpath)  # [F, C, H, W]
                support_videos.append(video_tensor)
                support_labels.append(class_idx)

            for vpath in query_paths:
                video_tensor = dataset.load_video(vpath)  # [F, C, H, W]
                query_videos.append(video_tensor)
                query_labels.append(class_idx)

        # Step 5: Stack into tensors
        # Support: [N*K, F, C, H, W] → reshape to [N, K, F, C, H, W]
        support_tensor = torch.stack(support_videos, dim=0)  # [N*K, F, C, H, W]
        F, C, H, W = support_tensor.shape[1:]
        support_tensor = support_tensor.view(n_way, k_shot, F, C, H, W)

        # Query: [N*n_query, F, C, H, W]
        query_tensor = torch.stack(query_videos, dim=0)  # [N*n_query, F, C, H, W]

        # Labels
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)

        # Move to device
        return (
            support_tensor.to(device),
            query_tensor.to(device),
            support_labels_tensor.to(device),
            query_labels_tensor.to(device),
        )

    @staticmethod
    def sample_batch_episodes(
        dataset: VideoDataset,
        batch_size: int = 1,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 5,
        device: torch.device = None,
    ) -> list:
        """
        Sample a batch of episodes (for potential parallel processing).

        Args:
            dataset: VideoDataset instance
            batch_size: Number of episodes to sample
            n_way, k_shot, n_query: Episode configuration
            device: Target device

        Returns:
            List of (support, query, sup_labels, que_labels) tuples
        """
        episodes = []
        for _ in range(batch_size):
            episode = EpisodicSampler.sample_episode(
                dataset=dataset,
                n_way=n_way,
                k_shot=k_shot,
                n_query=n_query,
                device=device,
            )
            episodes.append(episode)
        return episodes


class EpisodeDataset(torch.utils.data.Dataset):
    """
    Wrapper to allow episodic sampling via PyTorch DataLoader for multiprocessing.
    """
    def __init__(
        self,
        dataset: VideoDataset,
        num_episodes: int,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 5,
    ):
        self.dataset = dataset
        self.num_episodes = num_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int):
        # Always return tensors on CPU to allow multiprocessing
        return EpisodicSampler.sample_episode(
            dataset=self.dataset,
            n_way=self.n_way,
            k_shot=self.k_shot,
            n_query=self.n_query,
            device=torch.device('cpu')
        )
