"""
MANTA Evaluation Script — Testing with Confidence Intervals.

Paper Section 4.1 — Evaluation Protocol:
    "We evaluate on 10,000 randomly sampled test episodes and report
     the mean accuracy with 95% confidence interval."

Usage:
    python test.py --config configs/kinetics_resnet50.yaml \
                   --checkpoint checkpoints/kinetics_rn50_f16/best_model.pth

    python test.py --config configs/ssv2_resnet50.yaml \
                   --checkpoint checkpoints/ssv2_rn50_f16/best_model.pth \
                   --num_tasks 10000
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from models.manta import Manta
from datasets.video_dataset import VideoDataset
from datasets.episode_sampler import EpisodicSampler
from utils.metrics import compute_accuracy, confidence_interval


def test(
    cfg,
    checkpoint_path: str,
    num_tasks: int = 10000,
    save_results: bool = True,
):
    """
    Evaluate MANTA model on test set with confidence intervals.

    Args:
        cfg: Configuration object
        checkpoint_path: Path to trained model checkpoint
        num_tasks: Number of test episodes (default 10,000 per paper)
        save_results: Whether to save results to JSON file
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ==================== Build Model ====================
    model = Manta(
        backbone_name=cfg.model.backbone,
        d_model=cfg.model.d_model,
        scales=list(cfg.model.scales),
        d_state=cfg.model.d_state,
        d_conv=cfg.model.get('d_conv', 4),
        expand=cfg.model.get('expand', 2),
        temperature=cfg.model.temperature,
        lambda_ce=cfg.model.lambda_ce,
        pretrained_backbone=False,  # Will load from checkpoint
        freeze_backbone=True,
    ).to(device)

    # ==================== Load Checkpoint ====================
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Trained for {checkpoint.get('task_idx', 'unknown')} tasks")
        print(f"  Best val acc: {checkpoint.get('best_val_acc', 'unknown')}")
    else:
        # Direct state dict (no wrapper)
        model.load_state_dict(checkpoint)

    model.eval()

    # ==================== Load Test Dataset ====================
    test_dataset = VideoDataset(
        split_file=os.path.join(cfg.data.split_dir, 'test.txt'),
        num_frames=cfg.data.num_frames,
        image_size=cfg.data.image_size,
        resize_size=cfg.data.resize_size,
        use_flip=False,
        mode='test',
    )
    print(f"Test set: {test_dataset.get_num_classes()} classes, {len(test_dataset)} videos")

    # ==================== Evaluation Loop ====================
    print(f"\nEvaluating {num_tasks} episodes...")
    print(f"Setting: {cfg.episode.n_way}-way {cfg.episode.k_shot}-shot")
    print(f"Frames per video: F={cfg.data.num_frames}")
    print(f"Scales: O={list(cfg.model.scales)}")
    print("=" * 60)

    episode_accuracies = []
    episode_distances = []

    with torch.no_grad():
        for task_idx in tqdm(range(num_tasks), desc="Testing", ncols=80):
            # Sample test episode
            support, query, sup_labels, que_labels = EpisodicSampler.sample_episode(
                dataset=test_dataset,
                n_way=cfg.episode.n_way,
                k_shot=cfg.episode.k_shot,
                n_query=cfg.episode.num_query,
                device=device,
            )

            # Forward pass (test mode — no loss computation)
            result = model(
                support_videos=support,
                query_videos=query,
                support_labels=sup_labels,
                query_labels=que_labels,
                n_way=cfg.episode.n_way,
                k_shot=cfg.episode.k_shot,
                mode='test',
            )

            # Compute accuracy
            acc = compute_accuracy(result['pred_labels'], que_labels)
            episode_accuracies.append(acc)

            # Track mean distance (for analysis)
            mean_dist = result['distances'].mean().item()
            episode_distances.append(mean_dist)

    # ==================== Compute Final Results ====================
    mean_acc, ci = confidence_interval(episode_accuracies, confidence=0.95)

    # Print results
    print("\n" + "=" * 60)
    print("                    TEST RESULTS")
    print("=" * 60)
    print(f"  Dataset:     {cfg.data.dataset}")
    print(f"  Backbone:    {cfg.model.backbone}")
    print(f"  Setting:     {cfg.episode.n_way}-way {cfg.episode.k_shot}-shot")
    print(f"  Frames (F):  {cfg.data.num_frames}")
    print(f"  Scales (O):  {list(cfg.model.scales)}")
    print(f"  Episodes:    {num_tasks}")
    print(f"  λ (lambda):  {cfg.model.lambda_ce}")
    print(f"  τ (temp):    {cfg.model.temperature}")
    print("-" * 60)
    print(f"  Accuracy:    {mean_acc*100:.2f}% ± {ci*100:.2f}%")
    print(f"  Mean dist:   {np.mean(episode_distances):.4f}")
    print("=" * 60)

    # ==================== Save Results ====================
    if save_results:
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint_path,
            'config': {
                'dataset': cfg.data.dataset,
                'backbone': cfg.model.backbone,
                'n_way': cfg.episode.n_way,
                'k_shot': cfg.episode.k_shot,
                'num_frames': cfg.data.num_frames,
                'scales': list(cfg.model.scales),
                'lambda_ce': cfg.model.lambda_ce,
                'temperature': cfg.model.temperature,
            },
            'results': {
                'mean_accuracy': mean_acc,
                'confidence_interval_95': ci,
                'accuracy_percent': f"{mean_acc*100:.2f}% ± {ci*100:.2f}%",
                'num_episodes': num_tasks,
                'mean_distance': float(np.mean(episode_distances)),
                'std_accuracy': float(np.std(episode_accuracies)),
            },
            'per_episode_accuracies': episode_accuracies,
        }

        # Save to JSON
        results_dir = os.path.join(cfg.output.save_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir,
            f"test_{cfg.data.dataset}_{cfg.episode.n_way}way_{cfg.episode.k_shot}shot_f{cfg.data.num_frames}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return mean_acc, ci


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MANTA Testing')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--num_tasks', type=int, default=10000,
        help='Number of test episodes (default: 10000)'
    )
    parser.add_argument(
        '--no_save', action='store_true',
        help='Do not save results to file'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    test(cfg, args.checkpoint, args.num_tasks, save_results=not args.no_save)
