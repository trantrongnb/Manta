"""
MANTA Training Script — Episodic Training for Few-Shot Action Recognition.

Paper Section 4.1 — Training Details:
    - Episodic training: each iteration = one N-way K-shot task
    - Optimizer: SGD with momentum 0.9, weight decay 5e-4
    - Learning rate: 0.001 with cosine annealing
    - Number of training tasks: 10,000 (Kinetics/UCF/HMDB) or 75,000 (SSv2)
    - Validation: every 1,000 tasks
    - Backbone: frozen (pretrained on ImageNet)
    - Loss: L_total = λ × L_ce + L_hc (λ=4.0)

Usage:
    python train.py --config configs/kinetics_resnet50.yaml
    python train.py --config configs/ssv2_resnet50.yaml --resume checkpoints/latest.pth
"""

import os
import sys
import random
import argparse
from pathlib import Path

# === HOTFIX FOR RTX 5090 (sm_120) / MPS BYPASS ===
# Disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED crashes
import torch
torch.backends.cudnn.enabled = False

import numpy as np
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm

from models.manta import Manta
from datasets.video_dataset import VideoDataset
from datasets.episode_sampler import EpisodicSampler, EpisodeDataset
from torch.utils.data import DataLoader
from utils.metrics import compute_accuracy, confidence_interval
from utils.logger import setup_logger, TensorBoardLogger, Timer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg) -> Manta:
    """Build MANTA model from config."""
    model = Manta(
        backbone_name=cfg.model.backbone,
        d_model=cfg.model.d_model,
        scales=list(cfg.model.scales),
        d_state=cfg.model.d_state,
        d_conv=cfg.model.get('d_conv', 4),
        expand=cfg.model.get('expand', 2),
        temperature=cfg.model.temperature,
        lambda_ce=cfg.model.lambda_ce,
        pretrained_backbone=cfg.model.pretrained_backbone,
        freeze_backbone=cfg.model.get('freeze_backbone', True),
        backbone_weights_path=cfg.model.get('backbone_weights_path', None),
    )
    return model


def build_optimizer(model: Manta, cfg) -> tuple:
    """
    Build optimizer and scheduler.
    
    Only optimizes non-frozen parameters (backbone is frozen).
    """
    # Filter trainable parameters (exclude frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(
        trainable_params,
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_tasks,
        eta_min=cfg.training.get('min_lr', 1e-6),
    )

    return optimizer, scheduler


def evaluate(
    model: Manta,
    dataset: VideoDataset,
    cfg,
    device: torch.device,
    num_tasks: int = 1000,
) -> tuple:
    """
    Evaluate model on validation/test set.

    Args:
        model: Trained MANTA model
        dataset: Validation or test dataset
        cfg: Configuration
        device: Compute device
        num_tasks: Number of episodes to evaluate

    Returns:
        tuple of (mean_accuracy, confidence_interval)
    """
    model.eval()
    episode_accuracies = []

    val_episode_ds = EpisodeDataset(
        dataset=dataset,
        num_episodes=num_tasks,
        n_way=cfg.episode.n_way,
        k_shot=cfg.episode.k_shot,
        n_query=cfg.episode.num_query,
    )
    val_loader = DataLoader(
        val_episode_ds,
        batch_size=None,
        num_workers=cfg.training.get('num_workers', 4),
        pin_memory=True,
    )

    with torch.no_grad():
        for support, query, sup_labels, que_labels in val_loader:
            support = support.to(device, non_blocking=True)
            query = query.to(device, non_blocking=True)
            sup_labels = sup_labels.to(device, non_blocking=True)
            que_labels = que_labels.to(device, non_blocking=True)

            result = model(
                support_videos=support,
                query_videos=query,
                support_labels=sup_labels,
                query_labels=que_labels,
                n_way=cfg.episode.n_way,
                k_shot=cfg.episode.k_shot,
                mode='test',
            )

            acc = compute_accuracy(result['pred_labels'], que_labels)
            episode_accuracies.append(acc)

    mean_acc, ci = confidence_interval(episode_accuracies)
    return mean_acc, ci


def train(cfg, resume_path: str = None):
    """
    Main training loop following episodic training paradigm.

    Args:
        cfg: OmegaConf configuration object
        resume_path: Optional path to checkpoint for resuming training
    """
    # Setup
    set_seed(cfg.training.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logger('manta', log_dir=cfg.output.log_dir)
    tb_logger = TensorBoardLogger(cfg.output.log_dir)
    timer = Timer()

    logger.info(f"Device: {device}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Build model
    model = build_model(cfg).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Build optimizer
    optimizer, scheduler = build_optimizer(model, cfg)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = VideoDataset(
        split_file=os.path.join(cfg.data.split_dir, 'train.txt'),
        num_frames=cfg.data.num_frames,
        image_size=cfg.data.image_size,
        resize_size=cfg.data.resize_size,
        use_flip=cfg.data.use_horizontal_flip,
        mode='train',
    )
    val_dataset = VideoDataset(
        split_file=os.path.join(cfg.data.split_dir, 'val.txt'),
        num_frames=cfg.data.num_frames,
        image_size=cfg.data.image_size,
        resize_size=cfg.data.resize_size,
        use_flip=False,
        mode='test',
    )
    logger.info(f"Train: {train_dataset.get_num_classes()} classes, {len(train_dataset)} videos")
    logger.info(f"Val: {val_dataset.get_num_classes()} classes, {len(val_dataset)} videos")

    # Resume from checkpoint
    start_task = 0
    best_val_acc = 0.0

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_task = checkpoint['task_idx'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Resumed from task {start_task}, best_val_acc={best_val_acc:.4f}")

    # Create output directories
    os.makedirs(cfg.output.save_dir, exist_ok=True)

    # ==================== Training Loop ====================
    logger.info(f"Starting training from task {start_task} to {cfg.training.num_tasks}")

    train_episode_ds = EpisodeDataset(
        dataset=train_dataset,
        num_episodes=cfg.training.num_tasks - start_task,
        n_way=cfg.episode.n_way,
        k_shot=cfg.episode.k_shot,
        n_query=cfg.episode.num_query,
    )
    train_loader = DataLoader(
        train_episode_ds,
        batch_size=None,
        num_workers=cfg.training.get('num_workers', 4),
        pin_memory=True,
    )

    model.train()

    for task_idx, (support, query, sup_labels, que_labels) in zip(
        range(start_task, cfg.training.num_tasks), 
        tqdm(train_loader, desc="Training")
    ):
        support = support.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        sup_labels = sup_labels.to(device, non_blocking=True)
        que_labels = que_labels.to(device, non_blocking=True)

        # --- Forward pass ---
        result = model(
            support_videos=support,
            query_videos=query,
            support_labels=sup_labels,
            query_labels=que_labels,
            n_way=cfg.episode.n_way,
            k_shot=cfg.episode.k_shot,
            mode='train',
        )

        loss = result['loss']

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=cfg.training.get('grad_clip', 10.0),
        )

        optimizer.step()
        scheduler.step()

        # --- Logging ---
        if task_idx % cfg.training.get('log_interval', 100) == 0:
            current_lr = scheduler.get_last_lr()[0]
            tb_logger.log_train_step(
                step=task_idx,
                loss=loss.item(),
                accuracy=result['accuracy'].item(),
                L_ce=result['L_ce'].item(),
                L_hc=result['L_hc'].item(),
                lr=current_lr,
            )

            if task_idx % (cfg.training.get('log_interval', 100) * 10) == 0:
                eta = timer.get_eta(task_idx - start_task + 1,
                                    cfg.training.num_tasks - start_task)
                logger.info(
                    f"Task {task_idx}/{cfg.training.num_tasks} | "
                    f"Loss: {loss.item():.4f} (CE: {result['L_ce'].item():.4f}, "
                    f"HC: {result['L_hc'].item():.4f}) | "
                    f"Acc: {result['accuracy'].item()*100:.1f}% | "
                    f"LR: {current_lr:.6f} | ETA: {eta}"
                )

        # --- Validation ---
        if (task_idx + 1) % cfg.evaluation.interval == 0:
            val_acc, val_ci = evaluate(
                model, val_dataset, cfg, device,
                num_tasks=cfg.evaluation.get('val_tasks', 600),
            )
            tb_logger.log_val_step(task_idx, val_acc, val_ci)
            logger.info(
                f"[Validation @ Task {task_idx+1}] "
                f"Acc: {val_acc*100:.2f}% ± {val_ci*100:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(cfg.output.save_dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'task_idx': task_idx,
                    'best_val_acc': best_val_acc,
                    'config': OmegaConf.to_container(cfg),
                }, save_path)
                logger.info(f"  → New best model saved! ({best_val_acc*100:.2f}%)")

            # Resume training mode
            model.train()

        # --- Periodic checkpoint ---
        if (task_idx + 1) % cfg.training.get('save_interval', 5000) == 0:
            save_path = os.path.join(cfg.output.save_dir, f'checkpoint_task{task_idx+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'task_idx': task_idx,
                'best_val_acc': best_val_acc,
                'config': OmegaConf.to_container(cfg),
            }, save_path)

    # ==================== Training Complete ====================
    tb_logger.close()
    elapsed = timer.get_elapsed()
    logger.info(f"Training complete! Total time: {elapsed}")
    logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Best model saved at: {os.path.join(cfg.output.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MANTA Training')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint for resuming training'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override random seed from config'
    )
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override seed if specified
    if args.seed is not None:
        cfg.training.seed = args.seed

    # Run training
    train(cfg, resume_path=args.resume)
