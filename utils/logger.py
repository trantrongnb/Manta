"""
Logging Utilities for MANTA Training.

Provides:
    - Console logging with timestamps
    - TensorBoard integration for metric visualization
    - Training progress tracking
"""

import os
import sys
import time
import logging
from typing import Optional, Dict

import torch


def setup_logger(
    name: str = 'manta',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log file (if None, console only)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}.log')
        )
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """
    TensorBoard logger wrapper for MANTA training metrics.

    Tracks:
        - Training loss components (L_ce, L_hc, L_total)
        - Training accuracy
        - Validation accuracy
        - Learning rate
        - Per-scale analysis (optional)
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory for TensorBoard event files
        """
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_train_step(
        self,
        step: int,
        loss: float,
        accuracy: float,
        L_ce: float,
        L_hc: float,
        lr: float,
        **kwargs,
    ):
        """
        Log training metrics for a single step.

        Args:
            step: Current training step (episode number)
            loss: Total loss L_total
            accuracy: Episode accuracy
            L_ce: Cross-entropy loss
            L_hc: Hybrid contrastive loss
            lr: Current learning rate
            **kwargs: Additional metrics to log
        """
        self.writer.add_scalar('train/loss_total', loss, step)
        self.writer.add_scalar('train/loss_ce', L_ce, step)
        self.writer.add_scalar('train/loss_hc', L_hc, step)
        self.writer.add_scalar('train/accuracy', accuracy, step)
        self.writer.add_scalar('train/learning_rate', lr, step)

        for key, value in kwargs.items():
            self.writer.add_scalar(f'train/{key}', value, step)

    def log_val_step(self, step: int, accuracy: float, ci: float = 0.0):
        """
        Log validation metrics.

        Args:
            step: Current training step
            accuracy: Mean validation accuracy
            ci: 95% confidence interval half-width
        """
        self.writer.add_scalar('val/accuracy', accuracy, step)
        self.writer.add_scalar('val/ci_95', ci, step)

    def log_test_results(
        self,
        accuracy: float,
        ci: float,
        config: Dict,
    ):
        """
        Log final test results.

        Args:
            accuracy: Mean test accuracy
            ci: 95% confidence interval
            config: Experiment configuration dict
        """
        self.writer.add_text(
            'test/results',
            f"Accuracy: {accuracy*100:.2f}% ± {ci*100:.2f}%"
        )
        self.writer.add_text(
            'test/config',
            str(config)
        )

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class Timer:
    """Simple timer for tracking training speed."""

    def __init__(self):
        self.start_time = time.time()
        self.step_times = []

    def step(self):
        """Record a step completion."""
        current = time.time()
        if self.step_times:
            elapsed = current - self.step_times[-1]
        else:
            elapsed = current - self.start_time
        self.step_times.append(current)
        return elapsed

    def get_eta(self, current_step: int, total_steps: int) -> str:
        """
        Estimate time remaining.

        Args:
            current_step: Current step number
            total_steps: Total number of steps

        Returns:
            Formatted ETA string (e.g., "2h 15m 30s")
        """
        if current_step == 0:
            return "N/A"

        elapsed = time.time() - self.start_time
        rate = elapsed / current_step
        remaining = rate * (total_steps - current_step)

        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        seconds = int(remaining % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_elapsed(self) -> str:
        """Get total elapsed time as formatted string."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
