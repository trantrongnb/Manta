"""
Create Train/Val/Test Splits for FSAR Benchmarks.

Paper Section 4.1:
    "We follow the class-disjoint split protocol of Zhu & Yang (2018),
     ensuring D_train ∩ D_val ∩ D_test = ∅ (no class overlap between splits)."

Split ratios:
    - Kinetics-100: 64 train / 16 val / 20 test classes
    - SSv2-Full:    64 train / 12 val / 24 test classes
    - UCF101:       70 train / 10 val / 21 test classes
    - HMDB51:       31 train / 10 val / 10 test classes

Usage:
    python tools/create_splits.py \
        --frame_dir data/kinetics/frames \
        --output_dir data/kinetics/splits \
        --dataset kinetics
"""

import os
import random
import argparse
from pathlib import Path


# Standard split configurations per dataset
SPLIT_CONFIGS = {
    'kinetics': {'train': 64, 'val': 16, 'test': 20},
    'ssv2': {'train': 64, 'val': 12, 'test': 24},
    'ucf101': {'train': 70, 'val': 10, 'test': 21},
    'hmdb51': {'train': 31, 'val': 10, 'test': 10},
}


def create_fsar_splits(
    frame_dir: str,
    output_dir: str,
    dataset: str = 'kinetics',
    seed: int = 42,
):
    """
    Create class-disjoint train/val/test splits.

    Ensures no class appears in more than one split (critical for FSAR evaluation).

    Args:
        frame_dir: Directory containing class subdirectories with video frames
        output_dir: Output directory for split files
        dataset: Dataset name (determines split ratios)
        seed: Random seed for reproducibility
    """
    # Get all class directories
    classes = sorted([
        d for d in os.listdir(frame_dir)
        if os.path.isdir(os.path.join(frame_dir, d))
    ])
    total_classes = len(classes)
    print(f"Found {total_classes} classes in {frame_dir}")

    # Determine split sizes
    if dataset in SPLIT_CONFIGS:
        config = SPLIT_CONFIGS[dataset]
        n_train = config['train']
        n_val = config['val']
        n_test = config['test']

        # Verify total matches
        expected_total = n_train + n_val + n_test
        if total_classes != expected_total:
            print(f"Warning: Found {total_classes} classes, expected {expected_total}")
            print(f"Adjusting split proportionally...")
            ratio_train = n_train / expected_total
            ratio_val = n_val / expected_total
            n_train = int(total_classes * ratio_train)
            n_val = int(total_classes * ratio_val)
            n_test = total_classes - n_train - n_val
    else:
        # Default: 64/16/20 ratio
        n_train = int(total_classes * 0.64)
        n_val = int(total_classes * 0.16)
        n_test = total_classes - n_train - n_val

    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # Shuffle classes with fixed seed
    random.seed(seed)
    shuffled_classes = classes.copy()
    random.shuffle(shuffled_classes)

    # Assign classes to splits
    splits = {
        'train': shuffled_classes[:n_train],
        'val': shuffled_classes[n_train:n_train + n_val],
        'test': shuffled_classes[n_train + n_val:],
    }

    # Verify no overlap
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"

    # Write split files
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_classes in splits.items():
        lines = []
        total_videos = 0

        for cls in sorted(split_classes):
            cls_dir = os.path.join(frame_dir, cls)
            # Each subdirectory in the class dir is a video
            videos = sorted([
                v for v in os.listdir(cls_dir)
                if os.path.isdir(os.path.join(cls_dir, v))
            ])

            for video in videos:
                video_path = os.path.join(cls_dir, video)
                lines.append(f"{video_path} {cls}\n")
                total_videos += 1

        # Write split file
        split_file = os.path.join(output_dir, f'{split_name}.txt')
        with open(split_file, 'w') as f:
            f.writelines(lines)

        print(f"  {split_name}: {len(split_classes)} classes, {total_videos} videos → {split_file}")

    # Write class list files (for reference)
    for split_name, split_classes in splits.items():
        class_file = os.path.join(output_dir, f'{split_name}_classes.txt')
        with open(class_file, 'w') as f:
            for cls in sorted(split_classes):
                f.write(f"{cls}\n")

    print("\nDone! Split files created successfully.")
    print("Verify: No class overlap between splits ✓")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create FSAR Splits')
    parser.add_argument('--frame_dir', type=str, required=True,
                        help='Directory with class/video/frame structure')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for split files')
    parser.add_argument('--dataset', type=str, default='kinetics',
                        choices=['kinetics', 'ssv2', 'ucf101', 'hmdb51'],
                        help='Dataset name (determines split ratios)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    create_fsar_splits(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        seed=args.seed,
    )
