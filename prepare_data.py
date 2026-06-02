#!/usr/bin/env python3
"""
Data Preparation Script for MANTA.

Converts raw video files from the 'all/' directory into the format
expected by MANTA's VideoDataset:

Input (all/):
    CCC_SSS_NNN.mp4   (CCC=class, SSS=subject, NNN=sample)
    e.g., 001_001_001.mp4, 001_001_002.mp4, ..., 064_010_005.mp4

Output:
    data/frames/
        CCC/                      ← one dir per class
            CCC_SSS_NNN/          ← one dir per video
                frame_000000.jpg
                frame_000001.jpg
                ...
    data/splits/
        train.txt                 ← training classes
        val.txt                   ← validation classes
        test.txt                  ← test classes

Split Strategy (Few-Shot Learning):
    Classes are split (NOT videos), so train/val/test have DISJOINT classes.
    64 classes split as: 38 train / 12 val / 14 test
    (roughly 60/19/21 ratio — standard for FSAR benchmarks)

Usage:
    python prepare_data.py
    python prepare_data.py --source all --output data --fps 5
    python prepare_data.py --source all --output data --fps 0  # extract ALL frames
"""

import os
import sys
import argparse
import subprocess
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_frames(video_path: str, output_dir: str, fps: int = 0) -> int:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to input video (.mp4)
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract. 0 = extract all frames.

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build ffmpeg command
    cmd = ['ffmpeg', '-i', video_path, '-q:v', '2']  # quality 2 (high)

    if fps > 0:
        cmd.extend(['-vf', f'fps={fps}'])

    cmd.extend([
        os.path.join(output_dir, 'frame_%06d.jpg'),
        '-hide_banner', '-loglevel', 'error'
    ])

    subprocess.run(cmd, check=True)

    # Count extracted frames
    n_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    return n_frames


def parse_filename(filename: str) -> dict:
    """
    Parse video filename to extract class, subject, sample info.

    Args:
        filename: e.g., '001_002_003.mp4'

    Returns:
        dict with 'class_id', 'subject_id', 'sample_id', 'stem'
    """
    stem = Path(filename).stem  # '001_002_003'
    parts = stem.split('_')

    return {
        'class_id': parts[0],        # '001'
        'subject_id': parts[1],      # '002'
        'sample_id': parts[2],       # '003'
        'stem': stem,                # '001_002_003'
        'filename': filename,        # '001_002_003.mp4'
    }


def create_splits(
    class_ids: list,
    train_ratio: float = 0.6,
    val_ratio: float = 0.19,
    seed: int = 42,
) -> dict:
    """
    Split classes into train/val/test sets.

    For few-shot learning, classes (not samples) are split into
    disjoint sets. This ensures no class overlap between splits.

    Args:
        class_ids: Sorted list of unique class IDs
        train_ratio: Fraction of classes for training
        val_ratio: Fraction of classes for validation
        seed: Random seed for reproducibility

    Returns:
        dict mapping split_name → list of class_ids
    """
    rng = random.Random(seed)
    shuffled = list(class_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': sorted(shuffled[:n_train]),
        'val': sorted(shuffled[n_train:n_train + n_val]),
        'test': sorted(shuffled[n_train + n_val:]),
    }

    return splits


def write_split_file(
    split_file: str,
    videos: list,
    frames_dir: str,
):
    """
    Write a split file in the format expected by VideoDataset.

    Format: one line per video:
        /absolute/path/to/frames_dir/CLASS/VIDEO_STEM CLASS

    Args:
        split_file: Path to output split file
        videos: List of dicts from parse_filename
        frames_dir: Absolute path to frames directory
    """
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    with open(split_file, 'w') as f:
        for v in sorted(videos, key=lambda x: x['stem']):
            video_frame_dir = os.path.join(frames_dir, v['class_id'], v['stem'])
            f.write(f"{video_frame_dir} {v['class_id']}\n")


def main():
    parser = argparse.ArgumentParser(description='MANTA Data Preparation')
    parser.add_argument(
        '--source', type=str, default='all',
        help='Source directory containing raw video files (default: all)'
    )
    parser.add_argument(
        '--output', type=str, default='data',
        help='Output directory for processed data (default: data)'
    )
    parser.add_argument(
        '--fps', type=int, default=0,
        help='Frames per second to extract. 0 = all frames (default: 0)'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.6,
        help='Fraction of classes for training (default: 0.6)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.19,
        help='Fraction of classes for validation (default: 0.19)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for class splitting (default: 42)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers for frame extraction (default: 8)'
    )
    parser.add_argument(
        '--skip_extract', action='store_true',
        help='Skip frame extraction (only generate split files)'
    )
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, args.source) if not os.path.isabs(args.source) else args.source
    output_dir = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    frames_dir = os.path.join(output_dir, 'frames')
    splits_dir = os.path.join(output_dir, 'splits')

    # ==================== Step 1: Scan source videos ====================
    print(f"Scanning source directory: {source_dir}")
    video_files = sorted([
        f for f in os.listdir(source_dir)
        if f.endswith('.mp4')
    ])

    if not video_files:
        print(f"ERROR: No .mp4 files found in {source_dir}")
        sys.exit(1)

    # Parse all filenames
    videos = [parse_filename(f) for f in video_files]

    # Group by class
    class_to_videos = defaultdict(list)
    for v in videos:
        class_to_videos[v['class_id']].append(v)

    all_classes = sorted(class_to_videos.keys())
    print(f"  Found {len(video_files)} videos across {len(all_classes)} classes")
    print(f"  Classes: {all_classes[0]} to {all_classes[-1]}")
    print(f"  Videos per class: {len(class_to_videos[all_classes[0]])}")

    # ==================== Step 2: Extract frames ====================
    if not args.skip_extract:
        print(f"\nExtracting frames to: {frames_dir}")
        fps_msg = f"at {args.fps} FPS" if args.fps > 0 else "all frames"
        print(f"  Mode: {fps_msg}")
        print(f"  Workers: {args.workers}")

        # Build extraction tasks
        tasks = []
        for v in videos:
            video_path = os.path.join(source_dir, v['filename'])
            out_dir = os.path.join(frames_dir, v['class_id'], v['stem'])
            tasks.append((video_path, out_dir))

        # Execute in parallel
        completed = 0
        failed = 0
        total = len(tasks)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for video_path, out_dir in tasks:
                # Skip if already extracted
                if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
                    completed += 1
                    continue
                fut = executor.submit(extract_frames, video_path, out_dir, args.fps)
                futures[fut] = video_path

            for fut in as_completed(futures):
                completed += 1
                try:
                    n_frames = fut.result()
                    if completed % 100 == 0 or completed == total:
                        print(f"  [{completed}/{total}] Extracted {n_frames} frames from {os.path.basename(futures[fut])}")
                except Exception as e:
                    failed += 1
                    print(f"  ERROR extracting {futures[fut]}: {e}")

        print(f"\n  Extraction complete: {completed - failed} succeeded, {failed} failed")
    else:
        print("\nSkipping frame extraction (--skip_extract)")

    # ==================== Step 3: Verify extracted frames ====================
    print("\nVerifying extracted frames...")
    empty_dirs = []
    total_frames = 0
    for v in videos:
        frame_dir = os.path.join(frames_dir, v['class_id'], v['stem'])
        if not os.path.isdir(frame_dir):
            empty_dirs.append(v['stem'])
            continue
        n = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if n == 0:
            empty_dirs.append(v['stem'])
        total_frames += n

    if empty_dirs:
        print(f"  WARNING: {len(empty_dirs)} videos have no extracted frames!")
        for d in empty_dirs[:5]:
            print(f"    - {d}")
        if len(empty_dirs) > 5:
            print(f"    ... and {len(empty_dirs) - 5} more")
    else:
        print(f"  All {len(videos)} videos have frames extracted")

    print(f"  Total frames: {total_frames:,}")
    print(f"  Average frames per video: {total_frames / len(videos):.1f}")

    # ==================== Step 4: Create class splits ====================
    print(f"\nCreating train/val/test splits (seed={args.seed})...")
    splits = create_splits(
        all_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, split_classes in splits.items():
        split_videos = []
        for cls in split_classes:
            split_videos.extend(class_to_videos[cls])

        split_file = os.path.join(splits_dir, f'{split_name}.txt')
        write_split_file(split_file, split_videos, frames_dir)

        print(f"  {split_name:5s}: {len(split_classes):2d} classes, "
              f"{len(split_videos):4d} videos → {split_file}")

    # ==================== Step 5: Print summary ====================
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Frames:   {frames_dir}")
    print(f"  Splits:   {splits_dir}")
    print(f"  Classes:  {len(all_classes)} total")
    print(f"    Train:  {len(splits['train'])} classes ({', '.join(splits['train'][:5])}...)")
    print(f"    Val:    {len(splits['val'])} classes ({', '.join(splits['val'][:5])}...)")
    print(f"    Test:   {len(splits['test'])} classes ({', '.join(splits['test'][:5])}...)")
    print(f"  Videos:   {len(videos)} total")
    print(f"  Frames:   {total_frames:,} total")
    print("")
    print("Next steps:")
    print("  1. Update config YAML to point to the data directory")
    print("  2. Run training: python train.py --config configs/your_config.yaml")
    print("=" * 60)


if __name__ == '__main__':
    main()
