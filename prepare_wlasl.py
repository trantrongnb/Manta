#!/usr/bin/env python3
"""
Data Preparation Script for WLASL.

Converts raw video files from the 'data/wlasl/raw/' directory into the format
expected by MANTA's VideoDataset:

Input (data/wlasl/raw/):
    [class_name]/
        [video_id].mp4
    e.g., a/00295.mp4

Output:
    data/wlasl/frames/
        [class_name]/
            [video_id]/
                frame_000000.jpg
                frame_000001.jpg
                ...
    data/wlasl/splits/
        train.txt
        val.txt
        test.txt

Usage:
    python prepare_wlasl.py
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
    """Extract frames from a video file using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = ['ffmpeg', '-i', video_path, '-q:v', '2']
    if fps > 0:
        cmd.extend(['-vf', f'fps={fps}'])
    cmd.extend([
        os.path.join(output_dir, 'frame_%06d.jpg'),
        '-hide_banner', '-loglevel', 'error'
    ])

    subprocess.run(cmd, check=True)

    n_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    return n_frames


def create_splits(
    class_ids: list,
    train_ratio: float = 0.6,
    val_ratio: float = 0.19,
    seed: int = 42,
) -> dict:
    """Split classes into disjoint train/val/test sets."""
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
    """Write split file with format: /path/to/frames_dir/CLASS/VIDEO CLASS"""
    os.makedirs(os.path.dirname(split_file), exist_ok=True)

    # Note: For text labels, we map class strings to integer IDs.
    # We will build a stable mapping (sorted alphabetically).
    all_classes = sorted(list(set(v['class_name'] for v in videos)))
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    with open(split_file, 'w') as f:
        for v in sorted(videos, key=lambda x: (x['class_name'], x['stem'])):
            video_frame_dir = os.path.join(frames_dir, v['class_name'], v['stem'])
            class_idx = class_to_idx[v['class_name']]
            f.write(f"{video_frame_dir} {class_idx}\n")


def main():
    parser = argparse.ArgumentParser(description='WLASL Data Preparation')
    parser.add_argument(
        '--source', type=str, default='data/wlasl/raw',
        help='Source directory containing raw class folders (default: data/wlasl/raw)'
    )
    parser.add_argument(
        '--output', type=str, default='data/wlasl',
        help='Output directory for processed data (default: data/wlasl)'
    )
    parser.add_argument('--fps', type=int, default=0, help='Frames per second')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.19)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip_extract', action='store_true')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, args.source) if not os.path.isabs(args.source) else args.source
    output_dir = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    frames_dir = os.path.join(output_dir, 'frames')
    splits_dir = os.path.join(output_dir, 'splits')

    # ==================== Step 1: Scan source videos ====================
    print(f"Scanning source directory: {source_dir}")
    videos = []
    class_to_videos = defaultdict(list)

    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for f in os.listdir(class_path):
            if f.endswith('.mp4'):
                v = {
                    'class_name': class_name,
                    'stem': Path(f).stem,
                    'filename': f,
                    'rel_path': os.path.join(class_name, f)
                }
                videos.append(v)
                class_to_videos[class_name].append(v)

    all_classes = sorted(class_to_videos.keys())
    print(f"  Found {len(videos)} videos across {len(all_classes)} classes")

    # ==================== Step 2: Extract frames ====================
    if not args.skip_extract:
        print(f"\nExtracting frames to: {frames_dir}")
        tasks = []
        for v in videos:
            video_path = os.path.join(source_dir, v['rel_path'])
            out_dir = os.path.join(frames_dir, v['class_name'], v['stem'])
            tasks.append((video_path, out_dir))

        completed = 0
        failed = 0
        total = len(tasks)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for video_path, out_dir in tasks:
                if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
                    completed += 1
                    continue
                fut = executor.submit(extract_frames, video_path, out_dir, args.fps)
                futures[fut] = video_path

            for fut in as_completed(futures):
                completed += 1
                try:
                    fut.result()
                    if completed % 100 == 0 or completed == total:
                        print(f"  [{completed}/{total}] Extracted {os.path.basename(futures[fut])}")
                except Exception as e:
                    failed += 1
                    print(f"  ERROR extracting {futures[fut]}: {e}")
        print(f"\n  Extraction complete: {completed - failed} succeeded, {failed} failed")
    else:
        print("\nSkipping frame extraction")

    # ==================== Step 3: Create class splits ====================
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
        write_split_file(split_file, videos, frames_dir) # Note: we pass ALL videos to build global class_to_idx mapping but only write lines for split_videos?
        # Wait, if we do that, we need to pass only split_videos to write_split_file, but ensure class_to_idx is global.
        pass

    # Actually let's rewrite the split writing block carefully.
    os.makedirs(splits_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    
    for split_name, split_classes in splits.items():
        split_videos = []
        for cls in split_classes:
            split_videos.extend(class_to_videos[cls])
            
        split_file = os.path.join(splits_dir, f'{split_name}.txt')
        with open(split_file, 'w') as f:
            for v in sorted(split_videos, key=lambda x: (x['class_name'], x['stem'])):
                video_frame_dir = os.path.join(frames_dir, v['class_name'], v['stem'])
                class_idx = class_to_idx[v['class_name']]
                f.write(f"{video_frame_dir} {class_idx}\n")
        
        print(f"  {split_name:5s}: {len(split_classes):4d} classes, {len(split_videos):5d} videos → {split_file}")

    print("\nData preparation complete!")

if __name__ == '__main__':
    main()
