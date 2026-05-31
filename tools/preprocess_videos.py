"""
Video Preprocessing Tool — Decode videos to frames.

Paper specification:
    "All videos are decoded at their native frame rate and resized to 256×256."

Usage:
    python tools/preprocess_videos.py \
        --dataset kinetics \
        --video_dir data/kinetics/videos \
        --output_dir data/kinetics/frames \
        --size 256
"""

import os
import argparse
from pathlib import Path

import cv2
from decord import VideoReader, cpu
from tqdm import tqdm


def preprocess_dataset(
    video_dir: str,
    output_dir: str,
    size: int = 256,
    max_frames: int = None,
):
    """
    Decode all videos in a directory to JPEG frames.

    Args:
        video_dir: Root directory containing video files (can be nested)
        output_dir: Output directory for extracted frames
        size: Resize frames to size×size (default 256)
        max_frames: Maximum frames to extract per video (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    valid_extensions = {'.mp4', '.avi', '.webm', '.mkv', '.mov', '.flv'}
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                video_files.append(os.path.join(root, f))

    print(f"Found {len(video_files)} videos in {video_dir}")

    success_count = 0
    error_count = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        # Create output directory maintaining relative structure
        rel_path = os.path.relpath(video_path, video_dir)
        frame_dir = os.path.join(output_dir, os.path.splitext(rel_path)[0])
        os.makedirs(frame_dir, exist_ok=True)

        try:
            # Decode video using decord (fast GPU-accelerated decoder)
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            # Determine frames to extract
            if max_frames and total_frames > max_frames:
                # Uniform sampling if too many frames
                import numpy as np
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                indices = range(total_frames)

            # Extract and save frames
            for i, idx in enumerate(indices):
                frame = vr[idx].asnumpy()  # RGB format
                frame = cv2.resize(frame, (size, size))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(frame_dir, f'frame_{i:06d}.jpg')
                cv2.imwrite(
                    frame_path, frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

            success_count += 1

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            error_count += 1

    print(f"\nDone! Success: {success_count}, Errors: {error_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Preprocessing')
    parser.add_argument(
        '--dataset', type=str,
        choices=['ssv2', 'kinetics', 'ucf101', 'hmdb51'],
        help='Dataset name (for logging)'
    )
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for frames')
    parser.add_argument('--size', type=int, default=256,
                        help='Resize frames to size×size (default: 256)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames per video (default: all)')
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Frame size: {args.size}×{args.size}")

    preprocess_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        size=args.size,
        max_frames=args.max_frames,
    )
