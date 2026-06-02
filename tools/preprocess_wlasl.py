"""
WLASL Preprocessing Tool — Decode WLASL .mp4 videos to JPEG frames.

WLASL (Word-Level American Sign Language) dataset structure:
    WLASL/
        {class_word}/
            {video_id}.mp4

Output structure:
    data/wlasl/frames/
        {class_word}/
            {video_id}/
                frame_000000.jpg
                frame_000001.jpg
                ...

Usage:
    python tools/preprocess_wlasl.py \
        --video_dir WLASL \
        --output_dir data/wlasl/frames \
        --size 256 \
        --max_frames 32
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def preprocess_wlasl(
    video_dir: str,
    output_dir: str,
    size: int = 256,
    max_frames: int = 32,
):
    """
    Decode all WLASL videos to JPEG frames.

    Args:
        video_dir: Path to WLASL directory (class/video.mp4 structure)
        output_dir: Output directory for extracted frames
        size: Resize frames to size×size (default 256)
        max_frames: Maximum frames to extract per video (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all .mp4 files organized by class
    class_dirs = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])

    video_files = []
    for cls in class_dirs:
        cls_dir = os.path.join(video_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith('.mp4'):
                video_files.append((cls, os.path.join(cls_dir, fname)))

    print(f"Found {len(video_files)} videos across {len(class_dirs)} classes in {video_dir}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    error_count = 0
    skipped_count = 0

    for cls_name, video_path in tqdm(video_files, desc="Processing WLASL videos"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(output_dir, cls_name, video_id)

        if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 0:
            skipped_count += 1
            continue

        os.makedirs(frame_dir, exist_ok=True)

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                cap.release()
                error_count += 1
                continue

            if max_frames and total_frames > max_frames:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            else:
                indices = range(total_frames)

            frame_idx = 0
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (size, size))
                frame_path = os.path.join(frame_dir, f'frame_{frame_idx:06d}.jpg')
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1

            cap.release()

            if frame_idx > 0:
                success_count += 1
            else:
                os.rmdir(frame_dir)
                error_count += 1

        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            error_count += 1

    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess WLASL videos to frames')
    parser.add_argument('--video_dir', type=str, default='WLASL',
                        help='WLASL directory (default: WLASL)')
    parser.add_argument('--output_dir', type=str, default='data/wlasl/frames',
                        help='Output directory for frames (default: data/wlasl/frames)')
    parser.add_argument('--size', type=int, default=256,
                        help='Resize frames to size×size (default: 256)')
    parser.add_argument('--max_frames', type=int, default=32,
                        help='Max frames per video (default: 32)')
    args = parser.parse_args()

    preprocess_wlasl(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        size=args.size,
        max_frames=args.max_frames,
    )
