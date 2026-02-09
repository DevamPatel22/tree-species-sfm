import argparse
import os
import re
from src.colmap_runner import run_sfm
from src.vggt_runner import run_vggt_pointcloud

def _pick_evenly(items, count):
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    step = (len(items) - 1) / float(count - 1)
    return [items[int(round(i * step))] for i in range(count)]

def _list_frames(frames_dir):
    if not os.path.isdir(frames_dir):
        return []
    frames = [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if f.lower().endswith(".jpg")
    ]
    return frames


def _next_vggt_output_path(output_dir):
    vggt_dir = os.path.join(output_dir, "vggt")
    os.makedirs(vggt_dir, exist_ok=True)
    pattern = re.compile(r"^vggt_points_rgb_run(\d+)\.ply$")
    max_run = 0
    for name in os.listdir(vggt_dir):
        match = pattern.match(name)
        if match:
            max_run = max(max_run, int(match.group(1)))
    return os.path.join(vggt_dir, f"vggt_points_rgb_run{max_run + 1}.ply")


def main(
    video_path,
    output_dir,
    backend="colmap",
    vggt_frames=5,
    extract_frames_flag=False,
    skip_postprocess=False,
):
    frames_dir = os.path.join(output_dir, "frames")

    if backend == "colmap":
        from src.frame_extractor import extract_frames
        frames = extract_frames(video_path, frames_dir)
        pc = run_sfm(frames_dir, os.path.join(output_dir, "sfm"))
    else:
        if extract_frames_flag:
            from src.frame_extractor import extract_frames
            frames = extract_frames(video_path, frames_dir)
        else:
            frames = _list_frames(frames_dir)
        if not frames:
            raise RuntimeError(
                "No frames found for VGGT. Either run COLMAP first to create "
                f"{frames_dir}, or re-run with --extract-frames."
            )
        selected = _pick_evenly(frames, vggt_frames)
        pc = run_vggt_pointcloud(
            image_paths=selected,
            output_ply=_next_vggt_output_path(output_dir),
        )

    if not skip_postprocess:
        from src.tree_segmenter import segment_trees
        from src.tree_classifier import classify_species

        trees = segment_trees(pc)
        results = classify_species(trees)
        print("Results:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree species SfM pipeline")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="./results",
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--backend",
        choices=["colmap", "vggt"],
        default="colmap",
        help="Reconstruction backend (default: colmap)",
    )
    parser.add_argument(
        "--vggt-frames",
        type=int,
        default=5,
        help="Number of frames to feed into VGGT (default: 5)",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from video before running (requires OpenCV)",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip tree segmentation/classification (no Open3D/scikit-learn needed)",
    )
    args = parser.parse_args()
    main(
        args.video_path,
        args.output_dir,
        backend=args.backend,
        vggt_frames=args.vggt_frames,
        extract_frames_flag=args.extract_frames,
        skip_postprocess=args.skip_postprocess,
    )
