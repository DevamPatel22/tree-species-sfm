import os
from src.frame_extractor import extract_frames
from src.colmap_runner import run_sfm
from src.tree_segmenter import segment_trees
from src.tree_classifier import classify_species

def main(video_path, output_dir):
    frames = extract_frames(video_path, os.path.join(output_dir, "frames"))
    pc = run_sfm(os.path.join(output_dir, "frames"), os.path.join(output_dir, "sfm"))
    trees = segment_trees(pc)
    results = classify_species(trees)
    print("Results:", results)

if __name__ == "__main__":
  import sys
  main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "./results")
