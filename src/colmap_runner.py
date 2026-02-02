import subprocess
import os

def run_sfm(frames_dir, output_dir):
    # COLMAP commands for SfM (assumes COLMAP in PATH)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")
    
    # Feature extraction (lens correction handled here)
    subprocess.run(["colmap", "feature_extractor", "--database_path", database_path, "--image_path", frames_dir])
    # Matching
    subprocess.run(["colmap", "exhaustive_matcher", "--database_path", database_path])
    # Sparse reconstruction
    subprocess.run(["colmap", "mapper", "--database_path", database_path, "--image_path", frames_dir, "--output_path", sparse_dir])
    # Dense reconstruction (point cloud cleanup)
    subprocess.run(["colmap", "image_undistorter", "--image_path", frames_dir, "--input_path", sparse_dir, "--output_path", dense_dir])
    subprocess.run(["colmap", "patch_match_stereo", "--workspace_path", dense_dir])
    subprocess.run(["colmap", "stereo_fusion", "--workspace_path", dense_dir, "--output_path", os.path.join(dense_dir, "fused.ply")])
    
    return os.path.join(dense_dir, "fused.ply")  # Point cloud file
