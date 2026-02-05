import subprocess
import os

def _run(cmd, step_name):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed (exit {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

def run_sfm(frames_dir, output_dir):
    # COLMAP commands for SfM (assumes COLMAP in PATH)
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    
    # Feature extraction (lens correction handled here)
    _run(
        ["colmap", "feature_extractor", "--database_path", database_path, "--image_path", frames_dir],
        "Feature extraction",
    )
    # Matching
    _run(
        ["colmap", "exhaustive_matcher", "--database_path", database_path],
        "Feature matching",
    )
    # Sparse reconstruction
    _run(
        ["colmap", "mapper", "--database_path", database_path, "--image_path", frames_dir, "--output_path", sparse_dir],
        "Sparse reconstruction",
    )

    # Use sparse model to generate a point cloud (works without CUDA).
    sparse_model_dir = os.path.join(sparse_dir, "0")
    sparse_ply_path = os.path.join(output_dir, "sparse.ply")
    if not os.path.isdir(sparse_model_dir):
        raise RuntimeError(
            "Sparse model directory not found. "
            "COLMAP did not create a reconstruction at sparse/0."
        )
    _run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            sparse_model_dir,
            "--output_path",
            sparse_ply_path,
            "--output_type",
            "PLY",
        ],
        "Model conversion (PLY)",
    )

    return sparse_ply_path  # Sparse point cloud file
