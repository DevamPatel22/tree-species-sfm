import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def segment_trees(point_cloud_path, eps=0.5, min_samples=10):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise RuntimeError("Point cloud is empty. Check SfM output.")
    
    # Filter by height (assume Z is up; trees >2m)
    filtered_points = points[points[:, 2] > 2.0]
    if filtered_points.size == 0:
        raise RuntimeError("No points above height threshold. Try lowering the 2.0m cutoff.")
    
    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = db.labels_
    
    # Identify trunks: Points with high vertical density (novel heuristic)
    trunk_labels = []
    for label in np.unique(labels):
        if label == -1: continue  # Noise
        cluster = filtered_points[labels == label]
        height_range = cluster[:, 2].max() - cluster[:, 2].min()
        if height_range > 5.0:  # Tall clusters likely trunks
            trunk_labels.append(label)
    
    # Segment full trees (expand from trunks)
    trees = []
    for trunk_label in trunk_labels:
        trunk_points = filtered_points[labels == trunk_label]
        # Simple expansion: Add nearby points (within 2m radius)
        distances = np.linalg.norm(filtered_points - trunk_points.mean(axis=0), axis=1)
        tree_points = filtered_points[distances < 2.0]
        trees.append(tree_points)
    
    return trees  # List of point arrays per tree
