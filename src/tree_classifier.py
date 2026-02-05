import numpy as np

def classify_species(trees):
    results = []
    for tree in trees:
        # Extract features
        height = tree[:, 2].max() - tree[:, 2].min()
        width = np.ptp(tree[:, 0])  # X range
        ratio = height / width if width > 0 else 0
        
        # Heuristic classification
        if ratio > 5:
            species = "Pine (tall/thin)"
        elif ratio > 2:
            species = "Oak (medium)"
        else:
            species = "Unknown"
        
        results.append({"species": species, "height": height, "width": width})
    
    return results
