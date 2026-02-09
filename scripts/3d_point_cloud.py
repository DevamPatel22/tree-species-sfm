import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
#%%
image_names = ["/Users/devampatel/Desktop/tree-species-sfm/results/frames/frame_0000.jpg", "/Users/devampatel/Desktop/tree-species-sfm/results/frames/frame_0001.jpg", "/Users/devampatel/Desktop/tree-species-sfm/results/frames/frame_0002.jpg", "/Users/devampatel/Desktop/tree-species-sfm/results/frames/frame_0003.jpg", "/Users/devampatel/Desktop/tree-species-sfm/results/frames/frame_0004.jpg"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        
#model = VGGT()
#_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
#model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

#%%
#pts = predictions["world_points"].detach().cpu().numpy()[0, 0, :, :, :]
#depth = predictions["depth"].detach().cpu().numpy()

#%%
import os
import numpy as np
import torch

def to_numpy(x):
    return x.detach().float().cpu().numpy()


from PIL import Image

def load_rgb_images(paths, size=None):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        imgs.append(np.array(img))
    return np.stack(imgs)  # [T,H,W,3], uint8


def write_ply_xyzrgb(path, xyz, rgb):
    """
    xyz: [N,3] float
    rgb: [N,3] uint8
    """
    assert xyz.shape[0] == rgb.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x,y,z), (r,g,b) in zip(xyz, rgb):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

# --- pull tensors ---
wp = predictions["world_points"]          # expected to contain xyz in world coords
depth = predictions.get("depth", None)
conf = predictions.get("confidence", None) or predictions.get("conf", None) or predictions.get("scores", None)

print("world_points shape:", tuple(wp.shape))

wp_np = to_numpy(wp)

# --- normalize shape to [T, H, W, 3] (drop batch) ---
# Common cases:
# [B,T,H,W,3], [B,H,W,3], [B,T,3,H,W], [B,3,H,W]
if wp_np.ndim == 5 and wp_np.shape[-1] == 3:
    # [B,T,H,W,3]
    wp_np = wp_np[0]
elif wp_np.ndim == 4 and wp_np.shape[-1] == 3:
    # [B,H,W,3]
    wp_np = wp_np[0:1]  # -> [T=1,H,W,3]
elif wp_np.ndim == 5 and wp_np.shape[2] == 3:
    # [B,T,3,H,W] -> [B,T,H,W,3]
    wp_np = np.transpose(wp_np, (0,1,3,4,2))[0]
elif wp_np.ndim == 4 and wp_np.shape[1] == 3:
    # [B,3,H,W] -> [T=1,H,W,3]
    wp_np = np.transpose(wp_np, (0,2,3,1))[0:1]
else:
    raise ValueError(f"Unexpected world_points shape: {wp_np.shape}")

T, H, W, _ = wp_np.shape
print("Normalized to:", (T,H,W,3))

# --- build mask for valid points ---
# If confidence exists, threshold it. Otherwise use depth validity if present.
mask = np.isfinite(wp_np).all(axis=-1)


# Reload RGB images at correct resolution
rgb_imgs = load_rgb_images(image_names, size=(W, H))  # [T,H,W,3]

# Apply same stride as geometry
stride = 1
wp_ds   = wp_np[:, ::stride, ::stride, :]
rgb_ds  = rgb_imgs[:, ::stride, ::stride, :]
mask_ds = mask[:, ::stride, ::stride]

xyz = wp_ds[mask_ds]
rgb = rgb_ds[mask_ds]

print("XYZ:", xyz.shape, "RGB:", rgb.shape)

write_ply_xyzrgb("out/vggt_points_rgb.ply", xyz, rgb)
print("Wrote out/vggt_points_rgb.ply")
