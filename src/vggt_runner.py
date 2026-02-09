import os

import numpy as np
import torch
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _to_numpy(x):
    return x.detach().float().cpu().numpy()


def _load_rgb_images(paths, size=None):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        imgs.append(np.array(img))
    return np.stack(imgs)  # [T,H,W,3], uint8


def _write_ply_xyzrgb(path, xyz, rgb):
    assert xyz.shape[0] == rgb.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def run_vggt_pointcloud(image_paths, output_ply):
    if not image_paths:
        raise RuntimeError("No image paths provided to VGGT.")

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    wp = predictions["world_points"]
    wp_np = _to_numpy(wp)

    if wp_np.ndim == 5 and wp_np.shape[-1] == 3:
        wp_np = wp_np[0]
    elif wp_np.ndim == 4 and wp_np.shape[-1] == 3:
        wp_np = wp_np[0:1]
    elif wp_np.ndim == 5 and wp_np.shape[2] == 3:
        wp_np = np.transpose(wp_np, (0, 1, 3, 4, 2))[0]
    elif wp_np.ndim == 4 and wp_np.shape[1] == 3:
        wp_np = np.transpose(wp_np, (0, 2, 3, 1))[0:1]
    else:
        raise ValueError(f"Unexpected world_points shape: {wp_np.shape}")

    _, h, w, _ = wp_np.shape
    mask = np.isfinite(wp_np).all(axis=-1)

    rgb_imgs = _load_rgb_images(image_paths, size=(w, h))
    xyz = wp_np[mask]
    rgb = rgb_imgs[mask]

    _write_ply_xyzrgb(output_ply, xyz, rgb)
    return output_ply
