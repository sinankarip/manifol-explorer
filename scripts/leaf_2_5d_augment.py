import cv2
import numpy as np
from pathlib import Path
import argparse
import random

"""
Leaf synthetic augmentation pipeline.

Pipeline:
1. Saturation-based alpha extraction (fragile heuristic).
2. Perspective warp (simulates camera pose).
3. Shadow synthesis from alpha (approximate ambient occlusion).
4. Background randomization (low-frequency domain shift).
5. Mild color jitter & blur (sensor noise approx.).

Note: Does NOT model BRDF, viewpoint-dependent reflectance,
chromatic aberration, or leaf translucency. Use with caution.
"""

def sample_leaf_quad(size=224,
                     big_range=(0.6, 0.9),
                     small_range=(0.3, 0.5),
                     small_prob=0.16,
                     jitter_ratio=0.2):
    if random.random() < small_prob:
        s = random.uniform(*small_range)
    else:
        s = random.uniform(*big_range)

    box_w = s * size
    box_h = s * size

    margin_x = (size - box_w) * 0.5
    margin_y = (size - box_h) * 0.5
    cx = random.uniform(margin_x, size - margin_x)
    cy = random.uniform(margin_y, size - margin_y)

    hw = box_w * 0.5
    hh = box_h * 0.5
    pts = np.array(
        [
            [cx - hw, cy - hh],  
            [cx + hw, cy - hh],  
            [cx + hw, cy + hh],  
            [cx - hw, cy + hh],  
        ],
        dtype=np.float32,
    )

    max_j = jitter_ratio * min(box_w, box_h)
    jitter = np.random.uniform(-max_j, max_j, size=pts.shape).astype(np.float32)
    pts_j = pts + jitter

    pts_j[:, 0] = np.clip(pts_j[:, 0], 0, size - 1)
    pts_j[:, 1] = np.clip(pts_j[:, 1], 0, size - 1)

    return pts_j


def extract_leaf_alpha(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sat_thresh = 40   
    val_thresh = 40   
    mask = (s > sat_thresh) & (v > val_thresh)
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = 1 + np.argmax(areas)
        mask = np.where(labels == max_label, 255, 0).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def load_rgba(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image could not be read: {path}")

    alpha = extract_leaf_alpha(bgr)
    rgba = np.dstack([bgr, alpha])
    return rgba  # B,G,R,A


def random_background(size: int = 224) -> np.ndarray:
    base = np.full((size, size, 3), 150, dtype=np.uint8)
    noise = np.random.randint(-10, 11, size=(size, size, 3), dtype=np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def color_jitter_bgr(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    b = random.uniform(-0.15, 0.15)
    c = random.uniform(0.85, 1.15)
    img = img * c + b
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def compose_rgba_over_bg(fg_bgra: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    fg_bgr = fg_bgra[:, :, :3].astype(np.float32)
    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0
    bg = bg_bgr.astype(np.float32)
    out = fg_bgr * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_shadow_from_alpha(bg_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = bg_bgr.shape[:2]
    a = alpha.astype(np.float32) / 255.0
    k = random.choice([7, 9, 11])
    a_blur = cv2.GaussianBlur(a, (k, k), 0)

    dx = random.randint(3, 10)
    dy = random.randint(3, 10)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    a_shift = cv2.warpAffine(a_blur, M, (w, h), borderValue=0.0)

    strength = random.uniform(0.3, 0.6)
    shadow = 1.0 - a_shift * strength

    out = bg_bgr.astype(np.float32)
    out *= np.repeat(shadow[:, :, None], 3, axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_augmented_views(
    leaf_path: Path,
    out_dir: Path,
    num_views: int = 32,
    out_size: int = 224,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    leaf_bgra = load_rgba(leaf_path)
    h, w = leaf_bgra.shape[:2]

    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    for i in range(num_views):
        dst_pts = sample_leaf_quad(
            size=out_size,
            big_range=(0.7, 1.0),
            small_range=(0.3, 0.5),
            small_prob=0.2,
            jitter_ratio=0.12,
        )
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_bgra = cv2.warpPerspective(
            leaf_bgra,
            H,
            (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped_alpha = warped_bgra[:, :, 3]

        bg = random_background(out_size)
        bg = add_shadow_from_alpha(bg, warped_alpha)
        warped_bgra_j = warped_bgra.copy()
        warped_bgra_j[:, :, :3] = color_jitter_bgr(warped_bgra_j[:, :, :3])
        comp = compose_rgba_over_bg(warped_bgra_j, bg)

        if random.random() < 0.4:
            comp = cv2.GaussianBlur(comp, (3, 3), 0)
        if random.random() < 0.3:
            noise = np.random.normal(0, 4, comp.shape).astype(np.int16)
            comp = np.clip(comp.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out_path = out_dir / f"view_{i:04d}.png"
        cv2.imwrite(str(out_path), comp)
        if i % 10 == 0:
            print(f"[{i}/{num_views}] -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=64)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    leaf_path = Path(args.leaf_path)
    out_dir = Path(args.output_dir)
    assert leaf_path.is_file(), f"Leaf not found: {leaf_path}"

    generate_augmented_views(
        leaf_path=leaf_path,
        out_dir=out_dir,
        num_views=args.num_views,
        out_size=args.size,
    )


if __name__ == "__main__":
    main()
