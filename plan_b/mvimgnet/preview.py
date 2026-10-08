"""Generate real MVImgNet2.0 preview panels from Dataset outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .cameras import camera_intrinsics_matrix, letterbox_transform
from .colmap_io import camera_center_world, qvec_to_rotmat, read_sparse_model
from .dataset import MVImgNetEpisodeDataset, mvimgnet_collate, read_jsonl


DEFAULT_ROOT = Path("/data1/haksoo/data/MVImgNet2.0")


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _label_tile(img: Image.Image, label: str, height: int = 42) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), label[:80], fill=(0, 0, 0))
    return out


def _concat_h(images: List[Image.Image]) -> Image.Image:
    width = sum(im.width for im in images)
    height = max(im.height for im in images)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def make_batch_panel(sample: Dict, path: Path) -> None:
    mi = sample["model_inputs"]
    sup = sample["supervision"]
    meta = sample["metadata"]
    tiles = []
    for i, img in enumerate(mi["context_images"]):
        frame = meta["context_frame_ids"][i]
        tiles.append(_label_tile(_tensor_to_pil(img), f"Context {i + 1}\n{frame}"))
    tiles.append(_label_tile(_tensor_to_pil(sup["target_image"]), f"GT Target\n{meta['target_frame_id']}"))
    panel = _concat_h(tiles)
    footer = Image.new("RGB", (panel.width, 70), "white")
    draw = ImageDraw.Draw(footer)
    text = (
        f"{meta['class_id']} {meta.get('class_name')} | {meta['instance_uid']} | "
        f"target rel rot {meta['target_relative_rotation_deg']:.2f} deg | "
        f"scale {meta['normalization_scale']:.6g} | text unavailable"
    )
    draw.text((6, 6), text, fill=(0, 0, 0))
    out = Image.new("RGB", (panel.width, panel.height + footer.height), "white")
    out.paste(panel, (0, 0))
    out.paste(footer, (0, panel.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def make_raw_multiview_panel(row: Dict, path: Path, max_views: int = 8) -> None:
    instance_path = Path(row["instance_path"])
    images = sorted((instance_path / "images").glob("*"))[:max_views]
    tiles = []
    for p in images:
        img = Image.open(p).convert("RGB")
        img.thumbnail((220, 220), Image.Resampling.BILINEAR)
        tiles.append(_label_tile(img.copy(), f"{row['class_id']} {row.get('class_name')}\n{row['instance_id']} / {p.name}", height=48))
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        _concat_h(tiles).save(path)


def make_trajectory(sample: Dict, path: Path) -> None:
    meta = sample["metadata"]
    instance_path = Path(meta["paths"]["instance_path"])
    _, images, _ = read_sparse_model(instance_path / "sparse" / "0", with_points=False)
    ctx = set(meta["context_image_ids"])
    tgt = meta["target_image_id"]
    centers = {image_id: camera_center_world(rec.qvec, rec.tvec) for image_id, rec in images.items()}
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    all_xyz = np.stack(list(centers.values()), axis=0)
    ax.scatter(all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2], s=8, c="lightgray", label="registered")
    for image_id in ctx:
        c = centers[image_id]
        ax.scatter(c[0], c[1], c[2], s=60, c="tab:blue", marker="o")
    c = centers[tgt]
    ax.scatter(c[0], c[1], c[2], s=80, c="tab:red", marker="^", label="GT target")
    ax.set_title(meta["instance_uid"])
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_reprojection_overlay(sample: Dict, path: Path, max_points: int = 2000) -> None:
    meta = sample["metadata"]
    instance_path = Path(meta["paths"]["instance_path"])
    cameras, images, points = read_sparse_model(instance_path / "sparse" / "0", with_points=True, max_points=max_points)
    target_id = meta["target_image_id"]
    rec = images[target_id]
    camera = cameras[rec.camera_id]
    image_path = Path(meta["paths"]["target_image"])
    img = Image.open(image_path).convert("RGB")
    raw_w, raw_h = img.size
    img_proc = img.resize((raw_w, raw_h))
    draw = ImageDraw.Draw(img_proc)
    k = camera_intrinsics_matrix(camera)
    r = qvec_to_rotmat(rec.qvec)
    t = rec.tvec
    count = 0
    for point in points.values():
        xc = r @ point.xyz + t
        if xc[2] <= 1e-8:
            continue
        uvw = k @ (xc / xc[2])
        x, y = float(uvw[0]), float(uvw[1])
        if 0 <= x < raw_w and 0 <= y < raw_h:
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 0, 0))
            count += 1
            if count >= 300:
                break
    img_proc.thumbnail((512, 512), Image.Resampling.BILINEAR)
    labeled = _label_tile(img_proc, f"Sparse reprojection diagnostic: {meta['target_frame_id']} | points drawn {count}", height=42)
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args(argv)
    processed = args.processed_root or (args.root / "processed_plan_b")
    manifest = args.manifest or (processed / "manifests" / "preview_episodes.jsonl")
    report_dir = args.report_dir
    panels = report_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest)
    ds = MVImgNetEpisodeDataset(manifest, split="preview", fixed_episodes=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=mvimgnet_collate)
    links = []
    for idx, batch in enumerate(loader):
        if idx >= args.max_samples:
            break
        sample = {
            "model_inputs": {k: (v[0] if torch.is_tensor(v) else v[0]) for k, v in batch["model_inputs"].items()},
            "supervision": {k: v[0] for k, v in batch["supervision"].items()},
            "metadata": batch["metadata"][0],
        }
        raw_path = panels / f"raw_multiview_{idx:03d}.png"
        batch_path = panels / f"dataset_batch_{idx:03d}.png"
        traj_path = panels / f"trajectory_{idx:03d}.png"
        reproj_path = panels / f"reprojection_{idx:03d}.png"
        make_raw_multiview_panel(rows[idx], raw_path)
        make_batch_panel(sample, batch_path)
        make_trajectory(sample, traj_path)
        if idx < 3:
            make_reprojection_overlay(sample, reproj_path)
        links.append((idx, sample["metadata"], raw_path, batch_path, traj_path, reproj_path if idx < 3 else None))

    html_lines = [
        "<!doctype html><meta charset='utf-8'><title>MVImgNet2 Plan B Preview</title>",
        "<style>body{font-family:sans-serif;margin:24px} img{max-width:100%;border:1px solid #ccc;margin:8px 0} section{margin-bottom:32px}</style>",
        "<h1>MVImgNet2.0 Plan B Preview</h1>",
        "<p>Panels use real dataset images and real COLMAP metadata. GT Target is ground truth, not a generated result.</p>",
    ]
    for idx, meta, raw, batch_path, traj, reproj in links:
        html_lines.append("<section>")
        html_lines.append(f"<h2>{idx:03d} {html.escape(meta['instance_uid'])}</h2>")
        html_lines.append(f"<p>{html.escape(str(meta.get('class_name')))} | target frame {html.escape(meta['target_frame_id'])}</p>")
        for label, p in [("Raw multiview", raw), ("Dataset batch", batch_path), ("Camera trajectory", traj), ("Reprojection", reproj)]:
            if p is None:
                continue
            html_lines.append(f"<h3>{label}</h3><a href='{p.relative_to(report_dir)}'><img src='{p.relative_to(report_dir)}'></a>")
        html_lines.append("</section>")
    (report_dir / "preview.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    print(json.dumps({"preview_html": str(report_dir / "preview.html"), "panels": str(panels), "samples": len(links)}, indent=2))


if __name__ == "__main__":
    main()
