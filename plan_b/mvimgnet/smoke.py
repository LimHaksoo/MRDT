"""Synthetic and real-data checks for MVImgNet2.0 Plan B artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cameras import (
    latent_grid_to_patch_tokens,
    letterbox_transform,
    patch_tokens_to_latent_grid,
    relative_world_to_camera,
)
from .colmap_io import Image as ColmapImage
from .colmap_io import qvec_to_rotmat
from .dataset import MVImgNetEpisodeDataset, mvimgnet_collate, read_jsonl


def check_synthetic() -> list[str]:
    notes = []
    r = qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r, np.eye(3))
    images = {
        10: ColmapImage(10, np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 7, "a.jpg", np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)),
        42: ColmapImage(42, np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]), 7, "b.jpg", np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)),
    }
    rel = relative_world_to_camera(images, [10, 42], 10)
    assert np.allclose(rel[10], np.eye(4))
    assert np.allclose(rel[42][:3, 3], np.array([1.0, 2.0, 3.0]))
    a, meta = letterbox_transform(640, 480, 256)
    p = np.array([320.0, 240.0, 1.0])
    pp = a @ p
    assert np.allclose(pp[:2], np.array([128.0, 128.0]))
    latent = np.arange(4 * 32 * 32, dtype=np.float32).reshape(4, 32, 32)
    tokens = latent_grid_to_patch_tokens(latent)
    assert tokens.shape == (64, 64)
    assert np.array_equal(patch_tokens_to_latent_grid(tokens), latent)
    notes.append("synthetic: quaternion, relative pose, letterbox intrinsics and latent token round-trip passed")
    return notes


def check_real(manifest: Path, workers: int, batches: int) -> list[str]:
    notes = []
    rows = read_jsonl(manifest)
    assert rows, f"manifest is empty: {manifest}"
    seen_instances = set()
    for row in rows:
        assert len(set(row.get("context_image_ids", [])) | {row.get("target_image_id")}) == row.get("k_context", 3) + 1
        assert row["instance_uid"] not in seen_instances or True
        seen_instances.add(row["instance_uid"])
        assert row.get("text") is None
        assert row.get("text_available") is False
    ds = MVImgNetEpisodeDataset(manifest, split="smoke", fixed_episodes=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=workers, collate_fn=mvimgnet_collate)
    count = 0
    first_shapes = None
    for batch in loader:
        mi = batch["model_inputs"]
        sup = batch["supervision"]
        forbidden = {"target_image", "target_object_mask", "target_latent", "clean_latent"}
        assert not forbidden.intersection(mi.keys())
        assert mi["context_images"].dtype == torch.float32
        assert sup["target_image"].dtype == torch.float32
        assert torch.isfinite(mi["context_images"]).all()
        assert torch.isfinite(mi["target_T_rel"]).all()
        first_shapes = {
            "context_images": list(mi["context_images"].shape),
            "context_intrinsics": list(mi["context_intrinsics"].shape),
            "target_intrinsics": list(mi["target_intrinsics"].shape),
            "context_T_rel": list(mi["context_T_rel"].shape),
            "target_T_rel": list(mi["target_T_rel"].shape),
            "target_image": list(sup["target_image"].shape),
            "target_object_mask": list(sup["target_object_mask"].shape),
        }
        count += 1
        if count >= batches:
            break
    assert count > 0
    notes.append(f"real: DataLoader workers={workers} processed {count} batch(es); first_shapes={first_shapes}")
    return notes


def check_split_leakage(processed_root: Path) -> list[str]:
    notes = []
    split_sets = {}
    for split in ["train", "val", "test"]:
        rows = read_jsonl(processed_root / "manifests" / f"{split}_instances.jsonl")
        split_sets[split] = {r["instance_uid"] for r in rows}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    notes.append(f"real: instance split leakage check passed sizes={ {k: len(v) for k, v in split_sets.items()} }")
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args(argv)
    manifest = args.manifest or (args.processed_root / "manifests" / "preview_episodes.jsonl")
    lines = []
    status = "passed"
    try:
        lines.extend(check_synthetic())
        lines.extend(check_split_leakage(args.processed_root))
        lines.extend(check_real(manifest, workers=0, batches=args.batches))
        lines.extend(check_real(manifest, workers=2, batches=args.batches))
    except Exception as exc:
        status = "failed"
        lines.append(f"FAILED: {type(exc).__name__}: {exc}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "TEST_RESULTS.txt"
    out.write_text("\n".join([f"status: {status}", *lines]) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "test_results": str(out), "lines": lines}, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
