"""CLI for MVImgNet2.0 Plan B pilot extraction, audit and manifests."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

from .audit import (
    archive_inventory,
    extract_pilot_instances,
    fixed_episode,
    read_category_map,
    scan_extracted_instances,
    split_instances,
    write_audit_markdown,
    write_json,
    write_jsonl,
)


DEFAULT_ROOT = Path("/data1/haksoo/data/MVImgNet2.0")


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--category-map", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--mode", choices=["pilot", "audit"], default="pilot")
    parser.add_argument("--max-pilot-instances", type=int, default=12)
    parser.add_argument("--max-archives-for-pilot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-context", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    return parser


def write_catalog(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class_id",
                "class_name",
                "instance_id",
                "instance_uid",
                "num_rgb_files",
                "num_registered_images",
                "num_valid_views",
                "mask_files",
                "text_available",
                "instance_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})


def main(argv=None):
    args = build_argparser().parse_args(argv)
    root = args.root
    raw_root = args.raw_root or (root / "raw")
    processed = args.processed_root or (root / "processed_plan_b")
    reports_root = args.reports_root or (root / "reports_plan_b")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    report_dir = reports_root / run_id
    manifests = processed / "manifests"
    pilot_root = processed / "extracted_pilot"
    category_map_path = args.category_map or (root / "repo" / "mvimgnet_category.txt")
    category_map = read_category_map(category_map_path)

    inventory = archive_inventory(raw_root)
    if args.mode == "pilot" and not any(pilot_root.glob("*/*/images")):
        archives = [Path(a["name"]) for a in []]
        archive_paths = [raw_root / a["name"] for a in inventory["archives"][: args.max_archives_for_pilot]]
        selected = extract_pilot_instances(archive_paths, pilot_root, max_instances=args.max_pilot_instances)
    else:
        selected = []

    roots = []
    extracted = root / "extracted"
    if extracted.exists():
        roots.append(extracted)
    if pilot_root.exists():
        roots.append(pilot_root)
    rows, rejected, summary = scan_extracted_instances(roots, category_map)
    assigned = split_instances(rows, seed=args.seed)
    by_split = {split: [r for r in assigned if r["split"] == split] for split in ["train", "val", "test"]}
    preview_rows = [fixed_episode(r, seed=args.seed, k_context=args.k_context) for r in assigned[: args.max_pilot_instances]]
    val_rows = [fixed_episode(r, seed=args.seed, k_context=args.k_context) for r in by_split["val"]]
    test_rows = [fixed_episode(r, seed=args.seed, k_context=args.k_context) for r in by_split["test"]]

    processed.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": "mvimgnet2",
        "source_root": str(root),
        "raw_root": str(raw_root),
        "processed_root": str(processed),
        "reports_root": str(reports_root),
        "run_id": run_id,
        "k_context": args.k_context,
        "target_views": 1,
        "image_size": args.image_size,
        "target_vae_latent_shape": [4, 32, 32],
        "latent_patch_size": 4,
        "latent_tokens": 64,
        "latent_token_dim": 64,
        "preprocessing": "aspect_ratio_resize_letterbox",
        "translation_normalization": "context_median_baseline",
        "split_source": "local_instance_split",
        "split_seed": args.seed,
        "split_ratios": [0.9, 0.05, 0.05],
        "text_policy": "preserve_raw_if_present_else_unavailable",
        "target_supervision_not_condition": True,
        "selected_pilot_prefixes": selected,
    }
    write_json(processed / "inventory.json", inventory)
    write_json(processed / "DATASET_AUDIT.json", {"inventory": inventory, "summary": summary, "config": config})
    write_jsonl(processed / "instances.jsonl", assigned)
    write_jsonl(processed / "rejected.jsonl", rejected)
    write_jsonl(processed / "split_assignment.jsonl", assigned)
    for split, split_rows in by_split.items():
        write_jsonl(manifests / f"{split}_instances.jsonl", split_rows)
    write_jsonl(manifests / "preview_episodes.jsonl", preview_rows)
    write_jsonl(manifests / "val_episodes.jsonl", val_rows)
    write_jsonl(manifests / "test_episodes.jsonl", test_rows)
    with open(processed / "data_config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    report = {"inventory": inventory, "summary": summary, "config": config}
    write_json(report_dir / "DATASET_AUDIT.json", report)
    write_audit_markdown(report_dir / "DATASET_AUDIT.md", report)
    write_catalog(report_dir / "SAMPLE_CATALOG.csv", assigned)
    (report_dir / "BATCH_CONTRACT.md").write_text(
        "# MVImgNet2.0 Batch Contract\n\n"
        "model_inputs contains context_images, context cameras, target camera and text fields only.\n"
        "supervision contains target_image, target_object_mask and target_pixel_valid. Target RGB/mask/latent are not model_inputs.\n"
        "Shapes: context_images [B,K,3,256,256], target_image [B,3,256,256], context_T_rel [B,K,4,4], target_T_rel [B,4,4].\n",
        encoding="utf-8",
    )
    (report_dir / "NEXT_TRAINING_HANDOFF.md").write_text(
        "# Next Training Handoff\n\n"
        "- Dataset class: `plan_b.mvimgnet.dataset.MVImgNetEpisodeDataset`.\n"
        "- Collate: `plan_b.mvimgnet.dataset.mvimgnet_collate`.\n"
        "- Manifests: `processed_plan_b/manifests/{train,val,test}_instances.jsonl` and fixed episode manifests.\n"
        "- Target RGB to latent is pending for the next training step; expected latent shape is [4,32,32] and patch tokens [64,64].\n"
        "- Camera adapter from continuous relative pose/intrinsics into RDT is pending.\n"
        "- Stage-1 trainer is pending and was not run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"processed": str(processed), "report_dir": str(report_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
