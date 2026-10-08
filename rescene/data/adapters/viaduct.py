"""VIADUCT dataset adapter.

Expected layout (MVTec-style with five views)::

    VIADUCT/
      00<ObjectName>/
        train/good/Image_defect-free_<id>_<view>.png
        val/good/...
        test/good/...
        test/<defect_name>/Image_<defect_name>_<id>_<view>.png
        ground_truth/<defect_name>/<same filename as test image>.png
      01<ObjectName>/ ...

* the trailing ``_<view>`` numeral (0-4) is the camera position;
* the stem without the view suffix identifies a five-view set;
* ``ground_truth`` masks share the exact filename of their test image.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

from rescene.constants import defect_family_of, normalize_defect_label
from rescene.data.adapters.base import (AdapterError, DatasetScan, ImageRecord,
                                        is_image_file)

_CAT_RE = re.compile(r"^(\d+)(.*)$")
_VIEW_SUFFIX_RE = re.compile(r"^(.*)_(\d+)$")

SPLIT_DIRS = ("train", "val", "test")


def parse_category(folder_name: str) -> Tuple[Optional[int], str]:
    """'00ExampleObject' -> (0, '00ExampleObject'). Returns (None, name) if no digits."""
    m = _CAT_RE.match(folder_name)
    if m and m.group(1):
        return int(m.group(1)), folder_name
    return None, folder_name


def split_view_suffix(stem: str) -> Tuple[str, Optional[int]]:
    """'Image_defectA_000123_2' -> ('Image_defectA_000123', 2)."""
    m = _VIEW_SUFFIX_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, None


def detect_viaduct_layout(root: Path) -> dict:
    cats = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any((p / s).is_dir() for s in SPLIT_DIRS):
            cats.append(p)
    if not cats:
        raise AdapterError(
            f"Could not detect a VIADUCT layout under {root}. Expected category folders "
            f"like 00<ObjectName>/ containing train|val|test subfolders.\n"
            f"Top-level entries found: {sorted(q.name for q in root.iterdir())[:30]}"
        )
    return {"categories": [c.name for c in cats]}


def scan_viaduct(root: str, verbose: bool = True) -> DatasetScan:
    root_p = Path(root)
    if not root_p.is_dir():
        raise AdapterError(f"VIADUCT root does not exist: {root}")
    info = detect_viaduct_layout(root_p)
    scan = DatasetScan(dataset="viaduct", root=str(root_p), layout="viaduct_mvtec_style",
                       notes=info)

    for cat_name in info["categories"]:
        cat_dir = root_p / cat_name
        gt_dir = cat_dir / "ground_truth"
        for split in SPLIT_DIRS:
            split_dir = cat_dir / split
            if not split_dir.is_dir():
                continue
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect_name = defect_dir.name
                is_good = defect_name.lower() in ("good", "ok", "normal", "defect-free")
                for img in sorted(defect_dir.iterdir()):
                    if not (img.is_file() and is_image_file(img.name)):
                        continue
                    sample_id, view_id = split_view_suffix(img.stem)
                    if view_id is None:
                        scan.skipped.append(f"no trailing view numeral: {img}")
                        continue
                    mask_path = None
                    if not is_good:
                        cand = gt_dir / defect_name / img.name
                        if cand.is_file():
                            mask_path = str(cand)
                        else:
                            # some releases use <stem>_mask.png
                            alt = gt_dir / defect_name / f"{img.stem}_mask{img.suffix}"
                            if alt.is_file():
                                mask_path = str(alt)
                            else:
                                scan.skipped.append(f"missing ground_truth mask for {img}")
                    label = None if is_good else normalize_defect_label(defect_name)
                    scan.records.append(ImageRecord(
                        dataset="viaduct", category=cat_name, sample_id=sample_id,
                        view_id=view_id, image_path=str(img), is_anomaly=not is_good,
                        mask_path=mask_path,
                        defect_label=label, raw_defect=None if is_good else defect_name,
                        split_hint=split,
                    ))
    if not scan.records:
        raise AdapterError(
            f"VIADUCT adapter found 0 usable images under {root}. "
            f"Skipped {len(scan.skipped)} files. First skips: {scan.skipped[:10]}"
        )
    if verbose:
        print(f"[viaduct] categories={len(info['categories'])} images={len(scan.records)} "
              f"skipped={len(scan.skipped)}")
        for r in scan.records[:3]:
            print(f"  example: cat={r.category} sample={r.sample_id} view={r.view_id} "
                  f"anom={r.is_anomaly} label={r.defect_label} "
                  f"mask={'Y' if r.mask_path else 'N'}\n           {r.image_path}")
        for s in scan.skipped[:5]:
            print(f"  skipped: {s}")
    return scan
