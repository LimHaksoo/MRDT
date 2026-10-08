"""Real-IAD dataset adapter.

Supports two layouts:

1. official layout::

    Real-IAD/
      realiad_256|realiad_512|realiad_1024|realiad_raw/<object_name>/**.jpg|png
      realiad_jsons/<object_name>.json          (default metadata)
      realiad_jsons_sv/..., realiad_jsons_fuiad_*/...  (variants, selectable)

2. reorganized layout::

    real_iad/
      <object_name>/
        OK/<sample_id>/*.jpg
        NG/<defect_name>/<sample_id>/*.jpg
    (masks: PNG files next to images, or in a parallel ground-truth tree)

JSON metadata is preferred over filename guessing whenever available. The JSON
parser is tolerant: it accepts either ``{"train": [...], "test": [...]}`` or a
flat list, and probes several key aliases per field. When parsing fails, the
first few entries are printed so the layout can be diagnosed quickly.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rescene.constants import normalize_defect_label
from rescene.data.adapters.base import (AdapterError, DatasetScan, ImageRecord,
                                        infer_view_id, is_image_file)

RESOLUTION_DIRS = {
    "256": "realiad_256",
    "512": "realiad_512",
    "1024": "realiad_1024",
    "raw": "realiad_raw",
}

_IMG_KEYS = ("image_path", "img_path", "image", "img", "path", "file_name", "filename")
_MASK_KEYS = ("mask_path", "mask", "gt_path", "gt", "label_path", "anomaly_mask_path")
_LABEL_KEYS = ("anomaly_class", "defect_class", "defect", "label", "anomaly_type", "defect_type")
_VIEW_KEYS = ("view_id", "view", "camera_id", "camera", "cam", "camera_config")
_SAMPLE_KEYS = ("sample_id", "instance_id", "object_id", "item_id", "group_id")

_VIEW_TOKEN = re.compile(r"([_\-.])C([1-9])(?=[_\-.]|$)", re.IGNORECASE)


def _sample_key_from_stem(stem: str) -> str:
    """Group the five views of one physical item: drop the view token from the stem."""
    no_view = _VIEW_TOKEN.sub(r"\1", stem)
    no_view = re.sub(r"(view|cam(?:era)?)[_\-]?\d+", "", no_view, flags=re.IGNORECASE)
    no_view = re.sub(r"[_\-](\d)$", "", no_view)
    no_view = re.sub(r"[_\-.]+$", "", no_view)
    return no_view or stem

def _view_from_stem(stem: str) -> Optional[int]:
    """0-based view index from a Real-IAD stem, e.g.
    'audiojack_0001_OK_C4_20231021130235' -> 3.
    Uses the C1..C5 camera token, falling back to base.infer_view_id."""
    m = _VIEW_TOKEN.search(stem)
    if m:
        return int(m.group(2)) - 1  # C1->0 ... C5->4
    return infer_view_id(stem)


def _first_key(d: dict, keys) -> Optional[object]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _parse_view(entry: dict, stem: str) -> Optional[int]:
    raw = _first_key(entry, _VIEW_KEYS)
    if raw is not None:
        if isinstance(raw, int):
            return raw if raw < 5 else raw - 1  # tolerate 1-based ints > 4
        m = re.search(r"(\d+)", str(raw))
        if m:
            v = int(m.group(1))
            # C1..C5 style strings are 1-based
            if re.search(r"^c\d+$", str(raw).strip(), re.IGNORECASE) or v > 4:
                v -= 1
            return max(v, 0)
    return _view_from_stem(stem)


def detect_real_iad_layout(root: Path) -> Tuple[str, dict]:
    """Return (layout_name, info). layout_name in {"official", "reorganized"}."""
    res_dirs = {k: root / v for k, v in RESOLUTION_DIRS.items() if (root / v).is_dir()}
    json_dirs = sorted(p.name for p in root.glob("realiad_jsons*") if p.is_dir())
    if res_dirs:
        return "official", {"resolutions": sorted(res_dirs), "json_dirs": json_dirs}

    # reorganized: <object>/OK and/or <object>/NG right under root (or under root/real_iad)
    for base in (root, root / "real_iad"):
        if not base.is_dir():
            continue
        objs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith("real_iad_json")]
        ok_ng = [p for p in objs if (p / "OK").is_dir() or (p / "NG").is_dir()]
        if ok_ng:
            return "reorganized", {"base": str(base), "num_objects": len(ok_ng)}
    raise AdapterError(
        f"Could not detect a Real-IAD layout under {root}.\n"
        f"Expected either official subfolders ({', '.join(RESOLUTION_DIRS.values())} + realiad_jsons/) "
        f"or a reorganized layout (<object>/OK|NG/...).\n"
        f"Top-level entries found: {sorted(p.name for p in root.iterdir())[:30]}"
    )


def _load_json_entries(json_path: Path, scan: DatasetScan) -> List[Tuple[dict, Optional[str]]]:
    """Return list of (entry, split_hint)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: List[Tuple[dict, Optional[str]]] = []
    if isinstance(data, dict):
        for split in ("train", "val", "validation", "test"):
            for e in data.get(split, []) or []:
                if isinstance(e, dict):
                    out.append((e, "val" if split == "validation" else split))
        if not out:
            # maybe a dict of lists keyed by something else, or a single flat dict of entries
            for key, val in data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    out.extend((e, None) for e in val if isinstance(e, dict))
    elif isinstance(data, list):
        out = [(e, None) for e in data if isinstance(e, dict)]
    if not out:
        scan.warnings.append(f"{json_path.name}: no entries could be parsed; "
                             f"raw type={type(data).__name__}")
    return out


def _resolve_image(rel: str, search_dirs: List[Path]) -> Optional[Path]:
    rel = str(rel).replace("\\", "/").lstrip("/")
    for base in search_dirs:
        p = base / rel
        if p.is_file():
            return p
    return None


def _scoped_official_sample_id(
    base_sample_id: str,
    image_rel: str,
    is_anomaly: bool,
    defect: Optional[str],
) -> str:
    """Namespace repeated Real-IAD sample folders by OK/NG branch.

    Official JSON paths commonly look like ``OK/S0004/file.jpg`` or
    ``NG/BX/S0001/file.jpg``.  ``S0001`` is reused across defect folders, so
    it is not globally unique within a category and must not be treated as an
    exact normal/anomaly pair.
    """
    parts = Path(str(image_rel).replace("\\", "/")).parts
    upper = [part.upper() for part in parts]

    def clean(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())

    if "NG" in upper:
        idx = upper.index("NG")
        defect_scope = parts[idx + 1] if idx + 1 < len(parts) - 1 else (defect or "unknown")
        sample_scope = parts[idx + 2] if idx + 2 < len(parts) - 1 else base_sample_id
        return f"NG__{clean(defect_scope)}__{clean(sample_scope)}"

    if "OK" in upper:
        idx = upper.index("OK")
        sample_scope = parts[idx + 1] if idx + 1 < len(parts) - 1 else base_sample_id
        return f"OK__{clean(sample_scope)}"

    # Metadata variants without OK/NG paths keep their declared sample id.
    return clean(base_sample_id)


def scan_real_iad(
    root: str,
    resolution: str = "1024",
    json_dir: Optional[str] = None,
    verbose: bool = True,
) -> DatasetScan:
    root_p = Path(root)
    if not root_p.is_dir():
        raise AdapterError(f"Real-IAD root does not exist: {root}")
    layout, info = detect_real_iad_layout(root_p)
    scan = DatasetScan(dataset="real_iad", root=str(root_p), layout=layout, notes=info)

    if layout == "official":
        _scan_official(root_p, resolution, json_dir, scan, verbose)
    else:
        _scan_reorganized(Path(info["base"]), scan)

    if not scan.records:
        raise AdapterError(
            f"Real-IAD adapter found 0 usable images under {root} (layout={layout}). "
            f"Skipped {len(scan.skipped)} files. First skips: {scan.skipped[:10]}"
        )
    if verbose:
        _print_examples(scan)
    return scan


def _scan_official(root: Path, resolution: str, json_dir: Optional[str],
                   scan: DatasetScan, verbose: bool) -> None:
    if resolution not in RESOLUTION_DIRS:
        raise AdapterError(f"--real-iad-resolution must be one of {list(RESOLUTION_DIRS)}, "
                           f"got {resolution!r}")
    img_root = root / RESOLUTION_DIRS[resolution]
    if not img_root.is_dir():
        avail = [k for k, v in RESOLUTION_DIRS.items() if (root / v).is_dir()]
        raise AdapterError(f"{img_root} not found. Available resolutions: {avail}")

    if json_dir:
        jdir = Path(json_dir)
        if not jdir.is_absolute():
            jdir = root / jdir
    else:
        jdir = root / "realiad_jsons"
    scan.notes["image_root"] = str(img_root)
    scan.notes["json_dir"] = str(jdir) if jdir.is_dir() else None

    if jdir.is_dir() and any(jdir.glob("*.json")):
        _scan_official_with_json(root, img_root, jdir, scan, verbose)
    else:
        scan.warnings.append(
            f"No JSON metadata found at {jdir}; falling back to filename-based parsing."
        )
        _scan_image_tree(img_root, scan)


def _scan_official_with_json(root: Path, img_root: Path, jdir: Path,
                             scan: DatasetScan, verbose: bool) -> None:
    json_files = sorted(jdir.glob("*.json"))
    printed_example = False
    for jf in json_files:
        category = jf.stem
        cat_dir = img_root / category
        search_dirs = [img_root, cat_dir, root]
        entries = _load_json_entries(jf, scan)
        for entry, split_hint in entries:
            img_rel = _first_key(entry, _IMG_KEYS)
            if img_rel is None:
                if not printed_example:
                    print(f"[real_iad] WARNING: cannot find an image-path key in JSON entry "
                          f"from {jf.name}. Entry example:\n  {json.dumps(entry)[:500]}")
                    printed_example = True
                scan.skipped.append(f"{jf.name}: entry without image path key")
                continue
            img_path = _resolve_image(str(img_rel), search_dirs)
            if img_path is None:
                scan.skipped.append(f"missing image: {img_rel}")
                continue

            raw_label = _first_key(entry, _LABEL_KEYS)
            raw_label_s = str(raw_label) if raw_label is not None else None
            is_anomaly = raw_label_s is not None and raw_label_s.upper() not in ("OK", "GOOD", "NORMAL")
            # some variants keep an explicit flag
            for flag_key in ("is_anomaly", "anomaly", "is_ng"):
                if flag_key in entry:
                    is_anomaly = bool(entry[flag_key])
                    break
            if not is_anomaly and "/NG/" in str(img_rel).replace("\\", "/"):
                is_anomaly = True

            mask_path = None
            mask_rel = _first_key(entry, _MASK_KEYS)
            if is_anomaly and mask_rel:
                mp = _resolve_image(str(mask_rel), search_dirs)
                if mp is None:
                    scan.skipped.append(f"missing mask: {mask_rel} (image kept without mask)")
                else:
                    mask_path = str(mp)

            stem = Path(str(img_rel)).stem
            view_id = _parse_view(entry, stem)
            if view_id is None:
                scan.skipped.append(f"cannot infer view id: {img_rel}")
                continue
            sample_raw = _first_key(entry, _SAMPLE_KEYS)
            base_sample_id = (
                str(sample_raw) if sample_raw is not None else _sample_key_from_stem(stem)
            )
            sample_id = _scoped_official_sample_id(
                base_sample_id, str(img_rel), is_anomaly, raw_label_s
            )

            label = normalize_defect_label(raw_label_s) if is_anomaly else None
            scan.records.append(ImageRecord(
                dataset="real_iad", category=category, sample_id=sample_id,
                view_id=view_id, image_path=str(img_path), is_anomaly=is_anomaly,
                mask_path=mask_path, defect_label=label, raw_defect=raw_label_s,
                split_hint=split_hint,
            ))


def _scan_reorganized(base: Path, scan: DatasetScan) -> None:
    for obj_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if obj_dir.name.startswith("real_iad_json"):
            continue
        ok_dir, ng_dir = obj_dir / "OK", obj_dir / "NG"
        if not ok_dir.is_dir() and not ng_dir.is_dir():
            continue
        if ok_dir.is_dir():
            for sample_dir in sorted(p for p in ok_dir.iterdir() if p.is_dir()):
                _collect_sample_dir(scan, obj_dir.name, sample_dir, is_anomaly=False,
                                    defect=None)
            _collect_flat_images(scan, obj_dir.name, ok_dir, is_anomaly=False, defect=None)
        if ng_dir.is_dir():
            for defect_dir in sorted(p for p in ng_dir.iterdir() if p.is_dir()):
                for sample_dir in sorted(p for p in defect_dir.iterdir() if p.is_dir()):
                    _collect_sample_dir(scan, obj_dir.name, sample_dir, is_anomaly=True,
                                        defect=defect_dir.name)
                _collect_flat_images(scan, obj_dir.name, defect_dir, is_anomaly=True,
                                     defect=defect_dir.name)


def _scoped_reorganized_sample_id(base_sample_id: str, is_anomaly: bool,
                                    defect: Optional[str]) -> str:
    """Return a sample id that is unique within a reorganized Real-IAD category.

    Reorganized trees frequently reuse folder names such as ``S0001`` under
    ``OK/`` and under several independent ``NG/<defect>/`` directories. Using
    only ``S0001`` merges unrelated physical samples, creates duplicate episode
    ids, and can falsely mark an OK/NG pair as an exact ``paired`` example.
    Namespace the id by source branch while preserving all views in one folder
    as a single multi-view set.
    """
    if is_anomaly:
        defect_scope = str(defect or "unknown").strip().replace("/", "_")
        return f"NG__{defect_scope}__{base_sample_id}"
    return f"OK__{base_sample_id}"


def _collect_flat_images(scan, category, folder: Path, is_anomaly: bool, defect):
    """Images directly inside OK/ or NG/<defect>/ (no per-sample folder)."""
    files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p.name)]
    if not files:
        return
    imgs, masks = _split_images_and_masks(files)
    for p in imgs:
        base_sample_id = _sample_key_from_stem(p.stem)
        sample_id = _scoped_reorganized_sample_id(
            base_sample_id, is_anomaly=is_anomaly, defect=defect
        )
        _append_record(scan, category, sample_id=sample_id,
                       file=p, is_anomaly=is_anomaly, defect=defect, masks=masks)


def _collect_sample_dir(scan, category, sample_dir: Path, is_anomaly: bool, defect):
    files = [p for p in sample_dir.rglob("*") if p.is_file() and is_image_file(p.name)]
    if not files:
        scan.skipped.append(f"empty sample dir: {sample_dir}")
        return
    imgs, masks = _split_images_and_masks(files)
    sample_id = _scoped_reorganized_sample_id(
        sample_dir.name, is_anomaly=is_anomaly, defect=defect
    )
    for p in imgs:
        _append_record(scan, category, sample_id=sample_id, file=p,
                       is_anomaly=is_anomaly, defect=defect, masks=masks)


def _split_images_and_masks(files: List[Path]) -> Tuple[List[Path], Dict[str, Path]]:
    """Heuristic: files whose stem ends with _mask/_gt (or that are PNG twins of a JPG)
    are masks."""
    masks: Dict[str, Path] = {}
    imgs: List[Path] = []
    jpg_stems = {p.stem for p in files if p.suffix.lower() in (".jpg", ".jpeg")}
    for p in files:
        stem_l = p.stem.lower()
        if stem_l.endswith(("_mask", "_gt", "-mask", "-gt")):
            base = re.sub(r"[_\-](mask|gt)$", "", p.stem, flags=re.IGNORECASE)
            masks[base] = p
        elif p.suffix.lower() == ".png" and p.stem in jpg_stems:
            masks[p.stem] = p  # PNG twin of JPG image = pixel mask
        else:
            imgs.append(p)
    return imgs, masks


def _append_record(scan, category, sample_id, file: Path, is_anomaly, defect,
                   masks: Dict[str, Path]):
    view_id =  _view_from_stem(file.stem)     # ← 기존: infer_view_id(file.stem)
    if view_id is None:
        scan.skipped.append(f"cannot infer view id: {file}")
        return
    mask_path = masks.get(file.stem)
    label = normalize_defect_label(defect) if is_anomaly else None
    scan.records.append(ImageRecord(
        dataset="real_iad", category=category, sample_id=sample_id, view_id=view_id,
        image_path=str(file), is_anomaly=is_anomaly,
        mask_path=str(mask_path) if mask_path else None,
        defect_label=label, raw_defect=defect,
    ))


def _scan_image_tree(img_root: Path, scan: DatasetScan) -> None:
    """Last-resort scan of realiad_<res>/<object>/** without JSON metadata."""
    for obj_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        files = [p for p in obj_dir.rglob("*") if p.is_file() and is_image_file(p.name)]
        imgs, masks = _split_images_and_masks(files)
        for p in imgs:
            rel_parts = [q.upper() for q in p.relative_to(obj_dir).parts]
            is_anom = "NG" in rel_parts
            defect = None
            if is_anom:
                ng_idx = rel_parts.index("NG")
                orig_parts = p.relative_to(obj_dir).parts
                if ng_idx + 1 < len(orig_parts) - 1:
                    defect = orig_parts[ng_idx + 1]
            rel_path = str(p.relative_to(obj_dir))
            sample_id = _scoped_official_sample_id(
                _sample_key_from_stem(p.stem), rel_path, is_anom, defect
            )
            _append_record(scan, obj_dir.name,
                           sample_id=sample_id, file=p,
                           is_anomaly=is_anom, defect=defect, masks=masks)


def _print_examples(scan: DatasetScan) -> None:
    print(f"[real_iad] layout={scan.layout} images={len(scan.records)} "
          f"skipped={len(scan.skipped)}")
    for r in scan.records[:3]:
        print(f"  example: cat={r.category} sample={r.sample_id} view={r.view_id} "
              f"anom={r.is_anomaly} label={r.defect_label} mask={'Y' if r.mask_path else 'N'}\n"
              f"           {r.image_path}")
    for s in scan.skipped[:5]:
        print(f"  skipped: {s}")
