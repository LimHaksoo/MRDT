"""Common record types and helpers shared by the dataset adapters."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class AdapterError(RuntimeError):
    """Raised when a dataset root cannot be interpreted. The message must
    contain actionable diagnostics (what was found, what was expected)."""


@dataclass
class ImageRecord:
    """One image (a single view of a single physical sample)."""
    dataset: str                   # "real_iad" | "viaduct"
    category: str                  # object / category name (folder name)
    sample_id: str                 # multi-view set id; identical for all views of one set
    view_id: int                   # 0-based camera/view index
    image_path: str
    is_anomaly: bool
    mask_path: Optional[str] = None
    defect_label: Optional[str] = None   # normalized readable label ("scratch", ...)
    raw_defect: Optional[str] = None     # raw code/folder name from the dataset
    split_hint: Optional[str] = None     # dataset-provided split ("train"/"val"/"test") if any

    @property
    def set_key(self) -> str:
        return f"{self.dataset}/{self.category}/{self.sample_id}"


@dataclass
class DatasetScan:
    """Result of scanning one dataset root."""
    dataset: str
    root: str
    layout: str                    # detected layout name
    records: List[ImageRecord] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)     # paths skipped + reason
    warnings: List[str] = field(default_factory=list)
    notes: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        n_anom = sum(1 for r in self.records if r.is_anomaly)
        n_mask = sum(1 for r in self.records if r.mask_path)
        views: Dict[int, int] = {}
        defects: Dict[str, int] = {}
        cats = set()
        sets = set()
        for r in self.records:
            views[r.view_id] = views.get(r.view_id, 0) + 1
            cats.add(r.category)
            sets.add(r.set_key)
            if r.is_anomaly and r.defect_label:
                defects[r.defect_label] = defects.get(r.defect_label, 0) + 1
        return {
            "dataset": self.dataset,
            "root": self.root,
            "layout": self.layout,
            "num_categories": len(cats),
            "num_images": len(self.records),
            "num_normal_images": len(self.records) - n_anom,
            "num_anomaly_images": n_anom,
            "num_masks": n_mask,
            "num_multiview_sets": len(sets),
            "view_id_distribution": {str(k): v for k, v in sorted(views.items())},
            "defect_label_distribution": dict(sorted(defects.items())),
            "num_skipped_files": len(self.skipped),
            "warnings": self.warnings,
            "notes": self.notes,
        }


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# view-id patterns tried in order on the file stem
_VIEW_PATTERNS = [
    re.compile(r"[_\-.]C([1-9])\b", re.IGNORECASE),        # _C1_ / -C3.
    re.compile(r"\bC([1-9])[_\-.]", re.IGNORECASE),        # C1_
    re.compile(r"view[_\-]?(\d+)", re.IGNORECASE),         # view1 / view_1
    re.compile(r"cam(?:era)?[_\-]?(\d+)", re.IGNORECASE),  # camera1 / cam_1
    re.compile(r"[_\-](\d)$"),                             # trailing _0 .. _9
]


def infer_view_id(stem: str, one_based_c: bool = True) -> Optional[int]:
    """Best-effort view id from a filename stem. C1..C5 map to 0..4."""
    for i, pat in enumerate(_VIEW_PATTERNS):
        m = pat.search(stem)
        if m:
            v = int(m.group(1))
            if i <= 1 and one_based_c:
                v -= 1  # C1..C5 -> 0..4
            return v
    return None


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)
