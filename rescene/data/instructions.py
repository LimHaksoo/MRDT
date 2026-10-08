"""Deterministic pseudo-instruction generation from defect metadata and masks.

The training datasets do not ship human-written edit commands, so we build
instructions from:
  * the defect label / defect family
  * the target view id
  * mask geometry (location, size, shape)

Everything is deterministic given the episode key, so re-running data
preparation yields identical manifests. Hooks for LLM paraphrase expansion are
provided via ``expanded_instructions`` (left empty by default) and via
JSON-provided overrides (see ``prepare_rescene_data.py --instruction-overrides``).
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from rescene.constants import DEFECT_FAMILY_PHRASES, DEFAULT_DEFECT_FAMILY


@dataclass
class MaskStats:
    """Simple geometric statistics of a binary anomaly mask."""
    area_ratio: float = 0.0
    cx: float = 0.5           # normalized centroid x in [0, 1]
    cy: float = 0.5           # normalized centroid y in [0, 1]
    aspect_ratio: float = 1.0  # bbox long side / short side
    bbox: tuple = (0.0, 0.0, 1.0, 1.0)  # normalized (x0, y0, x1, y1)
    valid: bool = False
    location_words: List[str] = field(default_factory=list)
    shape_words: List[str] = field(default_factory=list)


def compute_mask_stats(mask: np.ndarray) -> MaskStats:
    """mask: (H, W) array; anything > 127 (uint8) or > 0.5 (float) is foreground."""
    if mask is None:
        return MaskStats()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == np.uint8:
        fg = mask > 127
    else:
        fg = mask > 0.5
    h, w = fg.shape
    area = int(fg.sum())
    if area == 0 or h == 0 or w == 0:
        return MaskStats()

    ys, xs = np.nonzero(fg)
    cy = float(ys.mean()) / max(h - 1, 1)
    cx = float(xs.mean()) / max(w - 1, 1)
    x0, x1 = float(xs.min()) / w, float(xs.max() + 1) / w
    y0, y1 = float(ys.min()) / h, float(ys.max() + 1) / h
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    aspect = max(bw, bh) / min(bw, bh)
    area_ratio = area / float(h * w)

    stats = MaskStats(
        area_ratio=area_ratio, cx=cx, cy=cy, aspect_ratio=aspect,
        bbox=(x0, y0, x1, y1), valid=True,
    )
    stats.location_words = _location_words(stats)
    stats.shape_words = _shape_words(stats)
    return stats


def _location_words(s: MaskStats) -> List[str]:
    words = []
    near_left = s.cx < 0.33
    near_right = s.cx > 0.67
    near_top = s.cy < 0.33
    near_bottom = s.cy > 0.67
    if (near_left or near_right) and (near_top or near_bottom):
        words.append("corner")
    elif near_left:
        words.append("left")
    elif near_right:
        words.append("right")
    elif near_top:
        words.append("top")
    elif near_bottom:
        words.append("bottom")
    else:
        words.append("center")
    # edge proximity of the bbox itself
    x0, y0, x1, y1 = s.bbox
    if min(x0, y0, 1 - x1, 1 - y1) < 0.05 and "corner" not in words:
        words.append("edge")
    return words


def _shape_words(s: MaskStats) -> List[str]:
    words = []
    if s.aspect_ratio >= 3.0:
        words.append("thin")
    else:
        words.append("blob-like")
    if s.area_ratio < 0.01:
        words.append("small")
    elif s.area_ratio > 0.10:
        words.append("large")
    return words


def _stable_int(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def _location_phrase(stats: MaskStats) -> str:
    if not stats.valid or not stats.location_words:
        return "on the visible surface"
    loc = stats.location_words[0]
    if loc == "center":
        return "near the center"
    if loc == "corner":
        return "near a corner"
    if loc in ("left", "right", "top", "bottom"):
        return f"on the {loc} side"
    return "on the visible surface"


def _size_adjective(stats: MaskStats) -> str:
    if not stats.valid:
        return ""
    if "small" in stats.shape_words:
        return "small "
    if "large" in stats.shape_words:
        return "large "
    return ""


def _shape_adjective(stats: MaskStats) -> str:
    if stats.valid and "thin" in stats.shape_words:
        return "thin "
    return ""


def generate_instruction(
    episode_key: str,
    defect_label: Optional[str],
    defect_family: Optional[str],
    view_id: Optional[int],
    mask_stats: Optional[MaskStats],
    has_support: bool = False,
) -> dict:
    """Return {"pseudo_instruction", "simplified_instruction", "expanded_instructions"}.

    Deterministic in ``episode_key``.
    """
    stats = mask_stats or MaskStats()
    family = defect_family or DEFAULT_DEFECT_FAMILY
    family_phrase = DEFECT_FAMILY_PHRASES.get(family, DEFECT_FAMILY_PHRASES[DEFAULT_DEFECT_FAMILY])
    label = (defect_label or "").strip().lower()
    # what noun to name the defect after
    if label and label not in ("defect", "good"):
        defect_phrase = f"{label}-like defect"
        simplified = f"add {label}"
    else:
        defect_phrase = f"{family_phrase} defect"
        simplified = "add a defect"

    size = _size_adjective(stats)
    shape = _shape_adjective(stats)
    loc = _location_phrase(stats)
    view_phrase = f"on view {view_id}" if view_id is not None else "on this view"

    templates = [
        f"add a {size}{shape}{defect_phrase} {loc} {view_phrase}",
        f"add a {size}{defect_phrase} {loc}",
        f"create a {shape}{defect_phrase} {view_phrase}",
        f"add a {family_phrase} anomaly {loc} {view_phrase}",
    ]
    if has_support:
        templates.append(f"apply a defect similar to the support anomaly {loc}")
        templates.append(f"add a {defect_phrase} similar to the support example {view_phrase}")

    idx = _stable_int(episode_key) % len(templates)
    pseudo = " ".join(templates[idx].split())  # collapse double spaces

    expanded: List[str] = []  # hook for later LLM paraphrase expansion
    return {
        "pseudo_instruction": pseudo,
        "simplified_instruction": simplified,
        "expanded_instructions": expanded,
    }
