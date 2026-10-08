"""Episode construction: turn scanned image records into ReSCENE episodes.

One episode = one anomalous target view (with mask) + a normal reference for
that view + multi-view normal context + optional few-shot support anomalies +
a pseudo natural-language instruction.

Pairing strategies (recorded per episode in ``pairing_strategy``):

* ``paired``      - dataset provides an exact normal counterpart of the target
                    anomaly view (same sample id + view id).
* ``restored``    - a restored normal counterpart precomputed by another model
                    (provided via ``restored_lookup``).
* ``same_category_same_view_reference`` - fallback: a normal image of the same
                    dataset/category/view. This is not pixel-registered. Latent-delta
                    targets contain instance differences; full-latent targets avoid
                    subtraction but remain unpaired conditional generation.
* ``support_application`` - reserved for generation-style training without
                    paired targets (not emitted by this v1 builder).
"""

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from rescene.constants import defect_family_of
from rescene.data.adapters.base import DatasetScan, ImageRecord
from rescene.data.instructions import MaskStats, compute_mask_stats, generate_instruction

PAIRING_WARNING = (
    "No exact paired normal/anomaly images found. Using same-category same-view "
    "normal references. Latent-delta targets may include instance/pose differences; "
    "full-latent targets avoid subtraction but the normal condition and anomaly target "
    "are still unpaired."
)


def _load_mask_stats(mask_path: Optional[str], max_side: int = 256) -> MaskStats:
    if not mask_path:
        return MaskStats()
    try:
        with Image.open(mask_path) as im:
            im = im.convert("L")
            if max(im.size) > max_side:
                scale = max_side / max(im.size)
                im = im.resize((max(1, int(im.width * scale)),
                                max(1, int(im.height * scale))), Image.NEAREST)
            arr = np.asarray(im)
    except Exception as e:  # unreadable mask -> treat as absent
        print(f"[episodes] WARNING: cannot read mask {mask_path}: {e}")
        return MaskStats()
    return compute_mask_stats(arr)


class EpisodeBuilder:
    def __init__(
        self,
        max_support: int = 2,
        seed: int = 0,
        restored_lookup: Optional[Callable[[ImageRecord], Optional[str]]] = None,
        instruction_overrides: Optional[Dict[str, str]] = None,
    ):
        self.max_support = max_support
        self.seed = seed
        self.restored_lookup = restored_lookup
        self.instruction_overrides = instruction_overrides or {}
        self.stats = defaultdict(int)
        self.warnings: List[str] = []

    def build(self, scans: List[DatasetScan], max_episodes: Optional[int] = None
              ) -> List[dict]:
        episodes: List[dict] = []
        for scan in scans:
            episodes.extend(self._build_for_scan(scan))
        episodes.sort(key=lambda e: e["episode_id"])

        # An episode id is used for deterministic support/reference sampling,
        # instruction overrides, logging, and auditability. Duplicate ids must
        # therefore fail fast instead of silently sharing those behaviors.
        seen_ids = set()
        duplicate_ids = []
        for episode in episodes:
            episode_id = episode["episode_id"]
            if episode_id in seen_ids:
                duplicate_ids.append(episode_id)
            else:
                seen_ids.add(episode_id)
        if duplicate_ids:
            examples = sorted(set(duplicate_ids))[:10]
            raise RuntimeError(
                "EpisodeBuilder produced non-unique episode_id values. "
                f"duplicate_lines={len(duplicate_ids)}, examples={examples}. "
                "Check dataset adapter sample_id construction."
            )

        if max_episodes is not None and len(episodes) > max_episodes:
            rng = random.Random(self.seed)
            episodes = rng.sample(episodes, max_episodes)
            episodes.sort(key=lambda e: e["episode_id"])
        if self.stats["pairing_same_category_same_view_reference"] > 0 and \
           self.stats["pairing_paired"] == 0 and self.stats["pairing_restored"] == 0:
            if PAIRING_WARNING not in self.warnings:
                self.warnings.append(PAIRING_WARNING)
        return episodes

    # ------------------------------------------------------------------
    def _build_for_scan(self, scan: DatasetScan) -> List[dict]:
        records = scan.records
        # index structures
        sets: Dict[str, List[ImageRecord]] = defaultdict(list)     # set_key -> records
        normals_by_cat_view: Dict[tuple, List[ImageRecord]] = defaultdict(list)
        anomalies_by_cat: Dict[str, List[ImageRecord]] = defaultdict(list)
        normals_by_set_view: Dict[tuple, ImageRecord] = {}
        cat_ids = {c: i for i, c in enumerate(sorted({r.category for r in records}))}

        for r in records:
            sets[r.set_key].append(r)
            if r.is_anomaly:
                if r.mask_path:
                    anomalies_by_cat[r.category].append(r)
            else:
                normals_by_cat_view[(r.category, r.view_id)].append(r)
                normals_by_set_view[(r.set_key, r.view_id)] = r

        # normal multi-view sets (sets whose every record is normal)
        normal_sets_by_cat: Dict[str, List[str]] = defaultdict(list)
        for key, recs in sets.items():
            if all(not r.is_anomaly for r in recs):
                normal_sets_by_cat[recs[0].category].append(key)
        for v in normal_sets_by_cat.values():
            v.sort()
        for v in normals_by_cat_view.values():
            v.sort(key=lambda r: r.image_path)
        for v in anomalies_by_cat.values():
            v.sort(key=lambda r: r.image_path)

        episodes = []
        for key in sorted(sets):
            recs = sets[key]
            anom_views = [r for r in recs if r.is_anomaly and r.mask_path]
            for target in sorted(anom_views, key=lambda r: r.view_id):
                ep = self._build_episode(
                    scan, target, sets, normals_by_set_view, normals_by_cat_view,
                    normal_sets_by_cat, anomalies_by_cat, cat_ids)
                if ep is not None:
                    episodes.append(ep)
        # count skips of anomalies without masks
        n_no_mask = sum(1 for r in records if r.is_anomaly and not r.mask_path)
        self.stats["anomaly_views_without_mask"] += n_no_mask
        return episodes

    # ------------------------------------------------------------------
    def _build_episode(self, scan, target: ImageRecord, sets, normals_by_set_view,
                       normals_by_cat_view, normal_sets_by_cat, anomalies_by_cat,
                       cat_ids) -> Optional[dict]:
        episode_id = (f"{target.dataset}/{target.category}/{target.sample_id}"
                      f"/v{target.view_id}")
        rng = random.Random(f"{self.seed}:{episode_id}")

        # ---- pairing strategy -------------------------------------------------
        pairing = None
        normal_ref: Optional[ImageRecord] = None
        restored_path: Optional[str] = None

        exact = normals_by_set_view.get((target.set_key, target.view_id))
        if exact is not None:
            pairing, normal_ref = "paired", exact
        elif self.restored_lookup is not None:
            restored_path = self.restored_lookup(target)
            if restored_path:
                pairing = "restored"
        if pairing is None:
            candidates = normals_by_cat_view.get((target.category, target.view_id), [])
            if not candidates:
                self.stats["skipped_no_normal_reference"] += 1
                return None
            pairing, normal_ref = "same_category_same_view_reference", rng.choice(candidates)

        # ---- normal multi-view context ---------------------------------------
        if pairing == "paired":
            ctx_set_key = target.set_key
        elif normal_ref is not None:
            ctx_set_key = normal_ref.set_key
        else:  # restored: pick a normal context set from the same category
            keys = normal_sets_by_cat.get(target.category, [])
            ctx_set_key = rng.choice(keys) if keys else None
        normal_views = []
        if ctx_set_key is not None:
            ctx = [r for r in sets[ctx_set_key] if not r.is_anomaly]
            seen = set()
            for r in sorted(ctx, key=lambda r: r.view_id):
                if r.view_id not in seen:
                    seen.add(r.view_id)
                    normal_views.append({"view_id": r.view_id, "image": r.image_path})

        target_normal_image = restored_path if pairing == "restored" else normal_ref.image_path
        # make sure the target view itself is present in the context list
        if not any(v["view_id"] == target.view_id for v in normal_views):
            normal_views.append({"view_id": target.view_id, "image": target_normal_image})
            normal_views.sort(key=lambda v: v["view_id"])

        # ---- support anomalies ------------------------------------------------
        pool = [r for r in anomalies_by_cat.get(target.category, [])
                if r.sample_id != target.sample_id]
        same_label = [r for r in pool if r.defect_label == target.defect_label]
        chosen: List[ImageRecord] = []
        for src in (same_label, pool):
            for r in rng.sample(src, min(self.max_support, len(src))):
                if len(chosen) >= self.max_support:
                    break
                if all(c.image_path != r.image_path for c in chosen):
                    chosen.append(r)
            if chosen:
                break
        support = [{
            "view_id": r.view_id,
            "image": r.image_path,
            "mask": r.mask_path,
            "defect_label": r.defect_label,
            "defect_family": defect_family_of(r.defect_label),
        } for r in chosen]

        # ---- instruction ------------------------------------------------------
        family = defect_family_of(target.defect_label)
        mask_stats = _load_mask_stats(target.mask_path)
        if not mask_stats.valid:
            self.stats["empty_or_unreadable_masks"] += 1
        instr = generate_instruction(
            episode_key=episode_id,
            defect_label=target.defect_label,
            defect_family=family,
            view_id=target.view_id,
            mask_stats=mask_stats,
            has_support=len(support) > 0,
        )
        if episode_id in self.instruction_overrides:
            instr["pseudo_instruction"] = self.instruction_overrides[episode_id]

        self.stats[f"pairing_{pairing}"] += 1
        return {
            "episode_id": episode_id,
            "dataset": target.dataset,
            "category": target.category,
            "category_id": cat_ids[target.category],
            "product_id": target.category,
            "sample_id": target.sample_id,
            "target_view_id": target.view_id,
            "pairing_strategy": pairing,
            "target_normal_image": target_normal_image,
            "target_anomaly_image": target.image_path,
            "target_mask": target.mask_path,
            "normal_views": normal_views,
            "support_anomalies": support,
            "defect_label": target.defect_label,
            "defect_family": family,
            "mask_stats": {
                "area_ratio": round(mask_stats.area_ratio, 6),
                "cx": round(mask_stats.cx, 4), "cy": round(mask_stats.cy, 4),
                "aspect_ratio": round(mask_stats.aspect_ratio, 3),
                "location_words": mask_stats.location_words,
                "shape_words": mask_stats.shape_words,
            },
            "pseudo_instruction": instr["pseudo_instruction"],
            "expanded_instructions": instr["expanded_instructions"],
            "simplified_instruction": instr["simplified_instruction"],
            "split_hint": target.split_hint,
        }


# ---------------------------------------------------------------------------
def assign_splits(
    episodes: List[dict],
    ratios=(0.8, 0.1, 0.1),
    seed: int = 0,
    holdout_categories: Optional[List[str]] = None,
) -> None:
    """Assign 'train'/'val'/'test' in-place.

    All episodes derived from the same multi-view anomaly set stay in the same
    split. ``holdout_categories`` (for leave-one-product/category experiments)
    are forced entirely into 'test'.
    """
    holdout = {c.lower() for c in (holdout_categories or [])}
    set_keys = sorted({(e["dataset"], e["category"], e["sample_id"]) for e in episodes})
    rng = random.Random(seed)
    rng.shuffle(set_keys)
    n = len(set_keys)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    split_of = {}
    for i, k in enumerate(set_keys):
        if i < n_train:
            split_of[k] = "train"
        elif i < n_train + n_val:
            split_of[k] = "val"
        else:
            split_of[k] = "test"
    for e in episodes:
        if e["category"].lower() in holdout:
            e["split"] = "test"
        else:
            e["split"] = split_of[(e["dataset"], e["category"], e["sample_id"])]
