"""Torch dataset over ReSCENE episode manifests.

Produces per-episode image slots in the fixed layout expected by
``ReSCENERDT``: [target view | normal context views x K | support overlays x S].
Support anomaly examples are rendered as image+mask overlays (red tint) so a
single frozen image encoder can consume them.

Modality dropout is applied inside the model (training only); the dataset just
reports which slots actually exist via ``slot_present``.
"""

from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from rescene.constants import DATASET_IDS, MAX_VIEWS
from rescene.data.manifest import read_jsonl
from rescene.utils.image_io import load_image_tensor, load_mask_tensor
from rescene.utils.viz import overlay_mask


class ReSCENEEpisodeDataset(Dataset):
    def __init__(
        self,
        manifest: Union[str, List[dict]],
        image_size: int = 128,
        max_normal_views: int = 5,
        max_support: int = 2,
        instruction_field: str = "pseudo_instruction",
    ):
        self.episodes = read_jsonl(manifest) if isinstance(manifest, str) else list(manifest)
        if not self.episodes:
            raise ValueError(f"Empty manifest: {manifest}")
        self.image_size = image_size
        self.max_normal_views = max_normal_views
        self.max_support = max_support
        self.num_slots = 1 + max_normal_views + max_support
        self.instruction_field = instruction_field

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict:
        ep = self.episodes[idx]
        size = self.image_size
        target_normal = load_image_tensor(ep["target_normal_image"], size)
        target_anomaly = load_image_tensor(ep["target_anomaly_image"], size)
        target_mask = load_mask_tensor(ep["target_mask"], size)

        S = self.num_slots
        slot_images = torch.zeros(S, 3, size, size)
        slot_present = torch.zeros(S, dtype=torch.bool)
        slot_view_ids = torch.full((S,), -1, dtype=torch.long)

        # slot 0: the target normal view (never dropped)
        slot_images[0] = target_normal
        slot_present[0] = True
        slot_view_ids[0] = int(ep["target_view_id"])

        # normal context views
        views = sorted(ep.get("normal_views", []), key=lambda v: v["view_id"])
        for i, v in enumerate(views[: self.max_normal_views]):
            slot = 1 + i
            slot_images[slot] = load_image_tensor(v["image"], size)
            slot_present[slot] = True
            slot_view_ids[slot] = int(v["view_id"])

        # support anomaly overlays
        supports = ep.get("support_anomalies", [])[: self.max_support]
        for i, s in enumerate(supports):
            slot = 1 + self.max_normal_views + i
            img = load_image_tensor(s["image"], size)
            if s.get("mask"):
                img = overlay_mask(img, load_mask_tensor(s["mask"], size))
            slot_images[slot] = img
            slot_present[slot] = True
            slot_view_ids[slot] = int(s.get("view_id", -1))

        return {
            "target_normal": target_normal,
            "target_anomaly": target_anomaly,
            "target_mask": target_mask,
            "slot_images": slot_images,
            "slot_present": slot_present,
            "slot_view_ids": slot_view_ids,
            "dataset_id": torch.tensor(DATASET_IDS.get(ep["dataset"], len(DATASET_IDS)),
                                       dtype=torch.long),
            "target_view_id": torch.tensor(min(int(ep["target_view_id"]), MAX_VIEWS),
                                           dtype=torch.long),
            "instruction": ep.get(self.instruction_field) or ep["pseudo_instruction"],
            "episode": ep,
        }


def collate_episodes(items: List[Dict]) -> Dict:
    out = {}
    for key in ("target_normal", "target_anomaly", "target_mask", "slot_images",
                "slot_present", "slot_view_ids", "dataset_id", "target_view_id"):
        out[key] = torch.stack([it[key] for it in items])
    out["instruction"] = [it["instruction"] for it in items]
    out["episode"] = [it["episode"] for it in items]
    return out
