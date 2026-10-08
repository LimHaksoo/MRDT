"""Qualitative debug visualization: labeled image grids saved as PNG."""

from pathlib import Path
from typing import List, Optional, Sequence

import torch
from PIL import Image, ImageDraw

from rescene.utils.image_io import tensor_to_pil

_CELL_PAD = 4
_LABEL_H = 14


def _norm_for_display(t: torch.Tensor) -> torch.Tensor:
    """Map arbitrary-range tensors (e.g. latent deltas, SDFs) into [0, 1]."""
    t = t.detach().float().cpu()
    lo, hi = t.min().item(), t.max().item()
    if hi - lo < 1e-8:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo)


def to_display(t: torch.Tensor, assume_01: bool = True) -> Image.Image:
    """(C,H,W) tensor -> PIL. Non-image tensors are min-max normalized."""
    t = t.detach().float().cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] > 3:
        t = t[:3]
    if not assume_01 or t.min() < -0.01 or t.max() > 1.01:
        t = _norm_for_display(t)
    return tensor_to_pil(t.clamp(0, 1))


def save_grid(
    rows: List[List[Optional[torch.Tensor]]],
    out_path,
    col_titles: Optional[Sequence[str]] = None,
    row_titles: Optional[Sequence[str]] = None,
    cell_size: int = 192,
) -> None:
    """rows: list of rows; each row a list of (C,H,W) tensors (or None for blank)."""
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    header = _LABEL_H + _CELL_PAD if col_titles else 0
    W = n_cols * (cell_size + _CELL_PAD) + _CELL_PAD
    H = header + n_rows * (cell_size + _CELL_PAD + (_LABEL_H if row_titles else 0)) + _CELL_PAD
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    if col_titles:
        for c, title in enumerate(col_titles):
            x = _CELL_PAD + c * (cell_size + _CELL_PAD)
            draw.text((x, 2), str(title)[:32], fill=(220, 220, 220))

    y = header + _CELL_PAD
    for r, row in enumerate(rows):
        if row_titles:
            draw.text((_CELL_PAD, y), str(row_titles[r])[:96], fill=(180, 220, 180))
            y += _LABEL_H
        for c, cell in enumerate(row):
            if cell is None:
                continue
            img = to_display(cell).resize((cell_size, cell_size), Image.NEAREST)
            x = _CELL_PAD + c * (cell_size + _CELL_PAD)
            canvas.paste(img, (x, y))
        y += cell_size + _CELL_PAD

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def overlay_mask(image: torch.Tensor, mask: torch.Tensor,
                 color=(1.0, 0.15, 0.15), alpha: float = 0.45) -> torch.Tensor:
    """Tint masked pixels. image: (3,H,W) in [0,1]; mask: (1,H,W) in [0,1]."""
    image = image.detach().float().cpu().clamp(0, 1)
    mask = mask.detach().float().cpu()
    if mask.shape[-2:] != image.shape[-2:]:
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=image.shape[-2:], mode="nearest").squeeze(0)
    color_t = torch.tensor(color).view(3, 1, 1)
    return image * (1 - alpha * mask) + color_t * (alpha * mask)
