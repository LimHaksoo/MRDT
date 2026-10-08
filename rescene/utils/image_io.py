"""Image loading helpers. Convention: RGB images as float tensors in [0, 1],
masks as float tensors in {0, 1}, shape (C, H, W)."""

import torch
from PIL import Image


def load_image_tensor(path, size: int) -> torch.Tensor:
    """RGB image -> (3, size, size) float tensor in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return arr.reshape(size, size, 3).permute(2, 0, 1).float() / 255.0


def load_mask_tensor(path, size: int) -> torch.Tensor:
    """Binary mask -> (1, size, size) float tensor in {0, 1}."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.NEAREST)
        arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        return (arr.reshape(1, size, size).float() / 255.0 > 0.5).float()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) or (1, H, W) float tensor in [0, 1] -> PIL image."""
    t = t.detach().float().cpu().clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    arr = (t * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)
