"""Bin15 camera-conditioned RDT wrapper for MVImgNet2.0 Stage-1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_MAIN_REPO = Path("/home/haksoo/ReSCENE-RDT")
if str(_MAIN_REPO) not in sys.path:
    sys.path.insert(0, str(_MAIN_REPO))

from models.rdt_runner import RDTRunner
from rescene.codec.base import EditActionSpec
from rescene.pipeline import configure_action_positional_embedding, load_rdt_pretrained_partial


def latent_to_patch_tokens(z: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    if z.ndim != 4 or tuple(z.shape[1:]) != (4, 32, 32):
        raise ValueError(f"expected latent (B,4,32,32), got {tuple(z.shape)}")
    b, c, h, w = z.shape
    p = int(patch_size)
    patches = z.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(b, (h // p) * (w // p), c * p * p)


def patch_tokens_to_latent(tokens: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (64, 64):
        raise ValueError(f"expected tokens (B,64,64), got {tuple(tokens.shape)}")
    b = tokens.shape[0]
    p = int(patch_size)
    grid = tokens.view(b, 8, 8, 4, p, p)
    grid = grid.permute(0, 3, 1, 4, 2, 5).contiguous()
    return grid.view(b, 4, 32, 32)


class Bin15CameraRDT(nn.Module):
    """Use binned camera features without ReSCENE discrete view embeddings."""

    def __init__(
        self,
        *,
        rdt_config: dict,
        img_token_dim: int,
        tokens_per_image: int = 64,
        k_max: int = 5,
        action_dim: int = 64,
        pred_horizon: int = 64,
        lang_token_dim: int = 4096,
        camera_feature_dim: int = 10,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.k_max = int(k_max)
        self.tokens_per_image = int(tokens_per_image)
        self.action_dim = int(action_dim)
        self.pred_horizon = int(pred_horizon)
        self.camera_feature_dim = int(camera_feature_dim)
        self.runner = RDTRunner(
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            config=rdt_config,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=self.action_dim,
            max_lang_cond_len=1,
            img_cond_len=self.k_max * self.tokens_per_image,
            img_pos_embed_config=[("image", (self.k_max, self.tokens_per_image))],
            lang_pos_embed_config=[("null_text", 1)],
            dtype=dtype,
        )
        hidden_img = max(128, img_token_dim // 4)
        self.context_camera_mlp = nn.Sequential(
            nn.Linear(camera_feature_dim, hidden_img),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_img, img_token_dim),
        )
        self.target_camera_mlp = nn.Sequential(
            nn.Linear(camera_feature_dim, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, self.action_dim),
        )
        self.null_lang_token = nn.Parameter(torch.zeros(1, 1, lang_token_dim))
        self.missing_image_token = nn.Parameter(torch.zeros(1, 1, 1, img_token_dim))
        nn.init.normal_(self.null_lang_token, std=0.02)
        nn.init.normal_(self.missing_image_token, std=0.02)

    @property
    def prediction_type(self) -> str:
        return self.runner.prediction_type

    def install_pretrained(self, ckpt_path: str) -> dict:
        report = load_rdt_pretrained_partial(self.runner, ckpt_path)
        spec = EditActionSpec(grid_channels=4, grid_h=32, grid_w=32, mask_channels=0, tokenization="patch2d", patch_size=4)
        pos = configure_action_positional_embedding(self.runner, spec, mode="2d_sincos")
        report["action_pos_embed"] = pos
        return report

    def _conditions(
        self,
        *,
        slot_feats: torch.Tensor,
        context_camera_features: torch.Tensor,
        target_camera_features: torch.Tensor,
        context_present: torch.Tensor,
    ):
        b, k, p, d = slot_feats.shape
        if k != self.k_max or p != self.tokens_per_image:
            raise ValueError(f"slot_feats expected (B,{self.k_max},{self.tokens_per_image},D), got {tuple(slot_feats.shape)}")
        present = context_present.bool()
        cam = self.context_camera_mlp(context_camera_features.to(slot_feats.device, slot_feats.dtype))
        feats = slot_feats + cam[:, :, None, :]
        feats = torch.where(present[:, :, None, None], feats, self.missing_image_token.to(feats.dtype))
        img_tokens = feats.reshape(b, k * p, d)
        img_mask = present[:, :, None].expand(b, k, p).reshape(b, k * p)
        state_tokens = self.target_camera_mlp(target_camera_features.to(slot_feats.device, slot_feats.dtype)).unsqueeze(1)
        lang = self.null_lang_token.expand(b, 1, -1).to(slot_feats.device, slot_feats.dtype)
        lang_mask = torch.ones((b, 1), dtype=torch.bool, device=slot_feats.device)
        return lang, lang_mask, img_tokens, img_mask, state_tokens

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_gt = batch["action_gt"]
        b = action_gt.shape[0]
        lang, lang_mask, img, img_mask, state = self._conditions(
            slot_feats=batch["slot_feats"],
            context_camera_features=batch["context_camera_features"],
            target_camera_features=batch["target_camera_features"],
            context_present=batch["context_present"],
        )
        device = action_gt.device
        noise = torch.randn_like(action_gt)
        timesteps = torch.randint(0, self.runner.num_train_timesteps, (b,), device=device).long()
        noisy = self.runner.noise_scheduler.add_noise(action_gt, noise, timesteps)
        action_mask = torch.ones(b, 1, self.action_dim, device=device, dtype=action_gt.dtype)
        raw = torch.cat([state, noisy], dim=1)
        raw = torch.cat([raw, action_mask.expand(-1, raw.shape[1], -1)], dim=2)
        lang_cond, img_cond, state_action = self.runner.adapt_conditions(lang, img, raw)
        pred = self.runner.model(
            state_action,
            torch.ones(b, device=device, dtype=action_gt.dtype),
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_mask,
            img_mask=img_mask,
        )
        if self.runner.prediction_type == "sample":
            target = action_gt
        elif self.runner.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unsupported prediction_type={self.runner.prediction_type}")
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor], *, num_steps: int = 20, seed: Optional[int] = None) -> torch.Tensor:
        lang, lang_mask, img, img_mask, state = self._conditions(
            slot_feats=batch["slot_feats"],
            context_camera_features=batch["context_camera_features"],
            target_camera_features=batch["target_camera_features"],
            context_present=batch["context_present"],
        )
        b = state.shape[0]
        device = state.device
        dtype = state.dtype
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(int(seed))
            noisy = torch.randn((b, self.pred_horizon, self.action_dim), device=device, dtype=dtype, generator=g)
        else:
            noisy = torch.randn((b, self.pred_horizon, self.action_dim), device=device, dtype=dtype)
        action_mask = torch.ones(b, 1, self.action_dim, device=device, dtype=dtype)
        lang_cond, img_cond, state_hidden = self.runner.adapt_conditions(
            lang, img, torch.cat([state, action_mask], dim=2)
        )
        self.runner.noise_scheduler_sample.set_timesteps(int(num_steps))
        for t in self.runner.noise_scheduler_sample.timesteps:
            action_raw = torch.cat([noisy, action_mask.expand(-1, self.pred_horizon, -1)], dim=2)
            action_hidden = self.runner.state_adaptor(action_raw)
            traj = torch.cat([state_hidden, action_hidden], dim=1)
            model_output = self.runner.model(
                traj,
                torch.ones(b, device=device, dtype=dtype),
                t.unsqueeze(-1).to(device),
                lang_cond,
                img_cond,
                lang_mask=lang_mask,
                img_mask=img_mask,
            )
            noisy = self.runner.noise_scheduler_sample.step(model_output, t, noisy).prev_sample.to(dtype)
        return noisy
