# -*- coding: utf-8 -*-
"""Checkpoint-compatible fusion network with learned weight corrections.

The paper configuration uses local log-contrast reconstruction. Historical
intensity and disabled auxiliary options are retained for source compatibility;
the release loader validates the paper configuration before inference.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_illumination import latent_illumination_reconstruction
from .latent_illumination_v2 import latent_illumination_reconstruction_v2
from .fusion_attention_blocks import BasicEncoder, ResidualBlock, BAttentionFeatureFusion
from .lamina import Lamina, FullResWeightRefine


def _rgb_to_value(rgb01: torch.Tensor) -> torch.Tensor:
    return torch.amax(rgb01, dim=1, keepdim=True)


def _local_mean_1ch(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Reconstruction choice: a border-preserving box mean for thesis blur_15."""
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)


def _local_structure_gate(x: torch.Tensor, low: torch.Tensor, kernel: int) -> torch.Tensor:
    """Reconstruction choice: soft support from gradients and local high frequency."""
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0)).abs()
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1)).abs()
    structure = _local_mean_1ch(dx + dy + (x - low).abs(), kernel)
    return structure / (structure + 0.02)


def _cross_modal_hint_gate(
    v01: torch.Tensor, v_low: torch.Tensor, nir01: torch.Tensor, nir_low: torch.Tensor, kernel: int
) -> torch.Tensor:
    """Reconstruction choice: map local high-frequency correlation into [0, 1]."""
    v_high = v01 - v_low
    nir_high = nir01 - nir_low
    covariance = _local_mean_1ch(v_high * nir_high, kernel)
    v_energy = _local_mean_1ch(v_high.square(), kernel)
    nir_energy = _local_mean_1ch(nir_high.square(), kernel)
    correlation = covariance / torch.sqrt(v_energy * nir_energy + 1e-6)
    return (0.5 + 0.5 * correlation.clamp(-1.0, 1.0)).clamp(0.0, 1.0)


def constrained_dark_repair(
    f_raw: torch.Tensor, v01: torch.Tensor, nir01: torch.Tensor, *, scale: float = 1.0, blur_kernel: int = 15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return repaired image, support mask, actual residual, and reliability gate.

    Thesis/history specify the dark and NIR-advantage sigmoids, positive lift gap,
    and multiplicative repair. The structure, agreement, saturation gates and
    neutral repair_gain=1 below reconstruct unspecified lost-code details.
    """
    fused_low = _local_mean_1ch(f_raw, blur_kernel)
    nir_low = _local_mean_1ch(nir01, blur_kernel)
    v_low = _local_mean_1ch(v01, blur_kernel)

    dark_mask = torch.sigmoid(8.0 * (0.44 - fused_low))
    nir_bright_adv = torch.sigmoid(7.0 * (nir_low - fused_low - 0.005))

    rgb_support = _local_structure_gate(v01, v_low, blur_kernel)
    nir_structure = _local_structure_gate(nir01, nir_low, blur_kernel)
    agreement = _cross_modal_hint_gate(v01, v_low, nir01, nir_low, blur_kernel)
    # Reconstruction choice: softly reject locally saturated RGB or NIR responses.
    rgb_risk = _local_mean_1ch(torch.sigmoid(30.0 * (v01 - 0.95)), blur_kernel)
    nir_risk = _local_mean_1ch(torch.sigmoid(30.0 * (nir01 - 0.95)), blur_kernel)
    nir_usefulness = nir_structure * (1.0 - rgb_risk) * (1.0 - nir_risk)
    reliability = (rgb_support * nir_usefulness * agreement).clamp(0.0, 1.0)
    repair_mask = (dark_mask * nir_bright_adv * reliability).clamp(0.0, 1.0)

    lift_gap = (nir_low - fused_low).clamp_min(0.0)
    # G_rep is not specified in the surviving text, so use neutral gain 1.
    proposed_delta = scale * repair_mask * lift_gap
    repaired = (f_raw + proposed_delta).clamp(0.0, 1.0)
    actual_delta = (repaired - f_raw).clamp_min(0.0)
    return repaired, repair_mask, actual_delta, reliability


def _guided_filter_1ch(guide: torch.Tensor, src: torch.Tensor, radius: int = 5, eps: float = 1e-5) -> torch.Tensor:
    if radius <= 0:
        return src
    k = 2 * radius + 1
    n = float(k * k)

    def box(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius) * n

    mean_i = box(guide) / n
    mean_p = box(src) / n
    corr_i = box(guide * guide) / n
    corr_ip = box(guide * src) / n
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = box(a) / n
    mean_b = box(b) / n
    return (mean_a * guide + mean_b).clamp(0.0, 1.0)


def _replace_bn_with_gn(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            ch = int(child.num_features)
            groups = min(32, ch)
            while groups > 1 and (ch % groups != 0):
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, ch, affine=True))
        else:
            _replace_bn_with_gn(child)


class AuxDispHead1Ch(nn.Module):
    def __init__(self, dmin_px: float = 0.05, dmax_px: float = 256.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Softplus(beta=1.0),
        )
        self.register_buffer("dmin_px", torch.tensor(float(dmin_px)), persistent=False)
        self.register_buffer("dmax_px", torch.tensor(float(dmax_px)), persistent=False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        disp = self.net(x01)
        disp = torch.nan_to_num(disp, nan=0.0, posinf=self.dmax_px, neginf=0.0)
        return disp.clamp(self.dmin_px, self.dmax_px)


class WeightFusionNet(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        reduction: int = 4,
        use_guided_filter: bool = False,
        guided_radius: int = 5,
        dark_repair_enable: bool = True,
        dark_repair_scale: float = 1.0,
        dark_repair_blur_kernel: int = 15,
        lifr_enable: bool = False,
        lifr2_enable: bool = False,
        fuse_domain: str = "intensity",
        lamina_prior: bool = True,
        lamina_prior_p: float = 2.0,
        lamina_polarity: bool = False,
        fullres_refine: bool = False,
    ):
        super().__init__()
        self.encoder = BasicEncoder(downsample=2, output_dim=int(feat_dim))
        self.fusion = BAttentionFeatureFusion(in_channels=int(feat_dim), reduction=int(reduction))
        # Attention fusion blocks use BatchNorm(track_running_stats=False), which is unstable for bs=1 and 1x1 maps.
        _replace_bn_with_gn(self.fusion)
        mid = max(64, int(feat_dim // 2))
        low = max(32, int(mid // 2))
        self.weight_head = nn.Sequential(
            ResidualBlock(int(feat_dim), mid, norm_fn="batch"),
            ResidualBlock(mid, low, norm_fn="batch"),
            nn.Conv2d(low, 2, kernel_size=3, padding=1),
        )
        self.use_guided_filter = bool(use_guided_filter)
        self.guided_radius = int(guided_radius)
        self.lifr_enable = bool(lifr_enable)
        self.lifr2_enable = bool(lifr2_enable)
        self.dark_repair_enable = bool(dark_repair_enable)
        self.dark_repair_scale = float(dark_repair_scale)
        self.dark_repair_blur_kernel = int(dark_repair_blur_kernel)
        # v2: lamina-stage (contrast-domain) fusion; with lamina_prior the logits start at the training-free
        # energy-ratio rule (zero network output == screen_lamina.py 'lam'), the network learns corrections.
        assert fuse_domain in ("intensity", "lamina"), fuse_domain
        self.fuse_domain = fuse_domain
        self.lamina = Lamina(polarity=lamina_polarity) if fuse_domain == "lamina" else None
        self.lamina_prior = bool(lamina_prior) and self.lamina is not None
        self.lamina_prior_p = float(lamina_prior_p)
        self.refine = FullResWeightRefine() if fullres_refine else None
        if self.lamina_prior:   # start exactly at the training-free rule
            nn.init.zeros_(self.weight_head[-1].weight); nn.init.zeros_(self.weight_head[-1].bias)

    def forward(self, rgb01: torch.Tensor, nir01: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb01 = rgb01.clamp(0.0, 1.0)
        nir01 = nir01.clamp(0.0, 1.0)
        v01 = _rgb_to_value(rgb01)

        rgb_in = rgb01 * 2.0 - 1.0
        nir_in = nir01.repeat(1, 3, 1, 1) * 2.0 - 1.0
        rgb_feat = self.encoder(rgb_in)
        nir_feat = self.encoder(nir_in)
        fused_feat = self.fusion(rgb_feat, nir_feat)

        logits = self.weight_head(fused_feat)
        logits = F.interpolate(logits, size=rgb01.shape[-2:], mode="bilinear", align_corners=False)
        lam = None
        # Lamina maths in fp32: under AMP the Gaussian convs would run in fp16 and contrast energies (~1e-4)
        # underflow -> log(0) = -inf -> NaN weights.
        with torch.autocast("cuda", enabled=False):
            logits = logits.float()
            prior = None
            if self.lamina is not None:
                lam = self.lamina(v01.float(), nir01.float())
                if self.lamina_prior:
                    prior = self.lamina_prior_p * torch.cat(
                        [torch.log(lam["e_v"] + 1e-12), torch.log(lam["e_n"] + 1e-12)], 1)
                    logits = logits + prior
            if self.refine is not None:
                if lam is None:
                    lam = Lamina()(v01.float(), nir01.float())
                logits = self.refine(logits, lam)
            weights = torch.softmax(logits, dim=1)
            # correction to the training-free rule (softmax only sees the channel difference)
            correction = None if prior is None else (logits - prior)[:, 0:1] - (logits - prior)[:, 1:2]
            w_v = weights[:, 0:1]
            w_n = weights[:, 1:2]
            if self.fuse_domain == "lamina":
                f_raw = self.lamina.reconstruct(lam["base"], w_v * lam["c_v"] + w_n * lam["c_n"])
        if self.fuse_domain != "lamina":
            f_raw = (w_v * v01 + w_n * nir01).clamp(0.0, 1.0)
        f01 = f_raw
        if self.use_guided_filter:
            f01 = _guided_filter_1ch(nir01, f01, radius=self.guided_radius)
        if self.dark_repair_enable:
            f01, repair_mask, repair_delta, reliability = constrained_dark_repair(
                f01, v01, nir01, scale=self.dark_repair_scale, blur_kernel=self.dark_repair_blur_kernel
            )
        else:
            repair_mask = torch.zeros_like(f01)
            repair_delta = torch.zeros_like(f01)
            reliability = torch.zeros_like(f01)
        f255 = (f01 * 255.0).clamp(0.0, 255.0).to(torch.float32)
        outputs = {
            "f255": f255,
            "f01": (f255 / 255.0),
            "f_raw": f_raw,
            "weights": weights,
            "w_v": w_v,
            "w_n": w_n,
            "v01": v01,
            "fused_feat": fused_feat,
            "nir_dark_reveal_mask": repair_mask,
            "nir_dark_reveal_delta": repair_delta,
            "dark_hint_reliability": reliability,
            "prior_correction": correction,
        }
        if self.lifr_enable or self.lifr2_enable:
            outputs.update(latent_illumination_reconstruction(v01, nir01, outputs["f01"]))
            if self.lifr2_enable:
                outputs.update(latent_illumination_reconstruction_v2(v01, nir01, outputs["f01"], outputs))
        return outputs
