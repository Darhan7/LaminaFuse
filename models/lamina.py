"""Differentiable local log-contrast decomposition and fixed reconstruction.

The public analytic interface uses softmax(2 * log(E + 1e-12)), matching
the paper checkpoint. The legacy energy_weight helper below is not used
by that interface. Gaussian truncation follows the original PyTorch model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, reflect padding, radius 3 sigma (matches cv2.GaussianBlur up to truncation)."""
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


class Lamina(nn.Module):
    def __init__(self, eps: float = 0.03, sigma_s: float = 4.0, sigma_e: float = 6.0, polarity: bool = False):
        super().__init__()
        self.eps, self.sigma_s, self.sigma_e, self.polarity = float(eps), float(sigma_s), float(sigma_e), bool(polarity)

    def forward(self, v01: torch.Tensor, nir01: torch.Tensor) -> dict:
        Lv, Ln = torch.log(v01 + self.eps), torch.log(nir01 + self.eps)
        bv, bn = gaussian_blur(Lv, self.sigma_s), gaussian_blur(Ln, self.sigma_s)
        cv, cn = Lv - bv, Ln - bn
        if self.polarity:
            s = torch.where(gaussian_blur(cv * cn, 3 * self.sigma_e) < 0, -1.0, 1.0).detach()
            cn = cn * s
        ev, en = gaussian_blur(cv * cv, self.sigma_e), gaussian_blur(cn * cn, self.sigma_e)
        return {"c_v": cv, "c_n": cn, "e_v": ev, "e_n": en, "base": 0.5 * (bv + bn)}

    def reconstruct(self, base: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (torch.exp(base + c) - self.eps).clamp(0.0, 1.0)

    @staticmethod
    def energy_weight(ev: torch.Tensor, en: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        """Training-free prior: w_v = E_v^p / (E_v^p + E_n^p)."""
        a, b = ev.clamp_min(1e-12) ** p, en.clamp_min(1e-12) ** p
        return a / (a + b)


class FullResWeightRefine(nn.Module):
    """Full-resolution residual on the (upsampled, coarse) weight logits from local lamina features."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + 5, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # starts as identity

    def forward(self, logits: torch.Tensor, lam: dict) -> torch.Tensor:
        le = lambda e: torch.log(e + 1e-6) / 10.0
        x = torch.cat([logits, lam["c_v"], lam["c_n"], le(lam["e_v"]), le(lam["e_n"]), lam["base"] / 4.0], 1)
        return logits + self.net(x)
