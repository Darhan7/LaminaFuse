"""Route-A LIFR prototype; independent of the thesis dark-repair equations.

This estimates plausible LOW-FREQUENCY fused illumination, not unclipped NIR
radiance or missing texture. All constants below are experimental assumptions.
There are no learned parameters and no inference-time geometry dependency.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _ramp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """C1 smoothstep with compact support (including exact saturation exclusion)."""
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t.square() * (3 - 2 * t)


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), k, stride=1)


def _neighbors(x: torch.Tensor) -> torch.Tensor:
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return (p[..., 1:-1, :-2] + p[..., 1:-1, 2:] + p[..., :-2, 1:-1] + p[..., 2:, 1:-1]) / 4


def _context_moments(x: torch.Tensor, y: torch.Tensor, q: torch.Tensor):
    """17-cell weighted moments, with explicitly weak whole-frame fallback.

    Adding 0.1 times global moments gives even deep hole interiors an initial
    estimate. If no trusted context exists, confidence is zero, not fabricated.
    """
    values = torch.cat((q, q*x, q*y, q*x*x, q*x*y), dim=1)
    moments = _box(values, 17) + 0.1 * values.mean(dim=(-2, -1), keepdim=True)
    mass, sx, sy, sxx, sxy = moments.split(1, dim=1)
    den = mass.clamp_min(1e-6)
    mx, my = sx / den, sy / den
    variance = (sxx / den - mx.square()).clamp_min(0)
    covariance = sxy / den - mx * my
    # Unit-slope ridge preserves shading in textureless context; bounded slope encodes
    # monotone local response, not a calibrated cross-spectral reflectance law.
    a = ((covariance + 1e-3) / (variance + 1e-3)).clamp(0, 3)
    b = my - a * mx
    confidence = mass / (mass + 0.05)
    return a, b, confidence


def latent_illumination_reconstruction(v: torch.Tensor, n: torch.Tensor, thesis: torch.Tensor):
    """Return experimental output and diagnostics; preserve every thesis tensor.

    Solve on an ~8-pixel grid, 64 Jacobi iterations per field. Supports extend
    across entire holes; the target is not restricted to a boundary band.
    Float32 arithmetic avoids half-precision ill-conditioning of local moments.
    """
    with torch.autocast(device_type=thesis.device.type, enabled=False):
        v, n, thesis = v.float(), n.float(), thesis.float()
        size = thesis.shape[-2:]
        coarse_size = tuple(max(2, math.ceil(s / 8)) for s in size)
        down = lambda x: F.adaptive_avg_pool2d(x, coarse_size)
        up = lambda x: F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        sat = _ramp(n, 0.85, 0.95)
        conflict = sat * (1 - _ramp(v, 0.05, 0.25)) * (1 - _ramp(thesis, 0.15, 0.45))
        known = 1 - sat
        u = down(known)
        # Saturated values have EXACTLY zero observation weight; never box-filter
        # raw NIR into the unknown region before excluding them.
        un = down(known * n)
        global_mass = u.mean(dim=(-2, -1), keepdim=True)
        n_global = un.mean(dim=(-2, -1), keepdim=True) / global_mass.clamp_min(1e-6)
        n_hat = (_box(un, 17) + 0.1 * global_mass * n_global) / (
            _box(u, 17) + 0.1 * global_mass).clamp_min(1e-6)
        for _ in range(64):
            n_hat = (4 * un + _neighbors(n_hat)) / (4 * u + 1)
        n_conf = (_box(u, 17) + 0.1 * global_mass)
        n_conf = n_conf / (n_conf + 0.1)

        lf = _box(down(thesis), 3)
        vl = _box(down(v), 3)
        m = _box(down(conflict), 3)
        # Trusted context excludes all saturated donors, conflict, very dark
        # fused observations, and nearly clipped fused observations.
        q_full = known * (1-conflict) * _ramp(thesis, 0.08, 0.2) * (1-_ramp(thesis, 0.85, 0.98))
        q = down(q_full)
        # Normalize BEFORE fitting: unmasked lowpass would leak dark unknown
        # pixels into trusted boundary targets and bias the fitted brightness.
        q_den = q.clamp_min(1e-6)
        context_f = down(q_full * thesis) / q_den
        context_v = down(q_full * v) / q_den
        context_n = down(q_full * n) / q_den
        av, bv, context_conf = _context_moments(context_v, context_f, q)
        an, bn, _ = _context_moments(context_n, context_f, q)
        t_rgb = (av * vl + bv).clamp(0, 1)
        t_nir = (an * n_hat + bn).clamp(0, 1)
        dx = F.pad((vl[..., 1:] - vl[..., :-1]).abs(), (0,1,0,0))
        dy = F.pad((vl[..., 1:, :] - vl[..., :-1, :]).abs(), (0,0,0,1))
        structure = _box(dx + dy, 3)
        c_rgb = context_conf * structure / (structure + 0.01)
        c_nir = context_conf * n_conf
        den = (c_rgb + c_nir).clamp_min(1e-6)
        target = (c_rgb * t_rgb + c_nir * t_nir) / den
        # Relative gradients only: offsets calibrate T, not the guidance field.
        guidance = (c_rgb * av * vl + c_nir * an * n_hat) / den
        anchor = 8 * (1-m)
        target_weight = 2 * m * context_conf
        z = (1-m) * lf + m * target
        # Energy: anchor*(Z-LF)^2 + target_weight*(Z-T)^2
        #       + 0.5 * sum_edges ((Zi-Zj)-(Gi-Gj))^2.
        for _ in range(64):
            z = (anchor*lf + target_weight*target + 2*(_neighbors(z-guidance)+guidance)) / (anchor+target_weight+2)
        low = up(lf)
        # Soft coarse support keeps the correction low-frequency. Outside support
        # it is exactly zero; no context -> exactly zero, even if all NIR clips.
        # Do not turn spatial confidence falloff into an artificial bright ring
        # and dark center. Local confidence already weights the interior target;
        # use only a framewise evidence factor for final conservative blending.
        context_mass = q.mean(dim=(-2, -1), keepdim=True)
        evidence = context_mass / (context_mass + 0.01)
        correction = up(m * evidence * (z-lf))
        l_star = low + correction
        high = thesis - low
        # Algebraically H_F + L_star; this form is exactly identity at zero delta.
        final = (thesis + correction).clamp(0, 1)
        return {
            "f_thesis": thesis, "f_lifr": final, "f_lifr255": final * 255,
            "lifr_conflict_mask": conflict, "lifr_saturation_mask": sat,
            "lifr_effective_mask": up(m), "lifr_n_hat": up(n_hat),
            "lifr_t_rgb": up(t_rgb), "lifr_t_nir": up(t_nir), "lifr_target": up(target),
            "lifr_target_coarse": target,
            "lifr_l_f": low, "lifr_l_star": l_star, "lifr_correction": correction,
            "lifr_high_frequency": high, "lifr_rgb_confidence": up(c_rgb),
            "lifr_nir_confidence": up(c_nir), "lifr_context_confidence": up(context_conf),
        }
