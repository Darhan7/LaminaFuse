"""Optional LIFR-2 illumination replacement using the original coarse target."""
import torch
import torch.nn.functional as F

from .latent_illumination import _box, _ramp


def _shift(x, dy, dx):
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return p[..., 1 + dy:1 + dy + x.shape[-2], 1 + dx:1 + dx + x.shape[-1]]


def _guided(guide, source, radius=8, eps=0.0025):
    """Single-channel guided filter of an already coarse photometric field."""
    k = 2 * radius + 1
    mean_g, mean_s = _box(guide, k), _box(source, k)
    var_g = (_box(guide * guide, k) - mean_g.square()).clamp_min(0)
    cov = _box(guide * source, k) - mean_g * mean_s
    a = cov / (var_g + eps)
    b = mean_s - a * mean_g
    return (_box(a, k) * guide + _box(b, k)).clamp(0, 1)


def latent_illumination_reconstruction_v2(v, n, thesis, old):
    """Propagate a strict seed through compatible coarse cells, then replace LF.

    The original target estimator and original LIFR output are supplied in old.
    No clipped NIR intensity is used as a donor or as high-resolution guidance.
    """
    with torch.autocast(device_type=thesis.device.type, enabled=False):
        v, n, thesis = v.float(), n.float(), thesis.float()
        size = thesis.shape[-2:]
        coarse = old['lifr_target_coarse']
        cs = coarse.shape[-2:]
        down = lambda x: F.adaptive_avg_pool2d(x, cs)
        up = lambda x: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        # Strict detector; used only to start spatial propagation.
        seed = (old['lifr_conflict_mask'] > 0.1).float()
        seeds = F.adaptive_max_pool2d(seed, cs)
        sat = old['lifr_saturation_mask']
        valid = 1 - sat
        rgb = _box(v, 17)
        fused = _box(thesis, 17)
        nir = _box(valid * n, 17) / _box(valid, 17).clamp_min(1e-4)
        nir_valid = _box(valid, 17)
        features = (down(rgb), down(fused), down(nir), down(nir_valid))
        # Edge conductance uses low-frequency RGB and thesis appearance. NIR
        # contributes only if both neighboring cells have unsaturated evidence.
        affinity = []
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            rv, rf, rn, ru = (_shift(x, dy, dx) for x in features)
            dv = (features[0] - rv).abs()
            df = (features[1] - rf).abs()
            dn = (features[2] - rn).abs() * torch.minimum(features[3], ru)
            affinity.append(torch.exp(-dv / 0.12 - df / 0.16 - dn / 0.18))
        reach = seeds
        for _ in range(14):
            candidates = [0.96 * a * _shift(reach, dy, dx) for a, (dy, dx) in
                          zip(affinity, ((0, -1), (0, 1), (-1, 0), (1, 0)))]
            reach = torch.maximum(seeds, torch.stack([reach] + candidates).amax(0))
        support_coarse = (reach > 0.18).float()
        # Soft boundary, but strictly zero beyond the propagated support.
        confidence_coarse = 0.95 * _ramp(reach, 0.18, 0.55) * support_coarse

        # The guide is spatially precise but blurred enough to avoid transferring
        # fine texture into illumination. NIR enters only where unsaturated.
        guide = (0.55 * rgb + 0.45 * fused)
        guide = (guide + 0.15 * nir_valid * (nir - guide)).clamp(0, 1)
        t_hr = _guided(guide, up(coarse))
        confidence = up(confidence_coarse)
        # A seed-connected support is retained where interpolation would dilute
        # narrow saturated marking cells.
        confidence = torch.maximum(confidence, 0.95 * seed)
        # No credible radiometric context means no reconstruction.
        evidence = old['lifr_context_confidence'].mean(dim=(-2, -1), keepdim=True)
        confidence = confidence * _ramp(evidence, 0.02, 0.15)
        support = confidence > 0
        low = old['lifr_l_f']
        l_new = low + confidence * (t_hr - low)
        correction = l_new - low
        result = (thesis + correction).clamp(0, 1)
        return {
            'f_lifr2': result, 'f_lifr2_255': result * 255,
            'lifr2_seed': seed, 'lifr2_support': confidence > 0,
            'lifr2_confidence': confidence, 'lifr2_target_hr': t_hr,
            'lifr2_l_new': l_new, 'lifr2_correction': correction,
            'lifr2_high_frequency': thesis - low,
        }
