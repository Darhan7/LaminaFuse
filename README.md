# LaminaFuse

**Overexposure-Robust RGB–NIR Fusion via Local Contrast**

LaminaFuse combines registered RGB and near-infrared (NIR) images using local log contrast. A contrast-energy prior and learned weight corrections combine structure from the two bands; a fixed reconstruction and visible-color transfer produce a color image.

This repository provides the current inference implementation and selected numerical results. The associated manuscript is in preparation and has not been posted to arXiv. Maintainer: [Darhan7](https://github.com/Darhan7).

The main observed benefit is retaining visible structure where recorded NIR values are near their maximum. The method does not recover information missing from both inputs, and dark RGB regions can remain too dark. The current MS2 regional evidence comes from one night sequence.

## Quick start

Install a PyTorch build for your CPU or CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_backbone.py
python scripts/make_demo_inputs.py
python inference.py --rgb outputs/demo/rgb.png --nir outputs/demo/nir.png \
  --mode learned --checkpoint checkpoints/laminafuse_epoch08.pt \
  --output outputs/demo/laminafuse.png --device cpu
```

The included example is synthetic and checks installation; it is not experimental evidence. The 7.3 MiB checkpoint contains the epoch-8 model parameters and configuration, without optimizer state. Its provenance and SHA-256 are recorded in [docs/CHECKPOINT.json](docs/CHECKPOINT.json).

The setup script obtains only two required modules from the official [PixelNIR repository](https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo), pinned to commit `8215f4d31d588c3c06e75ec6959b75474002bb7e`. It verifies their SHA-256 hashes before installation. Downloaded source lives in ignored `external/PixelNIR/` and is not distributed in this repository. The default location is found automatically, with no environment variable required.

## Fuse your images

```bash
python inference.py --rgb your_rgb.png --nir your_nir.png \
  --mode learned --checkpoint checkpoints/laminafuse_epoch08.pt \
  --output outputs/fused.png --raw-output outputs/fused_float32.npz --device cuda
```

Inputs must already be geometrically registered: an 8-bit RGB image and an 8-bit grayscale NIR image, on the same grid and at least 32 pixels per dimension. Normalize higher-bit-depth sensor images explicitly. No registration, automatic contrast enhancement or exposure adjustment is applied.

For invalid registration pixels, add `--valid-mask valid.png`. Zero mask values restore the original visible image there. The evaluation protocol separately excludes a one-pixel border. The output PNG is for viewing; the optional NPZ preserves float32 values for numerical evaluation. CPU and GPU results can differ slightly.

## Training-free controls

```bash
python inference.py --rgb outputs/demo/rgb.png --nir outputs/demo/nir.png \
  --mode analytic-contrast --output outputs/demo/contrast.png
python inference.py --rgb outputs/demo/rgb.png --nir outputs/demo/nir.png \
  --mode analytic-intensity --output outputs/demo/intensity.png
```

These modes require neither PixelNIR code nor a checkpoint. They share the exact prior `softmax(2 * log(E + 1e-12))`; only reconstruction changes. The default mode is `analytic-contrast`, not the learned LaminaFuse model. The legacy `Lamina.energy_weight` helper is not used by this interface.

## Restore or verify the external dependency

```bash
# Verify the installed source without network access:
python scripts/prepare_backbone.py --check
# Prepare offline from an official checkout already on your computer:
python scripts/prepare_backbone.py --source-dir /path/to/PixelNIR
```

Repeated setup verifies existing files and never overwrites modified source. For a custom location, set `PIXELNIR_CODE` to its absolute path; it overrides automatic discovery. Our GroupNorm conversion, learned weight corrections, contrast reconstruction and color interface are included here, so no patch containing upstream code is required. See [docs/upstream.json](docs/upstream.json) for the fixed revision and hashes.

## Recompute the matched-control table

```bash
python scripts/reproduce_control_table.py
```

This reconstructs Table V's eight numeric columns from recorded per-frame measurements, without a model or GPU. It does not rerun image scoring or feature extraction. The [reference table](results/control/paper_table_V.tex) and CSV records are included.

The high-NIR region is `N >= 0.98` and `0.10 < max(RGB) < 0.95`, within valid registration and an eroded border. Q uses equal-frame averaging. SIFT/ORB counts average over all 96 stereo pairs; precision and yield pool the underlying counts. Correct matches use a 3-pixel geometric tolerance. All measured regions remain in the CSV records; the table reports the prespecified high-NIR region.

## Reproduction scope and validation

```bash
python -m unittest discover -s tests -v
```

This release supports color inference, two analytic controls and Table V aggregation. It does not yet provide a complete reproduction of training, dataset registration, every manuscript figure/table or all external baselines. Original training uses an intensity-domain warm start and frozen stereo networks, and is not training from scratch.

Nine tests cover analytic invariants, input validation, offline dependency setup, repeatability and preservation of local modifications. On one original MS2 frame, the external-backbone CPU output equals the research implementation exactly, and both analytic outputs equal the cached paper controls. See [validation](docs/VALIDATION.json), [restoration checks](docs/RESTORE_VALIDATION.json), and [tested environment](environment-tested.txt). A fresh environment dependency installation has not yet been tested.

Historical inactive options remain in the model source for compatibility; the inference loader validates the paper configuration before loading weights. The previous intensity-domain implementation remains available in [commit 0728f6a](https://github.com/Darhan7/Learning-a-Single-Channel-Fusion-Representation-from-RGB-and-Active-NIR-via-Self-Supervision/tree/0728f6a6972e40f0c6f6f0ed27a555605dbafab8). It should not be mistaken for this LaminaFuse release.

## Attribution and license status

The externally loaded encoder and attention blocks follow Jinnyeong Kim and Seung-Hwan Baek, *Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision*, CVPR 2025. Please cite their work when using those components or the dataset. Obtain PixelNIR and MS2 data from their official providers; no dataset images are bundled here.

A license for LaminaFuse-owned code and checkpoint has not yet been selected. Public availability should not be interpreted as an MIT or Apache license grant. Third-party dependencies retain their own terms; fetching them does not create additional permissions. See [third-party notices](docs/THIRD_PARTY_NOTICES.txt). This repository makes no claim that the manuscript has been published or accepted.
