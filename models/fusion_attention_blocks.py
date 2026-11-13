"""Load the externally supplied PixelNIR backbone without redistributing it."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

release_root = Path(__file__).resolve().parents[1]
root = Path(os.environ.get('PIXELNIR_CODE') or release_root / 'external' / 'PixelNIR').expanduser()
if not root.is_dir():
    raise ImportError('Run python scripts/prepare_backbone.py, or set PIXELNIR_CODE to the pinned official PixelNIR checkout. Analytic modes do not require it.')
metadata = json.loads((release_root / 'docs/upstream.json').read_text())

def load(relative, name):
    path = Path(root) / relative
    if not path.is_file():
        raise ImportError(f'Missing external dependency: {path}')
    expected = metadata['files'][relative]['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ImportError(f'{relative} differs from the validated PixelNIR revision {metadata["commit"]}. Use the pinned checkout in README.md.')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

extractor = load('net/extractor.py', '_lamina_external_pixelnir_extractor')
fusion = load('net/fusion.py', '_lamina_external_pixelnir_fusion')
BasicEncoder = extractor.BasicEncoder
ResidualBlock = extractor.ResidualBlock
BAttentionFeatureFusion = fusion.BAttentionFeatureFusion
