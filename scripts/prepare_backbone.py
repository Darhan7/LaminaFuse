"""Fetch the two pinned PixelNIR modules required by LaminaFuse inference.

The downloaded source stays in external/ (excluded from the source release).
This command never executes downloaded code or overwrites a modified checkout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / 'external' / 'PixelNIR'
REQUIRED = ('net/extractor.py', 'net/fusion.py')
MAX_FILE_BYTES = 2 * 1024 * 1024


def metadata():
    return json.loads((ROOT / 'docs/upstream.json').read_text())


def checked_bytes(data, relative, spec):
    if hashlib.sha256(data).hexdigest() != spec['sha256']:
        raise ValueError(f'{relative}: SHA-256 mismatch; no files installed. Expected the pinned revision in docs/upstream.json.')
    return data


def verify(destination, manifest):
    for relative in REQUIRED:
        path = destination / relative
        if not path.is_file():
            raise ValueError(f'{path} is missing. Existing destinations are not modified; choose a new --dest or restore the missing file.')
        checked_bytes(path.read_bytes(), relative, manifest['files'][relative])


def prepare(destination, *, source_dir=None, check_only=False, timeout=20.0):
    manifest = metadata()
    if destination.exists():
        verify(destination, manifest)
        return 'already present and verified'
    if check_only:
        raise ValueError(f'{destination} is absent; run without --check to prepare it.')
    # Read and verify everything before creating the destination.
    payloads = {}
    for relative in REQUIRED:
        spec = manifest['files'][relative]
        if source_dir is not None:
            data = (source_dir / relative).read_bytes()
        else:
            expected_prefix = ('https://raw.githubusercontent.com/divisonofficer/'
                               'Pixel_aligned_RGB_NIR_Stereo/' + manifest['commit'] + '/')
            if spec['url'] != expected_prefix + relative:
                raise ValueError(f'Unexpected upstream URL for {relative}')
            request = urllib.request.Request(spec['url'], headers={'User-Agent':'LaminaFuse-backbone-setup'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f'{relative} exceeds the expected source-file size limit.')
        payloads[relative] = checked_bytes(data, relative, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pixelnir-setup-', dir=destination.parent) as td:
        staged = Path(td) / 'PixelNIR'
        for relative, data in payloads.items():
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        receipt = {'repository':manifest['repository'], 'commit':manifest['commit'],
                   'files':{name:manifest['files'][name] for name in REQUIRED},
                   'note':'Fetched external dependency. Upstream terms apply; this is not a new license grant.'}
        (staged / 'UPSTREAM.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # A concurrent or modified destination must never be replaced.
        if destination.exists():
            verify(destination, manifest)
            return 'already present and verified'
        staged.rename(destination)
    verify(destination, manifest)
    return 'prepared and verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST,
                        help='Default: repository external/PixelNIR; custom paths require PIXELNIR_CODE.')
    parser.add_argument('--source-dir', type=Path,
                        help='Copy from a local pinned official checkout, without network access.')
    parser.add_argument('--check', action='store_true', help='Verify installed files without downloading or modifying them.')
    parser.add_argument('--timeout', type=float, default=20, help='Network timeout in seconds per file (default: 20).')
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error('--timeout must be in (0,60] seconds')
    if args.source_dir and args.check:
        parser.error('--source-dir is not used with --check')
    destination = args.dest.expanduser().absolute()
    source = args.source_dir.expanduser().absolute() if args.source_dir else None
    try:
        status = prepare(destination, source_dir=source, check_only=args.check, timeout=args.timeout)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Backbone setup failed: {error}\n')
    print(f'PixelNIR {status}: {destination}')
    print('Revision:', metadata()['commit'])
    if destination.resolve() == DEFAULT_DEST.resolve():
        print('LaminaFuse will find this location automatically; PIXELNIR_CODE is optional.')
    else:
        print('Set PIXELNIR_CODE to this absolute directory before learned inference.')
    if os.environ.get('PIXELNIR_CODE') and Path(os.environ['PIXELNIR_CODE']).expanduser().resolve() != destination.resolve():
        print('Note: your current PIXELNIR_CODE overrides this location. Update or unset it to use these files.')


if __name__ == '__main__':
    main()
