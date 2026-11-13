"""Offline checks for dependency restoration and preservation of local files."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.prepare_backbone import REQUIRED, prepare

class BackboneSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'official'
        self.dest = self.root / 'external' / 'PixelNIR'
        self.files = {name: f'# fixture for {name}\n'.encode() for name in REQUIRED}
        manifest = {'repository':'https://example.invalid', 'commit':'pinned', 'files':{}}
        for name, contents in self.files.items():
            p = self.source / name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(contents)
            manifest['files'][name] = {'sha256':hashlib.sha256(contents).hexdigest()}
        mocked = patch('scripts.prepare_backbone.metadata', return_value=manifest)
        mocked.start();self.addCleanup(mocked.stop)

    def test_offline_restore_and_idempotence(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('Must remain offline')):
            prepare(self.dest, source_dir=self.source)
            prepare(self.dest, check_only=True)
            before = {p.relative_to(self.dest):(p.read_bytes(),p.stat().st_mtime_ns)
                      for p in self.dest.rglob('*') if p.is_file()}
            prepare(self.dest, source_dir=self.source)
            after = {p.relative_to(self.dest):(p.read_bytes(),p.stat().st_mtime_ns)
                     for p in self.dest.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        for name,data in self.files.items():self.assertEqual((self.dest/name).read_bytes(),data)

    def test_modified_destination_is_preserved(self):
        prepare(self.dest, source_dir=self.source)
        changed = self.dest / REQUIRED[0];changed.write_bytes(b'local modification\n')
        with self.assertRaises(ValueError):prepare(self.dest, source_dir=self.source)
        self.assertEqual(changed.read_bytes(),b'local modification\n')
        self.assertEqual((self.dest/REQUIRED[1]).read_bytes(),self.files[REQUIRED[1]])

    def test_bad_source_does_not_leave_partial_install(self):
        (self.source/REQUIRED[1]).write_bytes(b'corrupted download\n')
        with self.assertRaises(ValueError):prepare(self.dest, source_dir=self.source)
        self.assertFalse(self.dest.exists())

    def test_missing_install_check_does_not_write(self):
        with self.assertRaises(ValueError):prepare(self.dest, check_only=True)
        self.assertFalse(self.dest.parent.exists())

if __name__=='__main__':unittest.main()
