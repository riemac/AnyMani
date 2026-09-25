"""Exercise bundle integrity and extraction through the public command."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


COMMAND = Path(__file__).resolve().parents[1] / "scripts/download.py"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "models.tar.gz"
        self.manifest = self.root / "release.json"
        self.destination = self.root / "data"

    def tearDown(self):
        self.temp.cleanup()

    def make_bundle(self, extra=None, prefix=""):
        with tarfile.open(self.archive, "w:gz") as bundle:
            payload = b'{"models": {}}\n'
            info = tarfile.TarInfo(prefix + "model_manifest.json")
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
            if extra is not None:
                bundle.addfile(extra)
        blob = self.archive.read_bytes()
        spec = {"url": self.archive.as_uri(), "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        if prefix:
            spec["archive_root"] = prefix.rstrip("/")
        self.manifest.write_text(json.dumps({"artifacts": {"models": spec}}))

    def run_command(self):
        return subprocess.run(
            [
                sys.executable,
                str(COMMAND),
                "--component",
                "models",
                "--manifest",
                str(self.manifest),
                "--data-dir",
                str(self.destination),
            ],
            text=True,
            capture_output=True,
        )

    def test_install_and_cached_repeat(self):
        self.make_bundle()
        first = self.run_command()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue((self.destination / "models/model_manifest.json").is_file())
        self.archive.unlink()
        second = self.run_command()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already installed", second.stdout)

    def test_checksum_failure_does_not_install(self):
        self.make_bundle()
        self.archive.write_bytes(self.archive.read_bytes() + b"corrupt")
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse((self.destination / "models").exists())

    def test_versioned_archive_root(self):
        self.make_bundle(prefix="anymani-paper-models-v1.0.0/")
        result = self.run_command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.destination / "models/model_manifest.json").is_file())

    def test_escaping_link_is_rejected(self):
        link = tarfile.TarInfo("outside")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        self.make_bundle(link)
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.destination / "models").exists())
        self.assertFalse((self.destination / "outside").exists())

    def test_existing_unmanaged_directory_is_preserved(self):
        self.make_bundle()
        target = self.destination / "models"
        target.mkdir(parents=True)
        sentinel = target / "user-file.txt"
        sentinel.write_text("keep")
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
