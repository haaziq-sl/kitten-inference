#!/usr/bin/env python3
"""`kitten-tts-native --check-assets` against small fake bundles (no GPU needed).

Set KT2N_CLI_BINARY to the kitten-tts-native executable; without it every test
is skipped. Runs under pytest or directly (`python3 test_bundle.py`, exit 77
when skipped).
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = os.environ.get("KT2N_CLI_BINARY", "")
CLONING_FILES = ("s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors")
REFERENCE = ("--reference-audio", "reference.wav", "--reference-transcript", "A reference recording.")


@unittest.skipUnless(BINARY, "KT2N_CLI_BINARY is not set")
class CheckAssets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        native = {"version": 1, "decoder": "default"}
        for key in ("lm", "acoustic", "voices", "tokenizer"):
            native[key] = self.entry("native/" + key)
        native["text"] = {"directory": "native/text", "files": [self.entry("native/text/rules")]}
        self.native = native

    def tearDown(self):
        self._tmp.cleanup()

    def entry(self, name, data=b"test"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"file": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def add_cloning(self, names=CLONING_FILES, directory="native/cloning"):
        self.native["cloning"] = {"version": 1, "directory": directory,
                                  "files": [self.entry("native/cloning/" + name) for name in names]}

    def run_cli(self, *options, status=0, check=True):
        (self.root / "config.json").write_text(json.dumps({"type": "KITTEN2", "native": self.native}))
        command = [BINARY, "--assets", str(self.root)] + (["--check-assets"] if check else []) + list(options)
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, status, f"{command}\nstdout: {result.stdout}\nstderr: {result.stderr}")
        return result

    def test_valid_bundle(self):
        result = self.run_cli()
        self.assertIn("All 5 files", result.stdout)
        self.assertIn("Voice cloning: not prepared", result.stdout)
        self.run_cli("--stream", "true")
        self.run_cli("--stream", "false")

    def test_usage_errors(self):
        for value in ("0", "1", "False", "no"):
            self.assertIn("--stream must be true or false", self.run_cli("--stream", value, status=2).stderr)
        self.assertIn("unknown option --reference-language", self.run_cli("--reference-language", "en", status=2).stderr)
        for dropped in ("--backend", "--repo", "--revision", "--cache-dir", "--decoder"):
            self.assertIn("unknown option " + dropped, self.run_cli(dropped, "x", status=2).stderr)
        self.assertIn("cannot be combined", self.run_cli(*REFERENCE, "--voice", "Bruno", status=2).stderr)
        self.assertIn("--reference-transcript is required", self.run_cli("--reference-audio", "reference.wav", status=2).stderr)
        self.assertIn("requires --reference-audio", self.run_cli("--reference-transcript", "Reference.", status=2).stderr)
        self.assertIn("--text or --text-file is required", self.run_cli(status=2, check=False).stderr)

    def test_sha_and_size_mismatch(self):
        self.run_cli()
        (self.root / "native/lm").write_bytes(b"evil")
        self.assertIn("SHA256 mismatch", self.run_cli(status=1).stderr)
        (self.root / "native/lm").write_bytes(b"longer")
        self.assertIn("size mismatch", self.run_cli(status=1).stderr)
        (self.root / "native/lm").unlink()
        self.assertIn("missing asset", self.run_cli(status=1).stderr)

    def test_path_escape(self):
        (self.root.parent / (self.root.name + "-escape")).write_bytes(b"test")
        try:
            self.native["lm"]["file"] = "../" + self.root.name + "-escape"
            self.assertIn("invalid path", self.run_cli(status=1).stderr)
        finally:
            (self.root.parent / (self.root.name + "-escape")).unlink()
        self.native["lm"]["file"] = "/etc/hostname"
        self.assertIn("invalid path", self.run_cli(status=1).stderr)

    def test_text_dir_escape(self):
        self.native["text"]["directory"] = "elsewhere"
        self.assertIn("outside its directory", self.run_cli(status=1).stderr)
        self.native["text"]["directory"] = "../native/text"
        self.assertIn("invalid path", self.run_cli(status=1).stderr)
        self.native["text"] = {"directory": "native/text", "files": []}
        self.assertIn("asset list is empty", self.run_cli(status=1).stderr)

    def test_cloning_files(self):
        self.assertIn("no voice cloning files", self.run_cli(*REFERENCE, status=1).stderr)
        self.add_cloning()
        self.assertIn("Voice cloning: available", self.run_cli(*REFERENCE).stdout)
        self.assertIn("All 8 files", self.run_cli().stdout)
        (self.root / "native/cloning/voicebuild_consts.safetensors").write_bytes(b"evil")
        self.assertIn("SHA256", self.run_cli(*REFERENCE, status=1).stderr)
        self.assertIn("SHA256", self.run_cli(status=1).stderr)  # --check-assets checks every listed file

    def test_missing_cloning_files(self):
        self.add_cloning(names=CLONING_FILES[:1])
        self.assertIn("missing cloning asset: voicebuild_consts.safetensors", self.run_cli(*REFERENCE, status=1).stderr)

    def test_cloning_dir_escape(self):
        self.add_cloning(directory="elsewhere")
        self.assertIn("outside its directory", self.run_cli(*REFERENCE, status=1).stderr)

    def test_manifest_errors(self):
        (self.root / "config.json").write_text(json.dumps({"type": "KITTEN2"}))
        result = subprocess.run([BINARY, "--assets", str(self.root), "--check-assets"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("prepare_assets.py", result.stderr)
        self.native["decoder"] = "student_w4"
        self.assertIn("unsupported asset manifest", self.run_cli(status=1).stderr)

    def test_list_voices_rejects_bad_voices_file(self):
        self.assertIn("cannot read the voices file", self.run_cli("--list-voices", status=1, check=False).stderr)

    def test_help_and_version(self):
        result = subprocess.run([BINARY, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        for option in ("--assets", "--list-voices", "--save-voice", "--load-voice", "--check-assets", "--text-file"):
            self.assertIn(option, result.stdout)
        result = subprocess.run([BINARY, "--version"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith("kitten-tts-native "), result.stdout)


if __name__ == "__main__":
    if not BINARY:
        print("SKIP: set KT2N_CLI_BINARY to the kitten-tts-native executable")
        sys.exit(77)
    program = unittest.main(exit=False, verbosity=2)
    sys.exit(0 if program.result.wasSuccessful() else 1)
