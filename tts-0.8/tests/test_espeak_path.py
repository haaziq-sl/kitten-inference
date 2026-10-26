"""The data folder path given to espeak-ng (kitten_inference/tts08/_espeak_path.py): long paths, and on Windows paths
outside ASCII, would make espeak-ng end the process, so they are replaced by a short copy."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from kitten_inference.tts08 import _espeak_path as ep


def _make_data(folder):
    os.makedirs(os.path.join(folder, "voices", "!v"))
    for rel, data in (("phontab", b"phonemes"), ("en_dict", b"\x00\x01dict" * 100), ("voices/!v/m1", b"voice")):
        with open(os.path.join(folder, *rel.split("/")), "wb") as f:
            f.write(data)
    return folder


class EspeakPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kesp-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def deep_data(self):
        # tmp/<pad>/espeak-ng-data, at least 170 bytes (espeak-ng fails from 160)
        pad = "d" * max(10, 170 - len(self.tmp) - len("/espeak-ng-data") - 2)
        return _make_data(os.path.join(self.tmp, pad, "espeak-ng-data"))

    def test_rules(self):
        self.assertTrue(ep.fits("/usr/lib/espeak-ng-data", windows=False))
        self.assertFalse(ep.fits("/" + "x" * 149, windows=False))
        self.assertFalse(ep.fits("/" + "\u00e9" * 75, windows=False))  # bytes count, not characters
        self.assertTrue(ep.fits("/opt/Jos\u00e9/espeak-ng-data", windows=False))
        self.assertFalse(ep.fits("C:\\Users\\Jos\u00e9\\espeak-ng-data", windows=True))
        self.assertFalse(ep.fits("C:\\Users\\\u0141ukasz\\espeak-ng-data", windows=True))
        self.assertTrue(ep.fits("C:\\Users\\JOSE~1\\espeak-ng-data", windows=True))

    def test_a_short_path_is_used_as_it_is(self):
        src = _make_data(os.path.join(self.tmp, "espeak-ng-data"))
        if not ep.fits(src):
            self.skipTest("this machine's temp folder is not short ASCII")
        root = os.path.join(self.tmp, "root")
        self.assertEqual(ep._choose(src, "1.0", [([], root)]), src)
        self.assertFalse(os.path.exists(root))

    def test_a_deep_path_gets_one_short_copy_that_is_reused(self):
        src = self.deep_data()
        self.assertFalse(ep.fits(src))
        short = tempfile.mkdtemp(prefix="kesr-")
        self.addCleanup(shutil.rmtree, short, True)
        root = os.path.join(short, "espeak")
        path = ep._choose(src, "0.2.4", [([], root)])
        self.assertEqual(ep.how["how"], "copy")
        self.assertTrue(ep.fits(path), path)
        self.assertEqual(os.path.dirname(path), root)
        self.assertTrue(os.path.basename(path).startswith("0.2.4-"))
        self.assertEqual(ep.content_digest(path), ep.content_digest(src))
        with open(os.path.join(path, ep.MARKER), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["sha256"], ep.content_digest(src))
        stamp = os.stat(os.path.join(path, ep.MARKER)).st_mtime_ns
        self.assertEqual(ep._choose(src, "0.2.4", [([], root)]), path)  # reused, not copied again
        self.assertEqual(os.stat(os.path.join(path, ep.MARKER)).st_mtime_ns, stamp)
        self.assertEqual(os.listdir(root), [os.path.basename(path)])  # no staging folder left behind
        if os.name == "posix":
            self.assertEqual(os.stat(root).st_mode & 0o077, 0)

    def test_an_incomplete_copy_is_replaced(self):
        src = self.deep_data()
        short = tempfile.mkdtemp(prefix="kesr-")
        self.addCleanup(shutil.rmtree, short, True)
        root = os.path.join(short, "espeak")
        path = ep._choose(src, "1", [([], root)])
        os.remove(os.path.join(path, ep.MARKER))
        self.assertEqual(ep._choose(src, "1", [([], root)]), path)
        self.assertTrue(os.path.isfile(os.path.join(path, ep.MARKER)))

    def test_processes_starting_together_share_one_copy(self):
        src = self.deep_data()
        short = tempfile.mkdtemp(prefix="kesr-")
        self.addCleanup(shutil.rmtree, short, True)
        root = os.path.join(short, "espeak")
        code = ("import json, sys; from kitten_inference.tts08 import _espeak_path as ep; "
                "print(ep._choose(sys.argv[1], '1', [([], sys.argv[2])]))")
        procs = [subprocess.Popen([sys.executable, "-c", code, src, root], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE) for _ in range(4)]
        outs = [p.communicate(timeout=300) for p in procs]
        for p, (out, err) in zip(procs, outs):
            self.assertEqual(p.returncode, 0, err.decode("utf-8", "replace"))
        paths = {out.decode("utf-8").strip() for out, _ in outs}
        self.assertEqual(len(paths), 1, paths)
        self.assertEqual(os.listdir(root), [os.path.basename(paths.pop())])

    def test_no_usable_folder_raises_a_clear_error(self):
        src = self.deep_data()
        too_deep = os.path.join(self.tmp, "r" * 160)
        with self.assertRaises(ep.EspeakPathError) as cm:
            ep._choose(src, "1", [([], too_deep)])
        message = str(cm.exception)
        self.assertIn("would end the Python process", message)
        self.assertIn("Fix:", message)
        self.assertIn(src, message)

    @unittest.skipUnless(os.name == "posix", "symbolic links")
    def test_a_link_in_place_of_the_folder_is_refused(self):
        src = self.deep_data()
        short = tempfile.mkdtemp(prefix="kesr-")
        self.addCleanup(shutil.rmtree, short, True)
        elsewhere = os.path.join(short, "elsewhere")
        os.mkdir(elsewhere, 0o700)
        os.symlink(elsewhere, os.path.join(short, "espeak"))
        with self.assertRaises(ep.EspeakPathError) as cm:
            ep._choose(src, "1", [([], os.path.join(short, "espeak"))])
        self.assertIn("link", str(cm.exception))
        self.assertEqual(os.listdir(elsewhere), [])

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "getuid") and os.getuid() != 0, "folder modes")
    def test_a_folder_others_can_change_is_refused(self):
        src = self.deep_data()
        short = tempfile.mkdtemp(prefix="kesr-")
        self.addCleanup(shutil.rmtree, short, True)
        shared = os.path.join(short, "espeak")
        os.mkdir(shared)
        os.chmod(shared, 0o777)
        with self.assertRaises(ep.EspeakPathError) as cm:
            ep._choose(src, "1", [([], shared)])
        self.assertIn("other users can change it", str(cm.exception))


class WindowsLongPathTest(unittest.TestCase):
    def test_longest_engine_path(self):
        from kitten_inference.tts08 import cpp_model
        tmp = tempfile.mkdtemp(prefix="kelp-")
        self.addCleanup(shutil.rmtree, tmp, True)
        here = os.path.join(tmp, "site", "kitten_inference", "tts08")
        libs = os.path.join(tmp, "site", "kitten_inference_tts08.libs")
        os.makedirs(here)
        os.makedirs(libs)
        for path in (os.path.join(here, "_engine.cp312-win_amd64.pyd"), os.path.join(here, "cpp_model.py"),
                     os.path.join(libs, "msvcp140-0123456789abcdef0123456789abcdef.dll")):
            open(path, "wb").close()
        self.assertEqual(cpp_model._longest_engine_path(here),
                         len(os.path.join(libs, "msvcp140-0123456789abcdef0123456789abcdef.dll")))


if __name__ == "__main__":
    unittest.main()
