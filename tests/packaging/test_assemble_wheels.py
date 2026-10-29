"""The release wheel assembler (.github/scripts/assemble_wheels.py), its independent check (check_assembled.py) and
the lock writer (lock_inputs.py), on small synthetic inputs (stub_inputs.py): no network and no real engines. Needs
git and the packaging module."""
import hashlib
import json
import shutil
import struct
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stub_inputs as stub  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not stub.SCRIPTS.is_dir(), reason="not a source checkout (.github/scripts is not here)"),
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
]
pytest.importorskip("packaging")

INFO = "kitten_inference-0.2.0.dist-info"
LINUX = "kitten_inference-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl"
WINDOWS = "kitten_inference-0.2.0-cp312-cp312-win_amd64.whl"
MAC = "kitten_inference-0.2.0-cp312-cp312-macosx_11_0_arm64.whl"


@pytest.fixture
def world(tmp_path):
    return stub.World(tmp_path)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    w = stub.World(tmp_path_factory.mktemp("built"))
    return w, w.build()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(path: Path, name: str):
    """A member's compressed bytes and the header fields a raw copy keeps."""
    with zipfile.ZipFile(path) as z:
        i = z.getinfo(name)
    with open(path, "rb") as f:
        f.seek(i.header_offset + 26)
        n, e = struct.unpack("<HH", f.read(4))
        f.seek(i.header_offset + 30 + n + e)
        return f.read(i.compress_size), i.CRC, i.compress_type, i.external_attr, i.create_system, i.date_time


def record_lines(path: Path) -> list:
    with zipfile.ZipFile(path) as z:
        return z.read(next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))).decode().splitlines()


def lock_file(world, file):
    """Points the lock's row for `file` at the file as it is now (so only the other checks can refuse it)."""
    lock = world.lock()
    row = next(r for r in lock["inputs"] if r["file"] == file)
    row["sha256"], row["size"] = sha(world.inputs / file), (world.inputs / file).stat().st_size
    world.save_lock(lock)


def test_one_wheel_per_tts08_input_with_engines_from_cp39(built):
    world, out = built
    assert sorted(p.name for p in out.glob("*.whl")) == sorted([
        "kitten_inference-0.2.0-cp38-cp38-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        LINUX,
        "kitten_inference-0.2.0-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        MAC,
        "kitten_inference-0.2.0-cp38-cp38-win_amd64.whl",
        WINDOWS,
        "kitten_inference-0.2.0-cp313-cp313-android_24_arm64_v8a.whl"])
    for p in out.glob("*.whl"):
        names = zipfile.ZipFile(p).namelist()
        engines = {e for e in ("cuda", "rocm") if any(n.startswith(f"kitten_inference/_tts2_{e}/") for n in names)}
        assert engines == {LINUX: {"cuda", "rocm"}, WINDOWS: {"cuda"}}.get(p.name, set()), p.name
        tags = [line for line in zipfile.ZipFile(p).read(f"{INFO}/WHEEL").decode().splitlines()
                if line.startswith("Tag:")]
        if p.name == LINUX:  # the engines were built for manylinux_2_28 only: the 2_27 alias goes
            assert tags == ["Tag: cp312-cp312-manylinux_2_28_x86_64"]
        if p.name.startswith("kitten_inference-0.2.0-cp38-cp38-manylinux"):  # cp38: TTS 0.8 only, its own tags
            assert tags == ["Tag: cp38-cp38-manylinux_2_27_x86_64", "Tag: cp38-cp38-manylinux_2_28_x86_64"]


def test_copies_keep_bytes_modes_and_record_rows(built):
    world, out = built
    cuda = world.inputs / stub.PAYLOADS["cuda-linux"][0]
    rocm = world.inputs / stub.PAYLOADS["rocm"][0]
    for source, name in ((cuda, "kitten_inference/_tts2_cuda/bin/kitten-tts-native"),
                         (rocm, "kitten_inference/_tts2_rocm/lib/libkt2n_engine.so")):
        assert raw(out / LINUX, name) == raw(source, name)
        assert raw(out / LINUX, name)[3] >> 16 == 0o100755
    w = world.inputs / "kitten_inference_tts08-0.2.0-cp312-cp312-win_amd64.whl"
    dll = "kitten_inference/tts08/_msvc/vcruntime140_1.dll"
    assert raw(out / WINDOWS, dll) == raw(w, dll)  # Windows creator, mode 0600 and its own date kept
    for source, target in ((cuda, LINUX), (rocm, LINUX), (w, WINDOWS),
                           (world.inputs / "kitten_inference_tts08-0.2.0-cp312-cp312-macosx_11_0_arm64.whl", MAC)):
        names = set(zipfile.ZipFile(out / target).namelist())
        # The macOS input's RECORD has CRLF line ends: a copied row is the same string without the line end.
        copied = [line for line in record_lines(source) if line.split(",", 1)[0] in names
                  and line.split(",", 1)[0] not in stub.OVERLAY and ".dist-info/" not in line]
        assert copied and set(copied) <= set(record_lines(out / target))


def test_one_new_dist_info(built):
    world, out = built
    metadata = set()
    for p in out.glob("*.whl"):
        z = zipfile.ZipFile(p)
        names = z.namelist()
        assert {n.split("/", 1)[0] for n in names if ".dist-info" in n.split("/", 1)[0]} == {INFO}
        assert names[-1] == f"{INFO}/RECORD" and not any(n.endswith("/") for n in names)
        assert not any("/sboms/" in n or "/_tts2_rocm/licenses/" in n or "/top_level.txt" in n for n in names)
        meta = z.read(f"{INFO}/METADATA")
        metadata.add(meta)
        listed = sorted(line[len("License-File: "):] for line in meta.decode().split("\n\n")[0].splitlines()
                        if line.startswith("License-File: "))
        assert listed == sorted(n[len(f"{INFO}/licenses/"):] for n in names if n.startswith(f"{INFO}/licenses/"))
        assert "tts-2/licenses/lgpl-source/atempo.cc" in listed and "tts-2/licenses/rocm/HIP.txt" in listed
        assert z.read(f"{INFO}/licenses/tts-2/licenses/lgpl-source/atempo.h") == stub.SOURCE[
            "engines/tts2/include/kt2n/post/atempo.h"]
        assert "Root-Is-Purelib: false" in z.read(f"{INFO}/WHEEL").decode()
        assert b"kitten-tts = kitten_inference._cli:main" in z.read(f"{INFO}/entry_points.txt")
        assert (f"{INFO}/DELVEWHEEL" in names) == p.name.endswith("win_amd64.whl")
    assert len(metadata) == 1


def test_overlay_files_come_from_the_release_commit(built):
    world, out = built
    for p in out.glob("*.whl"):
        z = zipfile.ZipFile(p)
        for path, data in stub.RELEASE.items():
            assert z.read(path) == data
            assert z.getinfo(path).date_time == (2020, 2, 2, 0, 0, 0)
    z = zipfile.ZipFile(out / WINDOWS)
    assert b"delvewheel patch" in z.read("kitten_inference/tts08/__init__.py")  # the tested file, never replaced
    assert z.read("kitten_inference/tts08/preprocess.py").endswith(b"\r\n")


def test_check_assembled_passes(built):
    world, out = built
    seen = []
    problems, check = world.check(out, elf=lambda path: seen.append(path.name))
    assert problems == []
    assert check.engine_rows_seen == {(r["part"], r["path"]) for r in world.lock()["engine_files"]}
    assert len(check.engine_rows_seen) == 12
    # every distinct ELF file of the two manylinux_2_28_x86_64 wheels, once: 2 TTS 0.8 + 6 engine files, + 2 for cp38
    assert check.elf_files == len(seen) == 10


def test_the_check_is_independent_of_the_assembler():
    text = (stub.SCRIPTS / "check_assembled.py").read_text(encoding="utf-8")
    assert "import assemble_wheels" not in text and "from assemble_wheels" not in text


def test_two_builds_give_the_same_bytes(world):
    a = world.build("a")
    stub.git(world.source, "commit", "--allow-empty", "-q", "-m", "another commit, same tree")
    b = world.build("b")
    assert {p.name: sha(p) for p in a.glob("*.whl")} == {p.name: sha(p) for p in b.glob("*.whl")}
    report = json.loads((b / "assembly-report.json").read_text(encoding="utf-8"))
    assert len(report["wheels"]) == len(stub.TTS08) and report["inputs_checked"] == len(stub.TTS08) + 3


def rewrite_member(path: Path, name: str, data: bytes):
    """Changes one member of a finished wheel and its RECORD row to match."""
    with zipfile.ZipFile(path) as z:
        items = [(i, z.read(i)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w") as z:
        for i, content in items:
            if i.filename == name:
                content = data
            elif i.filename.endswith(".dist-info/RECORD"):
                content = "\n".join(f"{name},{stub.b64(data)},{len(data)}" if line.startswith(name + ",") else line
                                    for line in content.decode().splitlines()).encode() + b"\n"
            z.writestr(i, content)


def test_check_assembled_finds_changed_files(world):
    out = world.build()
    rewrite_member(out / MAC, "kitten_inference/tts08/preprocess.py", b"x = 2\n")
    (world.source / "src/kitten_inference/tts08/cpp_model.py").write_bytes(b"MESSAGE = 'other'\n")
    problems, _ = world.check(out, elf=lambda path: "needs GLIBC_2.34" if "libkt2n" in path.name else None)
    assert any(f"{MAC}: kitten_inference/tts08/preprocess.py differs from kitten_inference_tts08" in p
               for p in problems), problems
    assert any("overlay file kitten_inference/tts08/cpp_model.py differs" in p for p in problems), problems
    assert any("needs GLIBC_2.34" in p for p in problems), problems


def test_check_assembled_limits_path_length(tmp_path):
    long = "engines/tts2/licenses/" + "x" * 50 + ".txt"  # 115 characters under the dist-info folder of a .post99
    world = stub.World(tmp_path, extra_source={long: b"long\n"})
    problems, _ = world.check(world.build())
    assert problems and all("characters (with a .post99 version), over 90" in p for p in problems), problems


def test_inputs_must_match_the_lock(world):
    aw, _, _ = stub.scripts()
    lock = world.lock()
    lock["inputs"][0]["sha256"] = "0" * 64
    world.save_lock(lock)
    with pytest.raises(SystemExit, match="its sha256 differs from the lock"):
        world.build()
    world.relock()
    (world.inputs / stub.PAYLOADS["rocm"][0]).unlink()
    with pytest.raises(SystemExit, match="kitten_inference_tts2_rocm-0.2.0-py3-none-manylinux_2_28_x86_64.whl "
                                         r"\(artifact kitten-inference-tts2-rocm-manylinux_2_28_x86_64 of run 3\) "
                                         "is not in"):
        aw.main(["verify-inputs", "--lock", str(world.lock_path), "--inputs", str(world.inputs)])


def test_a_member_that_does_not_match_its_record_is_refused(world):
    file = stub.PAYLOADS["cuda-linux"][0]
    members = [stub.M(m.name, m.data + b"!" if m.name.endswith("kitten-tts-native") else m.data, m.mode, m.system,
                      m.date) for m in world.members(file)]
    world.rewrite(file, members, stale_record=True, relock=False)
    lock_file(world, file)
    with pytest.raises(SystemExit, match="kitten-tts-native does not match its RECORD row"):
        world.build()


def test_a_path_in_an_archive_twice_is_refused(world):
    file = stub.PAYLOADS["rocm"][0]
    members = world.members(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate name
        world.rewrite(file, members + [next(m for m in members if m.name.endswith("/__init__.py"))], relock=False)
    lock_file(world, file)
    with pytest.raises(SystemExit, match="_tts2_rocm/__init__.py is in the archive twice"):
        world.build()


@pytest.mark.parametrize("which, extra, match", [
    ("cuda-linux", "kitten_inference/tts2/_cli.py", r"is outside kitten_inference/_tts2_cuda/"),
    ("tts08", "kitten_inference/other.py", r"is outside kitten_inference/tts08/ and kitten_inference_tts08.libs/"),
    ("umbrella", "kitten_inference/tts08/x.py", r"expected only kitten_inference/__init__.py"),
])
def test_members_outside_their_part_are_refused(world, which, extra, match):
    file = {"cuda-linux": stub.PAYLOADS["cuda-linux"][0], "umbrella": "kitten_inference-0.2.0-py3-none-any.whl",
            "tts08": "kitten_inference_tts08-0.2.0-cp312-cp312-macosx_11_0_arm64.whl"}[which]
    world.rewrite(file, world.members(file) + [stub.M(extra, b"x = 1\n")], relock=False)
    if which != "umbrella":
        lock_file(world, file)
    with pytest.raises(SystemExit, match=match):
        world.build()


def test_the_overlay_never_replaces_init(world):
    lock = world.lock()
    lock["tts08_python_overlay"].append("kitten_inference/tts08/__init__.py")
    world.save_lock(lock)
    with pytest.raises(SystemExit, match="other than __init__.py"):
        world.build()


@pytest.mark.parametrize("change, match", [
    (lambda reqs: [r.replace("numpy>=1.23", "numpy>=1.26") for r in reqs], "root dependency table"),
    (lambda reqs: [r.split(";")[0] if r.startswith("tokenizers") else r for r in reqs], "root dependency table"),
    (lambda reqs: [r for r in reqs if not r.startswith("filelock")], "root dependency table"),
    (lambda reqs: reqs + ["kitten-inference-tts08>=0.2"], "invalid package dependency"),
    (lambda reqs: reqs + ["rich>=13"], "root dependency table"),
])
def test_requirement_drift_is_refused(tmp_path, change, match):
    world = stub.World(tmp_path, umbrella_requires=change(list(stub.UMBRELLA_REQUIRES)))
    with pytest.raises(SystemExit, match=match):
        world.build()


def test_post_releases_versions_and_the_engine_abi(tmp_path):
    world = stub.World(tmp_path / "post", umbrella_version="0.2.0.post1")
    out = world.build()
    assert all(p.name.startswith("kitten_inference-0.2.0.post1-") for p in out.glob("*.whl"))
    assert "kitten_inference-0.2.0.post1.dist-info/METADATA" in zipfile.ZipFile(out / LINUX.replace(
        "0.2.0", "0.2.0.post1")).namelist()
    assert world.check(out)[0] == []
    with pytest.raises(SystemExit, match="kitten-inference 0.3.0 is neither 0.2.0"):
        stub.World(tmp_path / "other", umbrella_version="0.3.0").build()
    with pytest.raises(SystemExit, match="engine ABI 3, kitten_inference.tts2 needs ABI 2"):
        stub.World(tmp_path / "abi", abi=3).build()


def test_engine_files_must_match_the_lock(world):
    lock = world.lock()
    lock["engine_files"][0]["sha256"] = "0" * 64
    world.save_lock(lock)
    with pytest.raises(SystemExit, match="differs from the lock's engine_files"):
        world.build()


def test_changed_engine_sources_are_refused(world):
    world.commit({"engines/tts2/src/engine.cc": b"// changed\n"}, "engine change")
    with pytest.raises(SystemExit, match=r"engines/tts2/src \(cuda\), engines/tts2/src \(rocm\).*run a new build and GPU-test"):
        world.build()


def test_only_the_overlay_python_may_change(world):
    world.commit({"src/kitten_inference/tts08/cpp_model.py": b"MESSAGE = 'again'\n"}, "overlay text")
    world.build("ok")
    world.commit({"src/kitten_inference/tts08/new.py": b"y = 1\n"}, "a new module")
    with pytest.raises(SystemExit, match=r"src/kitten_inference/tts08/new.py \(tts08\)"):
        world.build("refused")


def test_a_tag_run_needs_the_tested_tree(world, monkeypatch):
    _, _, li = stub.scripts()
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(SystemExit, match="tested_packaged_trees is not filled in"):
        world.build("a", tag_run=True)
    li.main(["--set-tested-trees", world.release_commit, "--out", str(world.lock_path), "--source",
             str(world.source)])
    assert set(world.lock()["tested_packaged_trees"]) >= {"umbrella", "engines/tts2/python", "src/kitten_inference/tts08"}
    world.build("b", tag_run=True)
    monkeypatch.setenv("GITHUB_REPOSITORY", "Example/Kitten-Inference")  # GitHub names ignore case
    world.build("b2", tag_run=True)
    monkeypatch.setenv("GITHUB_REPOSITORY", "KittenML/kitten-inference")
    with pytest.raises(SystemExit, match="this tag run is in KittenML/kitten-inference"):
        world.build("c", tag_run=True)
    monkeypatch.delenv("GITHUB_REPOSITORY")
    world.commit({"src/kitten_inference/__init__.py": b'__version__ = "0.2.0"  # changed\n'}, "umbrella")
    with pytest.raises(SystemExit, match="differ from the tested tree.*src/kitten_inference"):
        world.build("d", tag_run=True)
    world.build("e")  # a run that is not a tag run only warns


def test_lock_inputs_records_the_inputs(world):
    aw, _, li = stub.scripts()
    lock = world.lock()
    assert (lock["schema"], lock["version"], lock["tested_packaged_trees"]) == (2, "0.2.0", None)
    assert lock["tts08_python_overlay"] == stub.OVERLAY
    parts = [r["part"] for r in lock["inputs"]]
    assert parts.count("tts08") == len(stub.TTS08) and parts[-3:] == ["cuda-linux", "cuda-windows", "rocm"]
    assert all(r["artifact_digest"].startswith("sha256:") and r["artifact_id"] for r in lock["inputs"])
    modes = {(r["part"], r["path"]): r["mode"] for r in lock["engine_files"]}
    assert len(modes) == 12 and not any("licenses" in p for _, p in modes)
    assert modes[("cuda-linux", "_tts2_cuda/bin/kitten-tts-native")] == "0755"
    assert modes[("cuda-linux", "_tts2_cuda/lib/libkt2n_engine.so")] == "0644"
    assert modes[("rocm", "_tts2_rocm/lib/libkt2n_engine.so")] == "0755"
    trees = lock["source_trees"]
    assert set(trees["cuda"]) == set(trees["rocm"]) == set(aw.TTS2_SOURCES)
    assert trees["tts08"]["engines/tts08/CMakeLists.txt"] is None
    assert "src/kitten_inference/tts08/preprocess.py" in trees["tts08"]
    assert not any(p.endswith(("get_model.py", "_deps.py")) for p in trees["tts08"])  # the overlay is not recorded
    assert json.loads(li.dump(lock)) == lock


def test_lock_inputs_checks_each_wheel_against_its_artifact(world):
    aw, _, li = stub.scripts()
    artifacts = world.artifacts()
    artifacts[0]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(SystemExit, match="does not match the digest GitHub recorded"):
        li.make_lock("example/kitten-inference", world.runs(), artifacts, world.inputs, aw.Source(world.source))
    artifacts = world.artifacts()
    target = world.inputs / stub.PAYLOADS["rocm"][0]
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="is not the kitten_inference_tts2_rocm-0.2.0"):
        li.make_lock("example/kitten-inference", world.runs(), artifacts, world.inputs, aw.Source(world.source))


def test_lock_inputs_requires_successful_runs(monkeypatch):
    _, _, li = stub.scripts()
    run = {"path": ".github/workflows/tts-0.8.yml", "status": "completed", "conclusion": "failure",
           "repository": {"full_name": "o/r"}, "run_attempt": 2, "head_sha": "a" * 40}
    monkeypatch.setattr(li, "gh", lambda args, to=None: json.dumps(run).encode())
    with pytest.raises(SystemExit, match="completed/failure, not a success"):
        li.run_info("o/r", "tts08", 1)
    with pytest.raises(SystemExit, match="comes from tts-2-rocm.yml"):
        li.run_info("o/r", "rocm", 1)
    run["conclusion"] = "success"
    entry = {"id": 1, "attempt": 2, "commit": "a" * 40, "workflow": "tts-0.8.yml"}
    assert li.run_info("o/r", "tts08", 1) == (entry, "o/r")
    run["repository"]["full_name"] = "Owner/Repo"
    assert li.run_info("owner/repo", "tts08", 1) == (entry, "Owner/Repo")  # GitHub's spelling
    with pytest.raises(SystemExit, match="run 1 is in Owner/Repo, not o/r"):
        li.run_info("o/r", "tts08", 1)


def test_lock_inputs_writes_the_repository_as_github_spells_it(world, monkeypatch):
    _, _, li = stub.scripts()
    arts, runs = world.artifacts(), world.runs()
    monkeypatch.setitem(li.ARTIFACTS, "tts08", {a["name"]: "tts08" for a in arts if a["run"] == "tts08"})
    spelling = dict.fromkeys(runs, "Example/Kitten-Inference")
    monkeypatch.setattr(li, "run_info", lambda repo, key, run_id: (runs[key], spelling[key]))
    monkeypatch.setattr(li, "list_artifacts",
                        lambda repo, run_id: {a["name"]: a for a in arts if runs[a["run"]]["id"] == run_id})
    zips = {a["id"]: a["zip"] for a in arts}
    monkeypatch.setattr(li, "gh", lambda args, to=None: shutil.copyfile(zips[int(args[0].split("/")[-2])], to))
    out = world.root / "lock.json"
    argv = ["--repo", "example/kitten-inference", "--tts08-run", "1", "--cuda-run", "2", "--rocm-run", "3",
            "--inputs", str(world.inputs), "--out", str(out), "--source", str(world.source)]
    assert li.main(argv) == 0
    assert json.loads(out.read_text())["repository"] == "Example/Kitten-Inference"
    spelling["rocm"] = "example/kitten-inference"
    with pytest.raises(SystemExit, match="name their repository differently"):
        li.main(argv)


def test_the_real_lock_is_well_formed():
    aw, _, _ = stub.scripts()
    path = stub.SCRIPTS.parent / "release-inputs.json"
    if not path.is_file():
        pytest.skip("no .github/release-inputs.json")
    lock = aw.load_lock(path)
    parts = [r["part"] for r in lock["inputs"]]
    assert parts.count("tts08") == 54 and len(parts) == 57 and len(lock["engine_files"]) == 12
    assert not any("manylinux2014_x86_64" in r["file"] for r in lock["inputs"])
    assert {r["artifact"] for r in lock["inputs"] if r["part"] != "tts08"} == {
        "tts2-cu12-linux", "tts2-cu12-windows", "kitten-inference-tts2-rocm-manylinux_2_28_x86_64"}


def test_build_backends_may_normalize_dependency_names(world):
    # Hatchling spells underscores as hyphens; scikit-build may preserve them.
    aw, _, _ = stub.scripts()
    from types import SimpleNamespace
    public = SimpleNamespace(metadata=b'Name: kitten-inference\n', requires=stub.UMBRELLA_REQUIRES)
    normalized = [r.replace('espeakng_loader', 'espeakng-loader') for r in public.requires]
    aw.check_requirements(public, normalized, normalized)
