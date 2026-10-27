#!/usr/bin/env python3
"""Writes .github/release-inputs.json: the tested build outputs the kitten-inference wheels are assembled from.

    python .github/scripts/lock_inputs.py --repo OWNER/REPO --tts08-run ID --cuda-run ID --rocm-run ID \
        --inputs DIR --out .github/release-inputs.json [--source .] [--work DIR]
    python .github/scripts/lock_inputs.py --set-tested-trees COMMIT --out .github/release-inputs.json [--source .]

Run it once per build and GPU-test round, on the runs whose artifacts the GPU tests used: tts-0.8.yml (TTS 0.8),
tts-2-engine.yml (CUDA) and tts-2-rocm.yml (ROCm), in one repository. DIR holds those artifacts' wheels (gh run
download; one folder, or one per artifact). Through gh api it requires each run to have finished with success (and
records the attempt), downloads each artifact's zip (into --work, default a temporary folder), checks it against the
digest GitHub recorded at upload, and requires each wheel in DIR to be the one in that zip. Every wheel must match its
own RECORD and have one version. The lock then holds:
  repository    OWNER/REPO as GitHub spells it (a tag run must be in it; case does not matter)
  inputs        one row per wheel: file, part, sha256, size, run, artifact name, id and digest
  engine_files  the engine files of the CUDA and ROCm wheels (from their RECORD), which the GPU tests ran
  source_trees  per run, the git trees of the engine and TTS 0.8 sources at its commit (which must be in the local
                clone, --source): assemble_wheels.py build refuses a release commit whose sources differ
The linux-x86_64-manylinux2014 TTS 0.8 artifact is not used: every Linux x86-64 kitten-inference wheel is the
manylinux_2_28 build. tested_packaged_trees starts empty (null).

--set-tested-trees COMMIT: records tested_packaged_trees, the git trees of the packaged files at COMMIT, once that
commit's wheels passed the GPU tests and the test-PyPI run; a tag run assembles only that tree. Nothing else changes.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble_wheels as aw  # noqa: E402

WORKFLOWS = {"tts08": "tts-0.8.yml", "cuda": "tts-2-engine.yml", "rocm": "tts-2-rocm.yml"}
TTS08_FAMILIES = ("windows-arm64", "windows-x86_64", "macos-x86_64", "macos-arm64", "linux-x86_64-manylinux-2-28",
                  "linux-aarch64-manylinux2014", "linux-aarch64-manylinux-2-28", "android-arm64")
ARTIFACTS = {  # run -> artifact name -> part
    "tts08": {f"tts08-wheels-{family}": "tts08" for family in TTS08_FAMILIES},
    "cuda": {"tts2-cu12-linux": "cuda-linux", "tts2-cu12-windows": "cuda-windows"},
    "rocm": {"kitten-inference-tts2-rocm-manylinux_2_28_x86_64": "rocm"},
}
# The TTS 0.8 Python files a release takes from its own commit instead of the tested wheel (message text and the
# import guard); assemble_wheels.py refuses __init__.py.
OVERLAY = ["kitten_inference/tts08/_deps.py", "kitten_inference/tts08/get_model.py",
           "kitten_inference/tts08/cpp_model.py", "kitten_inference/tts08/_espeak_path.py"]
OLD_LINUX_X86 = ("manylinux2014_x86_64", "manylinux_2_17_x86_64")


def gh(args, to: Path = None) -> bytes:
    """gh api ARGS: its output, or written to `to`."""
    cmd = ["gh", "api", *args]
    try:
        if to is None:
            r = subprocess.run(cmd, capture_output=True)
        else:
            with open(to, "wb") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
    except OSError as exc:
        aw.refuse(f"cannot run gh ({exc})")
    if r.returncode:
        aw.refuse(f"gh api {args[-1]}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout


def run_info(repo: str, key: str, run_id: int) -> tuple:
    """The run's entry in the lock, and its repository as GitHub spells it."""
    run = json.loads(gh([f"repos/{repo}/actions/runs/{run_id}"]))
    workflow = str(run.get("path", "")).split("@", 1)[0].rsplit("/", 1)[-1]
    if workflow != WORKFLOWS[key]:
        aw.refuse(f"run {run_id} is a {workflow or '?'} run; the {key} input comes from {WORKFLOWS[key]}")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        aw.refuse(f"run {run_id} ({workflow}) is {run.get('status')}/{run.get('conclusion')}, not a success")
    full_name = str((run.get("repository") or {}).get("full_name", ""))
    if full_name.lower() != repo.lower():
        aw.refuse(f"run {run_id} is in {full_name or '?'}, not {repo}")
    return {"id": run_id, "attempt": run["run_attempt"], "commit": run["head_sha"], "workflow": workflow}, full_name


def list_artifacts(repo: str, run_id: int) -> dict:
    out = gh(["--paginate", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", "--jq", ".artifacts[]"])
    found = {}
    for line in out.decode("utf-8").splitlines():
        if line.strip():
            art = json.loads(line)
            if art["name"] in found:
                aw.refuse(f"run {run_id} has two artifacts named {art['name']}")
            found[art["name"]] = art
    return found


def make_lock(repo: str, runs: dict, artifacts: list, inputs: Path, src: aw.Source) -> dict:
    """The lock from the runs ({key: id, attempt, commit, workflow}), the downloaded artifacts ([{run, name, id,
    digest, zip}]) and the folder of wheels."""
    found = aw.find_wheels(inputs)
    rows, wheels = [], {}
    for art in artifacts:
        part = ARTIFACTS[art["run"]][art["name"]]
        label = f"artifact {art['name']} ({art['id']}) of run {runs[art['run']]['id']}"
        if art.get("digest"):
            if "sha256:" + aw.file_sha256(art["zip"]) != art["digest"]:
                aw.refuse(f"{label}: the downloaded zip does not match the digest GitHub recorded at upload")
        else:
            print(f"::warning::{label}: GitHub records no digest for it; the zip is not checked")
        with zipfile.ZipFile(art["zip"]) as z:
            members = [i for i in z.infolist() if i.filename.endswith(".whl")]
            if not members or (part != "tts08" and len(members) != 1):
                aw.refuse(f"{label}: holds {len(members)} wheels")
            for info in members:
                file = info.filename.rsplit("/", 1)[-1]
                digest, size = aw.member_digest(z, info)
                if file in wheels:
                    aw.refuse(f"{file} is in two artifacts")
                if file not in found:
                    aw.refuse(f"{file} ({label}) is not in {inputs}")
                if aw.file_sha256(found[file]) != digest.hex():
                    aw.refuse(f"{found[file]} is not the {file} of {label}")
                wheels[file] = (part, aw.Wheel(found[file]))
                rows.append({"file": file, "part": part, "sha256": digest.hex(), "size": size, "run": art["run"],
                             "artifact": art["name"], "artifact_id": art["id"], "artifact_digest": art.get("digest")})
    versions = sorted({w.version for _, w in wheels.values()})
    if len(versions) != 1:
        aw.refuse(f"the wheels have several versions: {', '.join(versions)}")
    for part, w in wheels.values():
        want = "kitten-inference-tts08" if part == "tts08" else aw.ENGINE_PARTS[part][0]
        if w.name != want:
            aw.refuse(f"{w.file}: Name is {w.name}, expected {want}")
        if part == "tts08" and any(t[2] in OLD_LINUX_X86 for t in w.tags):
            aw.refuse(f"{w.file}: manylinux2014 x86-64 wheels are not release inputs")
    counts = collections.Counter(p for p, _ in wheels.values())
    if not counts["tts08"] or any(counts[p] != 1 for p in aw.ENGINE_PARTS):
        aw.refuse(f"expected TTS 0.8 wheels and one wheel of each engine part, found {dict(counts)}")
    order = {p: i for i, p in enumerate(aw.PARTS)}
    rows.sort(key=lambda r: (order[r["part"]], r["file"]))
    engine_files = []
    for row in rows:
        part, w = wheels[row["file"]]
        if part == "tts08":
            continue
        for path in aw.select_payload(part, w)[0]:
            info = w.members[path]
            engine_files.append({"part": part, "path": path[len("kitten_inference/"):], "size": info.file_size,
                                 "mode": "%04o" % (info.external_attr >> 16 & 0o7777),
                                 "sha256": aw.hex_of(w.rows[path][1])})
    source_trees = {}
    for key, run in runs.items():
        if not src.has_commit(run["commit"]):
            aw.refuse(f"commit {run['commit']} of the {key} run is not in {src.root}: git fetch origin {run['commit']}")
        source_trees[key] = aw.source_trees(src, key, OVERLAY, rev=run["commit"])
    return {"schema": aw.SCHEMA, "version": versions[0], "repository": repo, "runs": runs, "inputs": rows,
            "engine_files": engine_files, "source_trees": source_trees, "tts08_python_overlay": list(OVERLAY),
            "tested_packaged_trees": None}


def dump(lock: dict) -> str:
    """The lock as JSON: one line per input row and engine file, and maps on one line when they are short."""
    def fmt(value, depth):
        pad, end = "  " * (depth + 1), "  " * depth
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            return "[\n" + ",\n".join(pad + json.dumps(v) for v in value) + "\n" + end + "]"
        if isinstance(value, dict) and value and (depth == 0 or len(json.dumps(value)) > 120):
            items = (f"{pad}{json.dumps(k)}: {fmt(v, depth + 1)}" for k, v in value.items())
            return "{\n" + ",\n".join(items) + "\n" + end + "}"
        return json.dumps(value)

    text = fmt(lock, 0) + "\n"
    if json.loads(text) != lock:
        raise AssertionError("the lock's JSON does not read back as the lock")
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", help="OWNER/REPO of the three runs")
    p.add_argument("--tts08-run", type=int)
    p.add_argument("--cuda-run", type=int)
    p.add_argument("--rocm-run", type=int)
    p.add_argument("--inputs", type=Path, help="folder with the runs' wheels")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("."), help="a clone that has the runs' commits")
    p.add_argument("--work", type=Path, help="folder for the downloaded artifact zips (default: a temporary one)")
    p.add_argument("--set-tested-trees", metavar="COMMIT", help="only record tested_packaged_trees from COMMIT")
    a = p.parse_args(argv)
    src = aw.Source(a.source)
    if a.set_tested_trees:
        lock = aw.load_lock(a.out)
        if not src.has_commit(a.set_tested_trees):
            aw.refuse(f"{a.set_tested_trees} is not a commit in {a.source}")
        lock["tested_packaged_trees"] = aw.tested_trees(src, rev=a.set_tested_trees)
        a.out.write_text(dump(lock), encoding="utf-8")
        print(f"{a.out}: tested_packaged_trees from {a.set_tested_trees}")
        return 0
    if not (a.repo and a.tts08_run and a.cuda_run and a.rocm_run and a.inputs):
        p.error("--repo, --tts08-run, --cuda-run, --rocm-run and --inputs are needed")
    ids = {"tts08": a.tts08_run, "cuda": a.cuda_run, "rocm": a.rocm_run}
    found = {key: run_info(a.repo, key, run_id) for key, run_id in ids.items()}
    runs = {key: run for key, (run, _) in found.items()}
    # The lock records GitHub's spelling of OWNER/REPO, whatever case --repo was typed in.
    names = sorted({name for _, name in found.values()})
    if len(names) != 1:
        aw.refuse(f"the runs name their repository differently: {', '.join(names)}")
    repo = names[0]
    if a.work:
        a.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.work) as tmp:
        artifacts = []
        for key, run_id in ids.items():
            listing = list_artifacts(repo, run_id)
            for name in ARTIFACTS[key]:
                art = listing.get(name)
                if art is None or art.get("expired"):
                    aw.refuse(f"run {run_id} has no {'unexpired ' if art else ''}artifact {name}")
                zip_path = Path(tmp) / f"{art['id']}.zip"
                gh([f"repos/{repo}/actions/artifacts/{art['id']}/zip"], to=zip_path)
                artifacts.append({"run": key, "name": name, "id": art["id"], "digest": art.get("digest"),
                                  "zip": zip_path})
        lock = make_lock(repo, runs, artifacts, a.inputs, src)
    a.out.write_text(dump(lock), encoding="utf-8")
    parts = collections.Counter(r["part"] for r in lock["inputs"])
    print(f"{a.out}: kitten-inference {lock['version']} inputs from {repo}: {parts['tts08']} TTS 0.8 wheels, "
          f"{len(lock['inputs']) - parts['tts08']} engine wheels, {len(lock['engine_files'])} engine files")
    for key, run in runs.items():
        print(f"  {key}: run {run['id']} attempt {run['attempt']} ({run['workflow']}) at {run['commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
