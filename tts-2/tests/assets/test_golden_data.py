"""The recorded test data in tests/data is well formed and agrees with the converter's projection set."""

from __future__ import annotations

import json
import re

from conftest import DATA_DIR

from kt2n_assets.convert import PROJECTION_RE

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SUBS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def test_projection_regex_selects_exactly_the_seven_projections():
    for layer in (0, 7, 27):
        for sub in SUBS:
            assert PROJECTION_RE.match(f"model.layers.{layer}.{sub}.weight")
    not_projections = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "lm_head.weight",
        "model.norm.weight",
        "spk_proj.0.weight",
        "model.embed_tokens.weight",
    ]
    for name in not_projections:
        assert not PROJECTION_RE.match(name)


def test_audit_names_are_the_projection_set():
    audit = json.loads((DATA_DIR / "ternary_audit_names.json").read_text(encoding="utf-8"))
    assert "source" not in audit
    names = audit["exactly_ternary_names"]
    assert len(names) == len(set(names)) == 196 == audit["exactly_ternary"]
    assert all(PROJECTION_RE.match(n) for n in names)
    layers = {int(PROJECTION_RE.match(n).group(1)) for n in names}
    assert layers == set(range(28))
    non_ternary = audit["not_exactly_ternary_names"]
    assert len(non_ternary) == 2
    assert non_ternary[1] == "spk_proj.0.weight"
    assert non_ternary[0].startswith("model.embed_")
    assert audit["checkpoint_sha256"] == "cb2abc13a252b0974da5af007b51f7c1dbc90b607887222868ff783c707ca6f8"


def test_pack_golden_file_is_well_formed(golden):
    pack, unpack = golden.pack_entries, golden.unpack_entries
    assert len(pack) == 45 and len(unpack) == 3
    errors = [e for e in pack if "error" in e]
    assert len(errors) == 5 and all(e["error"]["type"] == "ValueError" for e in errors)
    synthetic = [e for e in pack if any(c.startswith("synthetic_checkpoint:") for c in e["cases"])]
    assert len(synthetic) == 14
    for e in pack:
        assert HEX64.match(e["input_sha256"]) and len(e["input_shape"]) == 2 and e["cases"]
        if "error" not in e:
            n, k = e["input_shape"]
            assert e["packed"] == {"dtype": "int32", "shape": [n, k // 16], "sha256": e["packed"]["sha256"]}
            assert e["scales"] == {"dtype": "bfloat16", "shape": [n, k // 128], "sha256": e["scales"]["sha256"]}
            assert HEX64.match(e["packed"]["sha256"]) and HEX64.match(e["scales"]["sha256"])
    for e in unpack:
        assert HEX64.match(e["packed_sha256"]) and HEX64.match(e["scales_sha256"])
        assert e["output"]["shape"] == [e["packed_shape"][0], e["k"]] and HEX64.match(e["output"]["sha256"])
    keys = [(e["input_sha256"], tuple(e["input_shape"])) for e in pack]
    assert len(keys) == len(set(keys))
