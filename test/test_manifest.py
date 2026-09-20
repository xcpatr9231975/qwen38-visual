#!/usr/bin/env python3
"""Tests for the atlas pipeline: builder semantics + manifest invariants + monitor format."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "live"))

import build_manifest as bm
import monitor as mo

PACK = os.environ.get("QWEN_PACK", "/home/cpatriaw/models/Qwen3.8-Flash-Next-EXL3")
needs_pack = pytest.mark.skipif(not os.path.isdir(PACK), reason="model pack not present")


# ------------------------------------------------------------------ units

def test_categorize_order_matters():
    assert bm.categorize("model...mlp.experts.3.down_proj")[0] == "expert"
    assert bm.categorize("model...mlp.gate.weight")[0] == "gate"
    assert bm.categorize("ngram_embedding.tables.rows")[0] == "ple"
    assert bm.categorize("mtp.mtp_patch.w")[0] == "mtp"
    assert bm.categorize("totally.unknown")[0] == "other"


def test_layer_of():
    assert bm.layer_of("model.language_model.layers.7.mlp.x") == 7
    assert bm.layer_of("visual.blocks.2.norm.w") == 2
    assert bm.layer_of("embed_tokens.w") is None


def test_family_of_collapses_indices():
    f = bm.family_of("model.language_model.layers.7.mlp.experts.13.down_proj")
    assert ".N." in f and ".experts.*." in f
    assert bm.family_of("lm_head.w") == "lm_head.w"


def test_tensor_bytes_offsets_and_dtype():
    meta = {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]}
    assert bm.tensor_bytes(meta) == 32
    meta = {"dtype": "F16", "shape": [4, 4], "data_offsets": [100, 420]}
    assert bm.tensor_bytes(meta) == 320           # real span wins over dtype math
    assert meta["_dtype_mismatch"] == 32           # divergence remembered for audits
    assert bm.tensor_bytes({"dtype": None, "shape": [2]}) == 0
    with pytest.raises(ValueError):
        bm.tensor_bytes({"dtype": "NOPE", "shape": [2]})


def test_group_records_invariants():
    ts = [
        {"n": "layers.0.experts.1.w", "d": "F16", "s": [2, 2], "c": "expert", "f": "a.st", "b": 8},
        {"n": "layers.0.experts.2.w", "d": "F16", "s": [2, 2], "c": "expert", "f": "a.st", "b": 8},
        {"n": "layers.1.experts.1.w", "d": "F16", "s": [2, 2], "c": "expert", "f": "a.st", "b": 8},
        {"n": "lm_head.w", "d": "F32", "s": [3, 3], "c": "head", "f": "b.st", "b": 36},
    ]
    gs = bm.group_records(ts)
    assert sum(g["count"] for g in gs) == len(ts)
    assert sum(g["bytes"] for g in gs) == sum(t["b"] for t in ts)
    assert [g["count"] for g in gs] == [1, 3]          # head(36B) before expert(24B), sorted by bytes desc
    with pytest.raises(ValueError):
        bm.group_records(ts + [{"n": "layers.5.experts.9.w", "d": "F16", "s": [2, 2],
                                "c": "expert", "f": "a.st", "b": 99}])


def test_per_layer_matrix_band_and_exclusions():
    ts = [
        {"n": "layers.2.self_attn.q", "c": "attn", "l": 2, "b": 100},
        {"n": "mtp.layers.0.x", "c": "mtp", "l": 0, "b": 7},          # band offset
        {"n": "ngram_embedding.rows", "c": "ple", "l": None, "b": 999},
        {"n": "visual.blocks.1.attn", "c": "vision", "l": 1, "b": 50},  # excluded
    ]
    m = bm.per_layer_matrix(ts)
    assert m[2] == {"attn": 100}
    assert m[bm.MTP_BAND_OFFSET] == {"mtp": 7}
    assert 1 not in m and None not in m


def test_read_st_header_roundtrip():
    obj = {"w": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}
    blob = json.dumps(obj).encode()
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        with open(f.name, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
        hdr = bm.read_st_header(f.name)
    os.unlink(f.name)
    assert hdr["w"]["dtype"] == "F16"
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        with open(f.name, "wb") as fh:
            fh.write(struct.pack("<Q", 12))
            fh.write(b"{not json")
        with pytest.raises(ValueError):
            bm.read_st_header(f.name)
    os.unlink(f.name)


# ------------------------------------------------------------------ monitor format

def test_bar_shape():
    b = mo.bar(50, 100, 10)
    assert len(b) == 10
    assert b.count("@") == 5
    assert mo.bar(0, 100, 10).count("@") == 0


@pytest.fixture()
def fake_manifest(tmp_path):
    d = {
        "meta": {"model": "fake", "pack": "/fake", "digest": "0123456789abcd"[:4] + "ef" * 4,
                 "note": "test"},
        "arch": {"layers": 3, "layer_types": ["linear_attention", "linear_attention", "full_attention"],
                 "num_experts": 8, "experts_per_token": 2, "quant": {"method": "exl3", "bits": 4}},
        "totals": {"tensors": 10, "groups": 4, "bytes_total": 3 * 2**30,
                   "gib_total": 3.0, "gib_on_disk": 3.1,
                   "by_category": {"expert": {"tensors": 8, "bytes": 2**30, "gib": 1.0, "label": "Routed experts"},
                                   "ple": {"tensors": 2, "bytes": 2**31, "gib": 2.0, "label": "PLE ngram memory"}}},
        "quant_bits_by_family": {"layers.0.experts.*": [3, 4]},
        "layers": [
            {"i": 0, "type": "linear_attention", "bytes": 2**29, "gib": 0.5, "by_cat": {"expert": 2**29}},
            {"i": 1, "type": "linear_attention", "bytes": 2**30, "gib": 1.0, "by_cat": {"expert": 2**30}},
            {"i": 2, "type": "full_attention", "bytes": 2**30, "gib": 1.0, "by_cat": {"attn": 2**30}},
            {"i": 1000, "type": None, "bytes": 2**29, "gib": 0.5, "by_cat": {"mtp": 2**29}},
        ],
        "weights": [{"name": "layers.*.experts.*.w", "dtype": "F16", "shape": [4, 4],
                     "cat": "expert", "shard": "m.st", "count": 8, "bytes_each": 2**29 // 8, "bytes": 2**29}],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(d))
    return str(p)


def test_monitor_model_and_views(fake_manifest, monkeypatch):
    m = mo.Model(fake_manifest)
    assert m.data["totals"]["gib_total"] == 3.0
    assert not m.reload()                                   # unchanged -> no reload
    os.utime(fake_manifest, (0, time.time() + 5))
    assert m.changed() and m.reload()
    head = mo.header_lines(m)
    assert any("qwen3.8" in x.lower() for x in head)
    rows = mo.category_lines(m.data, "bytes", 80)
    assert rows[0][2] == mo.CAT_COLOR["ple"]                # ple biggest -> first
    strip = mo.layer_strip(m.data, 80)
    assert "full@2" in strip and "48+MTP" in strip          # mtp band rendered
    q = mo.quant_lines(m.data, 5)
    assert any("experts" in r for r in q)
    import io
    buf = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', buf)
    mo.plain_snapshot(m)                                    # must not raise
    out = buf.getvalue()
    assert "Routed experts" in out and "@" in out


# ------------------------------------------------------------------ integration vs real pack

@needs_pack
def test_collect_totals_and_categories():
    ts = bm.collect(PACK)
    assert ts and all(t["b"] > 0 for t in ts)
    known = set(bm.CAT_LABEL) | {"other"}
    assert all(t["c"] in known for t in ts)
    by_cat = {}
    for t in ts:
        by_cat[t["c"]] = by_cat.get(t["c"], 0) + t["b"]
    assert sum(by_cat.values()) == sum(t["b"] for t in ts)


@needs_pack
def test_end_to_end_manifest(tmp_path):
    web = tmp_path / "web"
    live = tmp_path / "live"
    web.mkdir()
    live.mkdir()
    bm.WEB_JS = str(web / "atlas_data.js")
    bm.LIVE_JSON = str(live / "manifest.json")
    assert bm.main() == 0
    d = json.load(open(bm.LIVE_JSON))
    t = d["totals"]
    assert t["bytes_total"] > 0
    # cubes account for every byte, once
    assert sum(g["bytes"] for g in d["weights"]) == t["bytes_total"]
    assert sum(g["count"] for g in d["weights"]) == t["tensors"]
    # categories agree with totals
    assert sum(e["bytes"] for e in t["by_category"].values()) == t["bytes_total"]
    # on-disk >= atlas, delta is only headers/padding
    assert 0 <= t["gib_on_disk"] - t["gib_total"] < 0.5
    # decoder towers + mtp band
    ids = [l["i"] for l in d["layers"]]
    assert max(i for i in ids if i < 999) == 47
    assert any(i >= 1000 for i in ids)
    for l in d["layers"]:
        assert sum(l["by_cat"].values()) == l["bytes"]
        assert not any(k in l["by_cat"] for k in ("ple", "vision"))
    # no per-layer bytes exceed the whole model
    assert max(l["bytes"] for l in d["layers"]) < t["bytes_total"] // 8
    # web payload parses as the same object
    js = open(bm.WEB_JS).read()
    assert js.find("const ATLAS = ") >= 0
    payload = js.split("const ATLAS = ", 1)[1].rstrip(" ;\n")
    assert json.loads(payload) == d
