#!/usr/bin/env python3
"""
build_manifest.py -- derive the Qwen3.8-Flash-Next atlas manifest from REAL pack bytes.

Reads (read-only):
  <pack>/config.json.native              -> true architecture geometry
  <pack>/pack_scan.json                  -> per-family quant bit schedules + ngram table facts
  <pack>/model-*.safetensors (+ ngram, mtp patch) -> safetensors JSON headers: every
                                             tensor name, dtype, shape, and its REAL
                                             on-disk byte span.
Writes:
  web/atlas_data.js       -- `const ATLAS = {...};` consumed by web/index.html
  live/manifest.json      -- same payload, for the text monitor + the live aggregator

Nothing is modeled or guessed: every cube's volume is the tensor's actual byte
count from `data_offsets`, so the atlas sums to the pack size on disk.

This script never writes into the model pack. Verified by the digest guard below.
"""

from __future__ import annotations

import hashlib
import time
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------

# Model pack. Overridable so the project stays portable.
PACK = os.environ.get(
    "QWEN_PACK", "/home/cpatriaw/models/Qwen3.8-Flash-Next-EXL3"
)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB_JS = os.path.join(ROOT, "web", "atlas_data.js")
LIVE_JSON = os.path.join(ROOT, "live", "manifest.json")

# safetensors dtype -> bytes per element
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "F8_E4M3FN": 1, "F8_E5M2FN": 1,
}

# MTP tensors carry their own layer indices; fold them into a high band so
# they never alias a real decoder layer.
MTP_BAND_OFFSET = 1000

# Collapse per-layer/per-expert indices into pack_scan's family keys, e.g.
#   model.language_model.layers.7.mlp.experts.13.down_proj -> model.language_model.layers.N.mlp.experts.*.down_proj
RE_INDEX = re.compile(r"\.(layers|blocks)\.\d+\.")
RE_EXPERT_IDX = re.compile(r"\.experts\.\d+\.")

# Name -> human category. Order matters: first hit wins.
CATEGORY_RULES = (
    ("ngram_embedding", "ple", "PLE ngram table"),
    ("ple.ple_embedding", "ple", "PLE embedding"),
    ("ple.", "ple", "PLE"),
    ("mlp.experts.", "expert", "Routed expert"),
    ("mlp.shared_expert.", "shared", "Shared expert"),
    ("mlp.gate.", "gate", "MoE gate"),
    ("mlp.hc_gate", "gate", "HC gate"),
    ("self_attn.indexer", "index", "QSA indexer"),
    ("self_attn.", "attn", "Full attention"),
    ("linear_attn", "mamba", "Linear attention"),
    ("hyper_connection", "hyper", "Hyper connection"),
    ("attn_hyper_connection", "hyper", "Hyper connection"),
    ("patch_embed", "vision", "ViT patch embed"),
    ("merger", "vision", "ViT merger"),
    ("visual.blocks", "vision", "ViT block"),
    ("visual.patch_norm", "vision", "ViT norm"),
    ("embed_tokens", "embed", "Token embedding"),
    ("lm_head", "head", "Output head"),
    ("mtp.", "mtp", "MTP drafter"),
    ("norm", "norm", "Norm"),
)

CAT_LABEL = {
    "ple": "PLE ngram memory", "expert": "Routed experts", "shared": "Shared expert",
    "gate": "MoE gate", "index": "QSA indexer", "attn": "Full attention",
    "mamba": "Linear attention", "hyper": "Hyper-connection", "vision": "Vision tower",
    "embed": "Embedding", "head": "Output head", "mtp": "MTP drafter",
    "norm": "Norms", "other": "Other",
}


# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def pack_digest(pack: str) -> str:
    """Stable fingerprint of the pack's file list -- guards against accidental writes."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(pack)):
        p = os.path.join(pack, name)
        if os.path.isfile(p):
            st = os.stat(p)
            h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def read_st_header(path: str) -> dict:
    """Parse a safetensors JSON header. Raises on a malformed header."""
    with open(path, "rb") as f:
        blob = f.read(8)
        if len(blob) != 8:
            raise ValueError(f"{path}: shorter than an 8-byte header")
        n = struct.unpack("<Q", blob)[0]
        if n == 0:  # some writers double-length-prefix
            blob = f.read(8)
            n = struct.unpack("<Q", blob)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {n}")
        hdr = f.read(n)
        if len(hdr) != n:
            raise ValueError(f"{path}: truncated header")
    data = json.loads(hdr.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: header is not an object")
    return data


def tensor_bytes(meta: dict) -> int:
    """Real byte footprint of one tensor, straight off data_offsets."""
    dt = meta.get("dtype")
    if dt is None:
        return 0
    if dt not in DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {dt!r}")
    shape = meta.get("shape") or []
    numel = 1
    for d in shape:
        numel *= int(d)
    offs = meta.get("data_offsets")
    if isinstance(offs, (list, tuple)) and len(offs) == 2:
        span = int(offs[1]) - int(offs[0])
        if span >= 0:
            # Trust the header span; sanity-check against dtype math.
            expect = numel * DTYPE_BYTES[dt]
            if expect != span:
                # EXL3 packs codebook/aux into the same span; keep the real span,
                # but remember the divergence so audits can surface it.
                meta["_dtype_mismatch"] = expect
            return span
    return numel * DTYPE_BYTES[dt]


def family_of(name: str) -> str:
    """Collapse layer/expert indices so pack_scan's per-family schedules can be looked up."""
    name = RE_INDEX.sub(lambda m: f".{m.group(1)}.N.", name)
    name = RE_EXPERT_IDX.sub(".experts.*.", name)
    return name


def categorize(name: str) -> tuple[str, str]:
    for pat, cat, label in CATEGORY_RULES:
        if pat in name:
            return cat, label
    return "other", "Other"


def layer_of(name: str) -> int | None:
    m = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return None


def collect(pack: str) -> list[dict]:
    """Every tensor in the pack, with real bytes, as flat records."""
    files = [f for f in sorted(os.listdir(pack)) if f.endswith(".safetensors")]
    if not files:
        raise SystemExit(f"no .safetensors files under {pack}")
    out: list[dict] = []
    for fname in files:
        hdr = read_st_header(os.path.join(pack, fname))
        hdr.pop("", None)
        hdr.pop("__metadata__", None)
        for tname, meta in hdr.items():
            if not isinstance(meta, dict):
                continue
            b = tensor_bytes(meta)
            if b == 0:
                continue
            cat, _ = categorize(tname)
            r = {
                "n": tname,
                "f": fname,
                "d": meta.get("dtype"),
                "s": meta.get("shape") or [],
                "b": b,
                "c": cat,
                "l": layer_of(tname),
            }
            if meta.get("_dtype_mismatch"):
                r["dm"] = 1
            out.append(r)
    log(f"  read {len(out):,} tensors from {len(files)} shard files")
    return out


# ---------------------------------------------------------------------------

def group_records(tensors: list[dict]) -> list[dict]:
    """
    Collapse identical tensor *templates* into one cube with a `count`.

    24,576 experts x 3 projections x 3 suffixes would be 304k individual cubes --
    unrenderable and pointless: every copy of a template has the same bytes.
    Group key = name with all layer/expert indices replaced + dtype + shape.
    """
    buckets: dict[tuple, dict] = {}
    for t in tensors:
        tpl = re.sub(r"(layers|blocks|experts)\.\d+\.", r"\1.*.", t["n"])
        key = (tpl, t["d"], tuple(t["s"]), t["c"], t["f"])
        g = buckets.get(key)
        if g is None:
            buckets[key] = g = {
                "name": tpl,
                "dtype": t["d"],
                "shape": t["s"],
                "cat": t["c"],
                "shard": t["f"],
                "count": 0,
                "bytes_each": t["b"],
                "bytes": 0,
            }
        elif g["bytes_each"] != t["b"]:
            raise ValueError(
                f"template {tpl!r} has tensors of differing size "
                f"({g['bytes_each']} vs {t['b']}) -- grouping would lie"
            )
        g["count"] += 1
        g["bytes"] += t["b"]
    return sorted(buckets.values(), key=lambda g: -g["bytes"])


def per_layer_matrix(tensors: list[dict]) -> dict[int, dict[str, int]]:
    """layer -> category -> real bytes. Drives the 48-tower height map."""
    m: dict[int, Counter] = defaultdict(Counter)
    for t in tensors:
        if t["l"] is None:
            continue

        # PLE ngram memory and the ViT tower are global, not decoder layers
        if t["c"] in ("ple", "vision"):
            continue
        # fold mtp-* layers into their own band so they do not alias layer N
        band = t["l"] if not t["n"].startswith("mtp.") else t["l"] + MTP_BAND_OFFSET
        m[band][t["c"]] += t["b"]
    return {k: dict(v) for k, v in sorted(m.items())}


def main() -> int:
    if not os.path.isdir(PACK):
        log(f"FAIL: pack not found: {PACK}")
        return 2
    before = pack_digest(PACK)
    log(f"pack: {PACK}\n  digest {before}")

    with open(os.path.join(PACK, "config.json.native")) as f:
        cfg = json.load(f)
    with open(os.path.join(PACK, "pack_scan.json")) as f:
        scan = json.load(f)
    text_cfg = cfg.get("text_config", {})
    vis_cfg = cfg.get("vision_config", {})

    tensors = collect(PACK)
    groups = group_records(tensors)
    matrix = per_layer_matrix(tensors)

    total = sum(t["b"] for t in tensors)
    on_disk = sum(
        os.path.getsize(os.path.join(PACK, f))
        for f in os.listdir(PACK) if f.endswith(".safetensors")
    )

    by_cat: Counter = Counter()
    cat_bytes: Counter = Counter()
    for t in tensors:
        by_cat[t["c"]] += 1
        cat_bytes[t["c"]] += t["b"]

    bits_by_family = {}
    for fam, d in (scan.get("dense_k_by_family") or {}).items():
        bits_by_family[fam] = sorted(d.keys())
    for fam, lst in (scan.get("expert_k_per_layer") or {}).items():
        bits_by_family[f"layers.%s.experts.*" % fam] = sorted(
            k for v in lst.values() for k in (v if isinstance(v, list) else [])
        )

    ng = (scan.get("ngram_tables") or {})
    ng_facts = ng.get(next(iter(ng))) if ng else {}
    atlas = {
        "meta": {
            "model": os.path.basename(PACK.rstrip("/")),
            "pack": PACK,
            "digest": before,
            "dtype_divergences": sum(1 for t in tensors if t.get("dm")),
            "built_at": int(time.time()),
            "generator": "tools/build_manifest.py",
            "note": ("all byte counts are real on-disk sizes read from safetensors "
                     "headers (data_offsets), not modeled estimates"),
        },
        "arch": {
            "layers": text_cfg.get("num_hidden_layers"),
            "layer_types": text_cfg.get("layer_types"),
            "hidden_size": text_cfg.get("hidden_size"),
            "num_experts": text_cfg.get("num_experts"),
            "experts_per_token": text_cfg.get("num_experts_per_tok"),
            "moe_intermediate": text_cfg.get("moe_intermediate_size"),
            "shared_intermediate": text_cfg.get("shared_expert_intermediate_size"),
            "attn_heads": text_cfg.get("num_attention_heads"),
            "kv_heads": text_cfg.get("num_key_value_heads"),
            "head_dim": text_cfg.get("head_dim"),
            "indexer": {
                "n_heads": text_cfg.get("indexer_n_heads"),
                "kv_heads": text_cfg.get("indexer_kv_heads"),
                "head_dim": text_cfg.get("indexer_head_dim"),
            },
            "linear_attn": {
                "key_heads": text_cfg.get("linear_num_key_heads"),
                "value_heads": text_cfg.get("linear_num_value_heads"),
                "key_head_dim": text_cfg.get("linear_key_head_dim"),
                "value_head_dim": text_cfg.get("linear_value_head_dim"),
                "conv_kernel": text_cfg.get("linear_conv_kernel_dim"),
            },
            "vocab": text_cfg.get("vocab_size"),
            "ple_layer_ids": text_cfg.get("ple_layer_ids"),
            "ngram": {
                "size": text_cfg.get("ngram_size"),
                "heads": text_cfg.get("heads_per_ngram"),
                "heads_pack": ng_facts.get("words"),
                "rows": ng_facts.get("total_rows"),
                "bits": ng_facts.get("bits"),
                "shards": ng_facts.get("num_shards"),
                "rows_per_shard": ng_facts.get("rows_per_shard"),
            },
            "mtp": {
                "num_hidden_layers": text_cfg.get("mtp_num_hidden_layers"),
                "layer_types": (text_cfg.get("mtp") or {}).get("layer_types"),
            },
            "rope_theta": (text_cfg.get("rope_parameters") or {}).get("rope_theta"),
            "partial_rotary": (text_cfg.get("rope_parameters") or {}).get("partial_rotary_factor"),
            "vision": {
                "hidden": vis_cfg.get("hidden_size"),
                "intermediate": vis_cfg.get("intermediate_size"),
                "heads": vis_cfg.get("num_heads"),
                "out_hidden": vis_cfg.get("out_hidden_size"),
                "patch": vis_cfg.get("patch_size"),
                "pos_emb": vis_cfg.get("num_position_embeddings"),
            },
            "quant": {
                "method": (cfg.get("quantization_config") or {}).get("quant_method"),
                "bits": (cfg.get("quantization_config") or {}).get("bits"),
                "head_bits": (cfg.get("quantization_config") or {}).get("head_bits"),
                "vision_bits": (cfg.get("quantization_config") or {}).get("vision_bits"),
                "mtp_bits": (cfg.get("quantization_config") or {}).get("mtp_bits"),
            },
        },
        "totals": {
            "tensors": len(tensors),
            "groups": len(groups),
            "bytes_total": total,
            "gib_total": round(total / 2**30, 4),
            "gib_on_disk": round(on_disk / 2**30, 4),
            "by_category": {c: {"tensors": by_cat[c], "bytes": cat_bytes[c],
                                "gib": round(cat_bytes[c] / 2**30, 4),
                                "label": CAT_LABEL.get(c, c)}
                            for c in sorted(cat_bytes)},
        },
        "quant_bits_by_family": bits_by_family,
        "layers": [
            {"i": i, "type": (lambda lt: lt[i] if i < len(lt) else None)(text_cfg.get("layer_types") or []) if i < 48 else None,
             "bytes": sum(v.values()),
             "gib": round(sum(v.values()) / 2**30, 5),
             "by_cat": v}
            for i, v in matrix.items()
        ],
        "weights": groups,
    }

    payload = json.dumps(atlas, separators=(",", ":"), sort_keys=False)
    os.makedirs(os.path.dirname(WEB_JS), exist_ok=True)
    def _sig(t):
        return re.sub(r'"built_at":\s*-?\d+', '"built_at":0', t)
    for path, text in (
        (WEB_JS, "/* generated by tools/build_manifest.py -- do not edit by hand */\nconst ATLAS = %s;\n" % payload),
        (LIVE_JSON, payload),
    ):
        try:
            old = open(path).read()
        except OSError:
            old = None
        if old is not None and _sig(old) == _sig(text):
            log(f"  kept  {path} (payload unchanged)")
            continue
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        log(f"  wrote {path} ({len(text):,} bytes)")

    after = pack_digest(PACK)
    if after != before:
        log(f"FAIL: pack digest changed during run ({before} -> {after})")
        return 3

    log("")
    log(f"  tensors      {len(tensors):>10,}")
    log(f"  cubes        {len(groups):>10,}")
    log(f"  atlas bytes  {total/2**30:>10.3f} GiB")
    log(f"  on-disk      {on_disk/2**30:>10.3f} GiB")
    log(f"  delta        {(on_disk-total)/2**30:>10.3f} GiB (headers/shard padding)")
    for c in sorted(cat_bytes):
        log(f"    {CAT_LABEL.get(c,c):<20} {by_cat[c]:>8,} tensors  {cat_bytes[c]/2**30:8.3f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
