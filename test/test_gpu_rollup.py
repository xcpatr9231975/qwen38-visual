"""gpu rollup: bucketing, busy math, blind buckets, pruning, persistence."""
import importlib.util
import json
import pathlib

_SRC = str(pathlib.Path(__file__).resolve().parent.parent / "live" / "telemetry_collector.py")
_SPEC = importlib.util.spec_from_file_location("atlas_telemetry_collector", _SRC)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
assert _MOD is not None
_SPEC.loader.exec_module(_MOD)
GpuRollup = _MOD.GpuRollup


def make(tmp_path, bucket_s=10.0, window_s=100.0, busy=5.0, period=0.5):
    import os.path as _p
    return GpuRollup(_p.join(str(tmp_path), "gpu_stats.json"), period, busy, bucket_s, window_s)


def test_no_evidence_is_blind(tmp_path):
    r = make(tmp_path)
    r.add(1000.0, None, False)
    assert r.buckets == {}


def test_busy_ratio_and_peak(tmp_path):
    r = make(tmp_path, bucket_s=1000.0, window_s=10000.0)
    for i in range(100):
        r.add(1000.0 + i * 0.5, 50.0, False)
    for i in range(100):
        r.add(1000.0 + (100 + i) * 0.5, 0.0, False)
    b = next(iter(r.buckets.values()))
    assert abs(b[1] - 100.0) < 1e-6
    assert abs(b[0] - 50.0) < 1e-6          # 100 busy samples x 0.5s
    assert b[2] == 50.0


def test_engine_activity_counts_even_without_util(tmp_path):
    r = make(tmp_path)
    r.add(2000.0, None, True)
    b = next(iter(r.buckets.values()))
    assert b[0] == 0.5 and b[1] == 0.5


def test_below_threshold_is_idle(tmp_path):
    r = make(tmp_path)
    r.add(3000.0, 4.9, False)
    b = next(iter(r.buckets.values()))
    assert b[0] == 0.0 and b[1] == 0.5


def test_persist_and_reload_roundtrip(tmp_path):
    import time as _t
    t0 = _t.time() - 10.0               # inside the 100s window, real clock
    r = make(tmp_path)
    for i in range(20):
        r.add(t0 + i * 0.5, 90.0, True)
    r.flush(t0)
    doc = json.load(open(f"{tmp_path}/gpu_stats.json"))
    assert doc["meta"]["bucket_s"] == 10.0
    r2 = make(tmp_path)
    assert r2.buckets == {k: v for k, v in
                          [(int(k), [float(x) for x in v]) for k, v in doc["buckets"].items()]}


def test_prune_drops_out_of_window(tmp_path):
    r = make(tmp_path)                        # window 100s, bucket 10s
    r.buckets[0] = [3.0, 5.0, 80.0]            # ancient bucket
    r.add(10000.0, 80.0, True)                # triggers prune on rollover
    assert 0 not in r.buckets
    assert all(k >= 980 for k in r.buckets)   # (10000-110)//10 = 989
