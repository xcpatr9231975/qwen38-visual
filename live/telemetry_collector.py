#!/usr/bin/env python3
"""
telemetry_collector.py -- tail the vllama server metrics and publish a live
telemetry feed for the atlas views (web overlay + monitor.py inference page).

Polls <server>/metrics (Prometheus text), windows the engine counters, and
writes live/telemetry.json atomically every tick:

  phase            idle | prefill | decode | prefill+decode   (per engine tick)
  prefill_tps      prompt tokens promoted /s   (windowed)
  decode_tps       generation tokens /s         (windowed)
  run / wait       running / waiting requests  (gauges)
  kv_perc          KV-cache utilisation 0..1
  ttft_ms itl_ms   windowed means from the *_sum/_count histogram pairs
  mtp_accept       spec-decode draft acceptance (accepted/draft, k=3 runs)
  preempt          preemption counter delta

Usage:
  python3 live/telemetry_collector.py                 # foreground
  python3 live/telemetry_collector.py --url http://host:port/metrics
  python3 live/telemetry_collector.py --once          # single sample to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import math
import subprocess
import urllib.request
from urllib.parse import urlparse
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_URL = os.environ.get("VLLAMA_URL", "http://127.0.0.1:8009/metrics")
OUT = os.path.join(ROOT, "live", "telemetry.json")

COUNTERS = (
    "prompt_tokens_total", "generation_tokens_total",
    "num_preemptions_total",
    "prefix_cache_hits_total", "prefix_cache_queries_total",
    "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
    "request_success_total", "http_requests_total",
    "http_request_size_bytes_sum", "http_response_size_bytes_sum",
)
GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")

SMI_QUERY = ("index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
             "power.draw,power.limit,temperature.gpu,clocks.sm,utilization.encoder,"
             "utilization.decoder,pstate")


def _num(tok):
    """'92 %' -> 92.0 ; '[N/A]' / 'N/A' -> None."""
    tok = tok.strip().strip("[]")
    if not tok or tok.startswith("N/A"):
        return None
    keep = ""
    for ch in tok:
        if ch in "+-.0123456789":
            keep += ch
        elif keep:
            break
    try:
        v = float(keep)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def gpus_sample():
    """nvtop-style device rows via nvidia-smi; None when unavailable."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + SMI_QUERY,
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.strip().split("\n"):
        f = [c.strip() for c in line.split(",")]
        if len(f) < 13:
            continue
        out.append({"i": int(_num(f[0]) or 0), "name": f[1].replace("NVIDIA ", ""),
                    "util": _num(f[2]), "mem": _num(f[3]),
                    "mu": _num(f[4]), "mt": _num(f[5]),
                    "pw": _num(f[6]), "cap": _num(f[7]),
                    "t": _num(f[8]), "clk": _num(f[9]), "mclk": None,
                    "enc": _num(f[10]), "dec": _num(f[11]),
                    "perf": f[12] if f[12] != "N/A" else None})
    return out or None

HIST = ("time_to_first_token_seconds", "inter_token_latency_seconds")
PREFIX = "vllm:"


def parse(text: str) -> dict:
    """base -> float for every counter/gauge/hist-sum/count we care about."""
    out: dict[str, float] = {}
    raw = set(COUNTERS) | set(GAUGES)
    for h in HIST:
        raw |= {h + "_sum", h + "_count"}
    for line in text.split("\n"):
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        base = name[len(PREFIX):] if name.startswith(PREFIX) else name
        if base in raw:
            tail = line.rsplit(" ", 1)[-1]
            try:
                v = float(tail)
            except ValueError:
                continue
            if math.isfinite(v):
                out[base] = out.get(base, 0.0) + v
    mm = re.search(r'model_name="([^"]+)"', text)
    out["_model"] = None if mm is None else mm.group(1)
    return out


def _clients(port: int):
    """nvtop-style attribution: who holds established sockets to the server."""
    try:
        r = subprocess.run(["ss", "-tnp", "state", "established", "dst", ":" + str(port)],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None, []
    if r.returncode != 0:
        return None, []
    agg: dict[str, dict] = {}
    pm = []
    for line in r.stdout.split("\n")[1:]:
        if "users:" not in line:
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        peer = cols[3].rsplit(":", 1)[0].strip("[]")
        try:
            proc = line.split('(("', 1)[1].split('"', 1)[0]
        except IndexError:
            proc = "?"
        try:
            pm.append((int(cols[2].rsplit(":", 1)[1]), proc, peer))
        except (IndexError, ValueError):
            pass
        k = proc + "@" + peer
        e = agg.setdefault(k, {"proc": proc, "addr": peer, "n": 0})
        e["n"] += 1
    return (sorted(agg.values(), key=lambda e: -e["n"]) or None), pm

_REQ_TAIL = re.compile(r"\"(\w+) (\S+) HTTP/[\d.]+\" (\d{3})")
_PEER = re.compile(r"(\[[0-9a-fA-F:.]+\]|[\w.\-]+):(\d+)")
_TS = re.compile(r"\[([^\]]*)\]")


def _parse_req(line):
    g = _REQ_TAIL.search(line)
    if not g:
        return None
    p = _PEER.search(line)
    if not p:
        return None
    t = _TS.search(line)
    return p.group(1), int(p.group(2)), (t.group(1) if t else ""), g.group(1), g.group(2), g.group(3)


def _container_id(port: int) -> str | None:
    """Discover the container publishing the inference port, via cgroup hints."""
    try:
        r = subprocess.run(["ss", "-tlnpe"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in r.stdout.split("\n"):
        if ":" + str(port) in line and "docker-" in line:
            m = re.search(r"docker-([0-9a-f]{16,64})", line)
            if m:
                return m.group(1)[:24]
    return None


def _docker_reqs(cid: str) -> str:
    try:
        r = subprocess.run(["docker", "logs", "--since", "4s", cid],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


class Collector:
    def __init__(self, url: str, period: float, keep: int) -> None:
        self.url = url
        self.period = period
        self.samples: deque = deque(maxlen=keep)
        self.prev: dict | None = None
        self.prev_t: float | None = None
        self.down = 0
        self.gpus_last = None
        self.hist = deque(maxlen=4200)
        self.clients_last = None
        self.tickn = 0
        self.sock_last = 0
        self.nethist = deque(maxlen=40)
        self.port = urlparse(url).port or 80
        self.cid = _container_id(self.port)
        self.chist: dict[str, dict] = {}
        self.pmap = deque(maxlen=800)
        self.rq_ring = deque(maxlen=8)
        self.seen: dict = {}
        self.rq_prev = None

    def tick(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url, timeout=4) as r:
                text = r.read().decode("utf-8", "replace")
        except OSError:
            self.down += 1
            return None
        m = parse(text)
        now = time.time()
        if self.prev is None or self.prev_t is None:
            self.prev, self.prev_t = m, now
            return None
        dt = max(0.15, now - self.prev_t)
        d = lambda k: max(0.0, m.get(k, 0.0) - self.prev.get(k, 0.0))
        prefill = d("prompt_tokens_total") / dt
        decode = d("generation_tokens_total") / dt
        run = m.get("num_requests_running", 0.0)
        wait = m.get("num_requests_waiting", 0.0)
        phase = "idle"
        if prefill > 10 and prefill > 3 * decode:
            phase = "prefill"
        elif decode > 2 and prefill <= 10:
            phase = "decode"
        elif prefill > 10 or decode > 2:
            phase = "prefill+decode"
        ttc = d("time_to_first_token_seconds_count")
        itc = d("inter_token_latency_seconds_count")
        acc, dr = d("spec_decode_num_accepted_tokens_total"), d("spec_decode_num_draft_tokens_total")
        s = {
            "t": round(now, 2),
            "phase": phase,
            "prefill_tps": round(prefill, 1),
            "decode_tps": round(decode, 1),
            "run": int(run), "wait": int(wait),
            "kv": round(m.get("kv_cache_usage_perc", 0.0), 4),
            "ttft_ms": None if ttc <= 0 else round(d("time_to_first_token_seconds_sum") / ttc * 1e3, 1),
            "itl_ms": None if itc <= 0 else round(d("inter_token_latency_seconds_sum") / itc * 1e3, 2),
            "mtp_accept": None if dr <= 0 else round(acc / dr, 3),
            "preempt": int(d("num_preemptions_total")),
            "gpus": gpus_sample(),
        }
        if s["gpus"] is None:
            s["gpus"] = self.gpus_last
        else:
            self.gpus_last = s["gpus"]
        self.nethist.append((now, m.get("http_request_size_bytes_sum", 0.0),
                            m.get("http_response_size_bytes_sum", 0.0), m.get("http_requests_total", 0.0)))
        while len(self.nethist) > 2 and now - self.nethist[0][0] > 7.0:
            self.nethist.popleft()
        nb = self.nethist[0]
        dtw = max(0.1, now - nb[0])
        s["net"] = {"in": round(max(0.0, m.get("http_request_size_bytes_sum", 0.0) - nb[1]) / dtw, 1),
                    "out": round(max(0.0, m.get("http_response_size_bytes_sum", 0.0) - nb[2]) / dtw, 1),
                    "req": round(max(0.0, m.get("http_requests_total", 0.0) - nb[3]) / dtw, 2)}
        self.hist.append((now, m.get("request_success_total", 0.0),
                          m.get("prompt_tokens_total", 0.0), m.get("generation_tokens_total", 0.0),
                          m.get("num_requests_running", 0.0) + m.get("num_requests_waiting", 0.0)))
        while self.hist and now - self.hist[0][0] > 1830.0:
            self.hist.popleft()
        b = self.hist[0]
        mins = max(0.1, (now - b[0]) / 60.0)
        nreq = max(0.0, m.get("request_success_total", 0.0) - b[1])
        s["win"] = {"min": round(mins, 1), "reqs": int(nreq),
                    "peak": int(max([e[4] for e in self.hist] + [0.0])),
                    "per_min": round(nreq / mins, 2),
                    "avg_in": None if nreq <= 0 else int((m.get("prompt_tokens_total", 0.0) - b[2]) / nreq),
                    "avg_out": None if nreq <= 0 else int((m.get("generation_tokens_total", 0.0) - b[3]) / nreq)}
        self.tickn += 1
        if self.tickn % 4 == 0:
            cs, pm = _clients(self.port)
            if cs is not None:
                self.clients_last = cs
                self.sock_last = sum(c["n"] for c in cs)
                live = set()
                for e in cs:
                    k = e["proc"] + "@" + e["addr"]
                    live.add(k)
                    h = self.chist.get(k) or {"proc": e["proc"], "addr": e["addr"], "n": 0, "last": now}
                    h["n"] = e["n"]
                    h["last"] = now
                    self.chist[k] = h
                for k in list(self.chist):
                    if k not in live:
                        self.chist[k]["n"] = 0
                    if now - self.chist[k]["last"] > 180.0:
                        del self.chist[k]
            for e in pm:
                self.pmap.append((now, e[0], e[1], e[2]))
            while len(self.pmap) > 800:
                self.pmap.popleft()
        if self.cid and self.tickn % 4 == 2:
            dsucc = dp = dg = 0.0
            if self.rq_prev is not None:
                dsucc = max(0.0, m.get("request_success_total", 0.0) - self.rq_prev[0])
                dp = max(0.0, m.get("prompt_tokens_total", 0.0) - self.rq_prev[1])
                dg = max(0.0, m.get("generation_tokens_total", 0.0) - self.rq_prev[2])
            self.rq_prev = (m.get("request_success_total", 0.0), m.get("prompt_tokens_total", 0.0),
                            m.get("generation_tokens_total", 0.0))
            per_i = None if dsucc <= 0 else dp / dsucc
            per_o = None if dsucc <= 0 else dg / dsucc
            approx = dsucc > 1.5
            for line in _docker_reqs(self.cid).split("\n"):
                rp = _parse_req(line)
                if not rp or rp[3].upper() != "POST" or "completions" not in rp[4]:
                    continue
                key = (rp[0], rp[1], rp[4], rp[5])
                if key in self.seen:
                    continue
                self.seen[key] = True
                if len(self.seen) > 500:
                    self.seen.clear()
                port_i = rp[1]
                proc = next((e[2] for e in reversed(self.pmap)
                            if e[1] == port_i and now - e[0] < 300.0), "?")
                path = "chat" if "chat" in rp[4] else "cmp"
                hhmm = ((rp[2] or "").split(" ")[-1] or
                       time.strftime("%H:%M:%S", time.localtime(now)))[:8]
                ti = "~" if per_i is None else (("~" if approx else "") + str(int(per_i)))
                to = "~" if per_o is None else (("~" if approx else "") + str(int(per_o)))
                self.rq_ring.append({"hhmm": hhmm, "proc": proc, "path": path,
                                     "ti": ti, "to": to, "code": rp[5]})
        s["clients"] = self.clients_last
        s["socks"] = self.sock_last
        s["callers_active"] = sorted(self.chist.values(), key=lambda e: -e["last"])[:6]
        s["rq"] = list(self.rq_ring)[-4:]
        self.prev, self.prev_t = m, now
        self.down = 0
        self.samples.append(s)
        return s

    def publish(self) -> None:
        now = time.time()
        last_t = self.samples[-1]["t"] if self.samples else 0.0
        if self.down > 6:
            status = "down"
        elif now - last_t > 3.0:
            status = "stale"
        else:
            status = "ok"
        payload = {
            "meta": {"url": self.url, "model": self.prev.get("_model") if self.prev else None,
                    "collected_at": round(now, 1), "period": self.period, "status": status,
                    "age": round(now - last_t, 1),
                    "note": "windowed rates from /metrics counters; phase per engine tick"},
            "samples": list(self.samples),
        }
        text = json.dumps(payload, separators=(",", ":"))
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, OUT)


def main() -> int:
    ap = argparse.ArgumentParser(description="vllama -> atlas telemetry feed")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--keep", type=int, default=240)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    c = Collector(a.url, a.interval, a.keep)
    if a.once:
        c.tick()
        s = c.tick()
        print(json.dumps(s, indent=2, sort_keys=True))
        return 0 if s else 1
    print(f"telemetry: {a.url} -> {OUT} every {a.interval}s", flush=True)
    while True:
        t0 = time.time()
        c.tick()
        c.publish()
        time.sleep(max(0.05, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
