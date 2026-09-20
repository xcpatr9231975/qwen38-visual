#!/usr/bin/env python3
"""query tagger: loopback proxy :8010 -> :8009.

Streams every request through untouched (incl. SSE) and records a local-only
ring of one-line query snippets in live/queries.json. Caller identity comes
from SO_PEERCRED on the accepted socket, so it is authoritative, not sampled.
Privacy: set ATLAS_TAG=0 to record metadata only (no snippet text).
"""
from __future__ import annotations

import http.client
import subprocess
import json
import os
import re
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

UP_HOST, UP_PORT = "127.0.0.1", 8009
BIND, PORT = "127.0.0.1", int(os.environ.get("ATLAS_TAG_PORT", "8010"))
HERE = Path(__file__).resolve().parent
OUT = HERE / "queries.json"
MAXR, SNAP = 24, 90
TAG = os.environ.get("ATLAS_TAG", "1") != "0"

_ring: deque[dict] = deque(maxlen=MAXR)
_pccache: dict = {}
_lock = threading.Lock()
_ws = re.compile(r"\s+")


def _procname(pid: int) -> str:
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "?"
    try:
        argv = open(f"/proc/{pid}/cmdline").read().rsplit("\0")
    except OSError:
        return comm or "?"
    if argv and os.path.basename(argv[0]).startswith("python") and len(argv) > 1:
        return os.path.basename(argv[1]).removesuffix(".py") or comm
    return comm or "?"
def peer(conn, addr) -> tuple[str, int]:
    try:
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        pid, _uid, _gid = struct.unpack("=iii", raw)
        if pid > 0:
            return _procname(pid), pid
    except Exception:
        pass
    now = time.time()
    try:
        lport = addr[1]
        key = (addr[0], lport)
        hit = _pccache.get(key)
        if hit and now - hit[0] < 5.0:
            return hit[1], hit[2]
        r = subprocess.run(["ss", "-tnp", "state", "established", "dst", ":" + str(PORT)],
                          capture_output=True, text=True, timeout=2)
        for line in r.stdout.split("\n")[1:]:
            cols = line.split()
            if len(cols) >= 4 and cols[2].rsplit(":", 1)[-1] == str(lport) and "users:" in line:
                try:
                    proc = line.split('(("', 1)[1].split('"', 1)[0]
                except IndexError:
                    proc = "?"
                mtp = re.search(r"pid=(\d+)", line)
                pid = int(mtp.group(1)) if mtp else 0
                _pccache[key] = (now, proc, pid)
                return proc, pid
    except Exception:
        pass
    return "?", 0



def _snip(body: bytes) -> str | None:
    if not TAG:
        return None
    try:
        j = json.loads(body)
    except (ValueError, UnicodeError):
        return None
    text = None
    if isinstance(j.get("messages"), list) and j["messages"]:
        for m in reversed(j["messages"]):
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                text = c
                break
            if isinstance(c, list):
                text = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
                if text.strip():
                    break
    elif isinstance(j.get("prompt"), str):
        text = j["prompt"]
    elif isinstance(j.get("prompt"), list) and j["prompt"]:
        text = " ".join(str(x) for x in j["prompt"][:1])
    if not text:
        return None
    t = _ws.sub(" ", text).strip()
    return (t[:SNAP] + "…") if len(t) > SNAP else t


def _publish() -> None:
    with _lock:
        doc = {"meta": {"updated": round(time.time(), 2), "port": PORT, "upstream": UP_PORT,
                        "tag": TAG}, "requests": list(_ring)}
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, OUT)


def _record(rec: dict) -> None:
    with _lock:
        _ring.appendleft(rec)
    _publish()


def _handle(conn, addr) -> None:
    proc, pid = peer(conn, addr)
    conn.settimeout(300.0)
    try:
        rf = conn.makefile("rb")
        line = rf.readline(8192)
        if not line:
            return
        meth, path, _v = line.decode("latin-1").split(" ", 2)
        headers = {}
        while True:
            h = rf.readline(8192)
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length") or 0)
        body = rf.read(n) if n else b""
        snippet = _snip(body) if "completions" in path else None
        fwd = {k: v for k, v in headers.items() if k not in ("host", "connection", "proxy-connection", "transfer-encoding")}
        fwd["host"] = UP_HOST + ":" + str(UP_PORT)
        fwd["connection"] = "close"
        t0 = time.time()
        up = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=300)
        up.request(meth, path, body=body if body else None, headers=fwd)
        resp = up.getresponse()
        hdrs = [(k, v) for k, v in resp.getheaders() if k.lower() not in ("transfer-encoding", "connection", "keep-alive")]
        head = "\r\n".join([f"HTTP/1.1 {resp.status} {resp.reason}", *[f"{k}: {v}" for k, v in hdrs]]).encode("latin-1") + b"\r\n\r\n"
        conn.sendall(head)
        total = 0
        while True:
            ch = resp.read(65536)
            if not ch:
                break
            conn.sendall(ch)
            total += len(ch)
        up.close()
        raw_len = total
        _record({"t": round(t0, 1), "hhmm": time.strftime("%H:%M:%S", time.localtime(t0)),
                 "proc": proc, "pid": pid, "ip": addr[0], "path": ("chat" if "chat" in path else ("cmp" if "completions" in path else path.split("/")[-1])),
                 "model": None, "snip": snippet, "req": len(body), "resp": raw_len,
                 "code": resp.status, "ms": int((time.time() - t0) * 1e3)})
    except Exception:
        try:
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass
    finally:
        try:
            conn.close()
        except OSError:
            pass
def main() -> None:
    sv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sv.bind((BIND, PORT))
    sv.listen(64)
    print(f"query tagger :{PORT} -> http://{UP_HOST}:{UP_PORT}  ring={OUT}", flush=True)
    while True:
        c, a = sv.accept()
        threading.Thread(target=_handle, args=(c, a), daemon=True).start()


if __name__ == "__main__":
    main()
