#!/usr/bin/env python3
"""
monitor.py -- nvitop-style live text view of the Qwen3.8-Flash-Next weight atlas.

Reads live/manifest.json (produced by tools/build_manifest.py) and renders the
same telemetry the WebGL atlas shows, in the terminal:

  * per-category allocation bars (real on-disk bytes, colored)
  * per-layer heat strip (48 decoder towers + MTP band), height-bucketed glyphs
  * quant-bit schedules per tensor family
  * live re-read whenever the manifest file changes on disk (nvitop-style refresh)

Usage:
  python3 live/monitor.py                 # curses UI, watch manifest
  python3 live/monitor.py --plain         # single text snapshot (no curses)
  python3 live/monitor.py --regen 10      # re-run the builder every 10s too
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

try:
    import curses
except ImportError:
    curses = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MANIFEST = os.path.join(ROOT, "live", "manifest.json")
TELEMETRY = os.path.join(ROOT, "live", "telemetry.json")
BUILDER = os.path.join(ROOT, "tools", "build_manifest.py")

# nvitop-style shade ramp, dark -> full
RAMP = " .:-=+*#%@$"

CAT_COLOR = {
    "ple": 5, "expert": 3, "shared": 2, "gate": 6, "index": 4, "attn": 4,
    "mamba": 6, "hyper": 5, "vision": 1, "embed": 3, "head": 1, "mtp": 2,
    "norm": 7, "other": 7,
}


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def gib(n: int) -> str:
    return f"{n / 2**30:.2f}"


def bar(n: int, nmax: int, width: int) -> str:
    """nvitop-style block bar with a shade ramp tail."""
    if nmax <= 0:
        return ""
    frac = n / nmax
    full = int(frac * width)
    out = "@" * full
    if full < width and full * 1.0 / width < frac:
        rem = (frac * width - full) * len(RAMP)
        out += RAMP[min(len(RAMP) - 1, max(1, int(rem)))]
    return out.ljust(width)


class Model:
    def __init__(self, path: str) -> None:
        self.path = path
        self.mtime = 0
        self.data = None
        self.reload()

    def changed(self) -> bool:
        try:
            return int(os.path.getmtime(self.path)) != self.mtime
        except OSError:
            return False

    def reload(self) -> bool:
        if not self.changed():
            return False
        self.data = load_manifest(self.path)
        self.mtime = int(os.path.getmtime(self.path))
        self.at = time.time()
        return True


def header_lines(m: Model) -> list[str]:
    d = m.data
    t, a = d["totals"], d["arch"]
    age = max(0, int(time.time() - m.at))
    return [
        f" qwen3.8-flash-next atlas  {d['meta']['model']}  digest {d['meta']['digest']}",
        f" atlas {t['gib_total']} GiB  on-disk {t['gib_on_disk']} GiB  "
        f"tensors {t['tensors']:,}  cubes {t['groups']:,}  "
        f"layers {a['layers']}  experts {a['num_experts']}({a['experts_per_token']}/tok)  "
        f"quant {(a.get('quant') or {}).get('method')} {(a.get('quant') or {}).get('bits')}b  "
        f"[{age}s ago]",
    ]


def category_lines(d: dict, sort: str, width: int) -> list[tuple[str, str, int]]:
    """-> (label, bar, color_id). Real bytes, sorted as requested."""
    cats = d["totals"]["by_category"]
    items = list(cats.items())
    items.sort(key=lambda kv: -kv[1]["bytes"])
    if sort == "count":
        items.sort(key=lambda kv: -kv[1]["tensors"])
    nmax = max((e["bytes"] for _, e in items), default=1)
    barw = max(8, width - 46)
    out = []
    for c, e in items:
        label = f" {e['label']:<20} {e['gib']:>8.3f} GiB {e['tensors']:>7,}t"
        out.append((label, bar(e["bytes"], nmax, barw), CAT_COLOR.get(c, 7)))
    return out


def layer_strip(d: dict, width: int) -> str:
    """48 decoder towers + MTP band as one heat line (nvitop memory-map style)."""
    layers = [l for l in d["layers"] if l["i"] < 48]
    mtp = [l for l in d["layers"] if l["i"] >= 1000]
    vals = [l["bytes"] for l in layers] + [l["bytes"] for l in mtp]
    vmax = max(vals, default=1) or 1
    cells = []
    full_ids = []
    for l in layers:
        lvl = int(l["bytes"] / vmax * (len(RAMP) - 1))
        glyph = RAMP[max(1, min(len(RAMP) - 1, lvl))]
        if (l.get("type") or "").startswith("full"):
            full_ids.append(l["i"])
        cells.append(glyph if l["bytes"] else ".")
    row = "".join(cells)
    if mtp:
        lvl = int(mtp[0]["bytes"] / vmax * (len(RAMP) - 1))
        row += "|" + RAMP[max(1, min(len(RAMP) - 1, lvl))] + " " * 3
    row = row.ljust(width - 14)
    tag = " full@" + ",".join(map(str, full_ids)) if full_ids else ""
    return f" layers {' ' * 3}{row}  48+MTP  max {max(vals) / 2**30 if vals else 0:.2f} GiB{tag}"


def quant_lines(d: dict, limit: int) -> list[str]:
    qf = d.get("quant_bits_by_family") or {}
    rows = []
    for fam, bits in sorted(qf.items()):
        if not bits:
            continue
        short = fam.replace("model.language_model.layers.N.", "L.")
        bits_s = ",".join(str(b) for b in sorted(bits))
        rows.append(f" {short:<44} {bits_s}")
    return rows[:limit]


def plain_snapshot(m: Model) -> None:
    d = m.data
    for line in header_lines(m):
        print(line)
    print()
    for label, b, _ in category_lines(d, "bytes", 90):
        print(f"{label} |{b}|")
    print()
    print(layer_strip(d, 90))
    print()
    print(" quant bits / family")
    for r in quant_lines(d, 30):
        print(r)
    tele = _tele()
    if tele:
        print()
        print(" inference (live telemetry)")
        for line, _ in inference_lines(90):
            print(line)


_tele_cache = [None, 0]


def _tele():
    path = TELEMETRY
    try:
        mt = int(os.path.getmtime(path))
    except OSError:
        return None
    if _tele_cache[1] == mt:
        return _tele_cache[0]
    try:
        with open(path) as f:
            _tele_cache[0] = json.load(f)
        _tele_cache[1] = mt
    except (OSError, ValueError):
        return None
    return _tele_cache[0]


def p95(vals):
    sv = sorted(vals)
    return max(0.1, sv[min(len(sv) - 1, int(len(sv) * 0.95))] if sv else 0.1)


def _spark(vals, width):
    sv = sorted(vals)
    mx = max(1.0, sv[min(len(sv) - 1, int(len(sv) * 0.95))] if sv else 1.0)
    cells = [RAMP[max(1, min(len(RAMP) - 1, int(v / mx * (len(RAMP) - 1))))] for v in vals]
    return ("".join(cells) + "." * width)[:width]


def inference_lines(width=90):
    d = _tele()
    if not d or not d.get("samples"):
        return [(" no telemetry -- run: python3 live/telemetry_collector.py", 7)]
    ss = d["samples"]
    last = ss[-1]
    win = ss[-48:]
    phase = last.get("phase") or "-"
    pc = {"prefill": 4, "decode": 3, "prefill+decode": 2}.get(phase, 7)
    out = [(f" inference {d['meta'].get('model') or 'vllm'}  phase {phase.upper():<15}"
            f" run {last.get('run', 0)}  wait {last.get('wait', 0)}  [{len(ss)} samples]", pc)]
    out.append((f" prefill {_spark([float(x.get('prefill_tps') or 0.0) for x in win], width - 22)}"
                f" cur {float(last.get('prefill_tps') or 0.0):>7.1f}  peak {p95([float(x.get('prefill_tps') or 0.0) for x in win]):>7.1f} t/s", 4))
    out.append((f" decode  {_spark([float(x.get('decode_tps') or 0.0) for x in win], width - 22)}"
                f" cur {float(last.get('decode_tps') or 0.0):>7.1f}  peak {p95([float(x.get('decode_tps') or 0.0) for x in win]):>7.1f} t/s", 3))
    out.append((f" conc    {_spark([float((x.get('run') or 0) + (x.get('wait') or 0)) for x in win], width - 22)}"
                f" run {last.get('run', 0)}  wait {last.get('wait', 0)}  socks {last.get('socks', '-')}", 5))
    for x in (last.get("gpus") or []):
        u = 0.0 if x.get("util") is None else float(x["util"])
        c = 1 if u > 85 else (2 if u > 60 else 6)
        clk = "-" if x.get("clk") is None else str(int(x["clk"]))
        pw = "-" if x.get("pw") is None else f"{x['pw']:.0f}"
        tp = "-" if x.get("t") is None else str(int(x["t"]))
        enc = x.get("enc") or 0.0
        dec = x.get("dec") or 0.0
        out.append((f" gpu{x['i']} {x['name']:<10} |{bar(u, 100, 22)}| {u:5.1f}%  "
                    f"{clk}MHz  {pw}W  {tp}C  enc {enc:.0f}% dec {dec:.0f}%  P:{x.get('perf') or '-'}", c))
    net = last.get("net") or {}
    def bw(v):
            if v is None: return "-"
            return f"{v / 2**20:.2f} MB/s" if v >= 2**20 else f"{v / 1024:.1f} KB/s"
    out.append((f" net    in {bw(net.get('in'))}  out {bw(net.get('out'))}   req {net.get('req', '-')}/s", 7))
    wn = last.get("win") or {}
    if wn.get("reqs") is not None:
        out.append((f" {wn.get('min', '-')}min  {wn['reqs']} reqs ({wn.get('per_min', '-')}  per-min)   avg "
                    f"{wn.get('avg_in') if wn.get('avg_in') is not None else '-'} in / "
                    f"{wn.get('avg_out') if wn.get('avg_out') is not None else '-'} out tok  "
                    f"| peak conc {wn.get('peak', '-')}", 7))
    cs = last.get("callers_active") or last.get("clients") or []
    if cs:
        out.append((" callers " + "  ".join(("[" + c["proc"] + " x" + str(c["n"]) + "]" if c.get("n") else "(" + c["proc"] + ")")
                                            for c in cs[:5]), 5))
    for q in list(last.get("rq") or [])[::-1][:3]:
        out.append((f"  q {q['hhmm']} {str(q['proc'])[:10]:<10} {q['path']}  {q['ti']} -> {q['to']} tok  [{q['code']}]", 6))
    def ms(v): return "-" if v is None else f"{v}ms"
    mtp = "-" if last.get("mtp_accept") is None else f"{int(last['mtp_accept'] * 100)}%"
    kv = "-" if last.get("kv") is None else f"{int((last.get('kv') or 0.0) * 100)}%"
    out.append((f" ttft {ms(last.get('ttft_ms'))}  itl {ms(last.get('itl_ms'))}  mtp {mtp}"
                f"  kv {kv}  preempt {last.get('preempt', 0)}", 7))
    return out


def run_curses(m: Model, fps: float) -> None:
    def ui(stdscr):
        curses.use_default_colors()
        curses.start_color()
        for pair, fg in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)):
            curses.init_pair(pair, fg, -1)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        sort, view, tick = "bytes", "atlas", 0.0
        while True:
            now = time.time()
            if m.changed():
                m.reload()
            if now - tick > 1.0 / max(0.5, fps):
                tick = now
                stdscr.erase()
                h, w = stdscr.getmaxyx()
                y = 0
                for line in header_lines(m):
                    stdscr.addnstr(y, 0, line[: w - 1], w - 1); y += 1
                y += 1
                if view == "quant":
                    for r in quant_lines(m.data, h - y - 3):
                        stdscr.addnstr(y, 0, r[: w - 1], w - 1); y += 1
                elif view == "inference":
                    for r, c in inference_lines(w):
                        stdscr.attron(curses.color_pair(c))
                        stdscr.addnstr(y, 0, r[: w - 1], w - 1)
                        stdscr.attroff(curses.color_pair(c))
                        y += 1
                else:
                    stdscr.addnstr(y, 0, f" category {'─' * (w - 12)}", w - 1); y += 1
                    for label, b, c in category_lines(m.data, sort, w):
                        try:
                            stdscr.attron(curses.color_pair(c))
                        except curses.error:
                            pass
                        stdscr.addnstr(y, 0, f"{label} |{b}|"[: w - 1], w - 1)
                        stdscr.attroff(curses.color_pair(c))
                        y += 1
                    y += 1
                    stdscr.addnstr(y, 0, layer_strip(m.data, w)[: w - 1], w - 1); y += 2
                bot = " q quit  r reload  s sort(%s)  v view(%s)  live on manifest change " % (sort, view)
                stdscr.addnstr(h - 1, 0, bot[: w - 1], w - 1)
                stdscr.refresh()
            if float(os.environ.get("MON_AUTOCLOSE", "0") or 0) > 0 \
               and time.time() - m.at > float(os.environ.get("MON_AUTOCLOSE", "0")):
                return
            ch = stdscr.getch()
            if ch in (ord("q"), 27):
                return
            if ch in (ord("r"),):
                m.data = load_manifest(m.path); m.at = time.time()
            if ch == ord("s"):
                sort = "count" if sort == "bytes" else "bytes"
            if ch == ord("v"):
                seq = ("atlas", "quant", "inference"); view = seq[(seq.index(view) + 1) % 3]
            time.sleep(0.02)
    curses.wrapper(ui)


def main() -> int:
    ap = argparse.ArgumentParser(description="nvitop-style text atlas monitor")
    ap.add_argument("--path", default=DEFAULT_MANIFEST)
    ap.add_argument("--plain", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--regen", type=int, default=0,
                    help="also re-run the builder every N seconds (0=off)")
    a = ap.parse_args()
    if not os.path.isfile(a.path):
        print(f"FAIL: manifest not found: {a.path} -- run: python3 tools/build_manifest.py")
        return 2

    m = Model(a.path)
    last_gen = 0.0
    if a.regen:
        subprocess.run([sys.executable, BUILDER], check=False)
        last_gen = time.time()
        m.reload()

    if a.plain or not sys.stdout.isatty() or curses is None:
        plain_snapshot(m)
        return 0
    try:
        run_curses(m, a.fps)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
