# Qwen3.8 Visual Atlas — nvtop for a local inference stack

A self-contained, zero-dependency observability deck for serving
`Qwen3.8-Flash-Next` (EXL3-style safetensors pack) on a GB10 box via a
vllama-compatible OpenAI server. It renders the model's **weight layout as a
real WebGL scene**, and overlays it with **live engine, GPU, bandwidth,
caller, and query telemetry** — nvtop-style, in both a browser and a
terminal.

Everything is read-only against the serving stack: nothing here is in the
request path unless you explicitly opt the query tagger in, and no byte ever
leaves the box.

---

## Table of contents

1. [Design doctrine](#1-design-doctrine)
2. [Feature summary](#2-feature-summary)
3. [Architecture](#3-architecture)
4. [Repository layout](#4-repository-layout)
5. [Requirements](#5-requirements)
6. [Quickstart](#6-quickstart)
7. [Component reference](#7-component-reference)
   - [tools/build_manifest.py](#71-toolsbuild_manifestpy-the-manifest-builder)
   - [web/index.html](#72-webindexhtml-the-webgl-atlas)
   - [live/monitor.py](#73-livemonitorpy-the-terminal-view)
   - [live/telemetry_collector.py](#74-livetelemetry_collectorpy-the-telemetry-heart)
   - [live/query_tagger.py](#75-livequery_taggerpy-opt-in-query-snippets)
8. [Telemetry contract](#8-telemetry-contract)
9. [Platform findings (GB10 / container quirks)](#9-platform-findings-gb10--container-quirks)
10. [Testing & verification](#10-testing--verification)
11. [Operations runbook](#11-operations-runbook)
12. [Troubleshooting](#12-troubleshooting)
13. [Privacy & security posture](#13-privacy--security-posture)
14. [Deliberate exclusions](#14-deliberate-exclusions)

---

## 1. Design doctrine

**Measured vs. derived is sacred.** Every pixel answers "where did you come
from":

| Class | Meaning | Examples |
|---|---|---|
| *measured* | a real counter/gauge | tokens/s, KV fill %, GPU util, temp, req counters, status codes |
| *joined* | measured from two sources | caller process (`ss` peer-port → proc), request lines (container log + counter deltas) |
| *derived* | honest animation of measured clocks | layer sweep = token clock; MTP trail = accept fraction |

Derived visuals are always tagged in the HUD (`tower sweep · derived from
live token clock + mtp accept`) and **freeze when the system is idle** —
nothing moves in the scene unless something is actually happening.

**Nothing faked where data doesn't exist.** The GB10 is unified-memory: there
is *no* dedicated VRAM counter, so the atlas never draws a "VRAM" bar. The KV
pool gauge is labelled as the pool proxy it is. `nvidia-smi dmon` refuses this
driver, so memory bandwidth is not displayed; API byte-rate is.

**The build never lies about the pack.** The builder reads safetensors
headers only (never 79 GiB of weight bytes) and the atlas is byte-accounted
against the on-disk pack, surfacing the delta (headers/shard padding) instead
of hiding it.

---

## 2. Feature summary

**WebGL atlas (browser, no framework, no assets)**

- 48 decoder towers + MTP band; segment heights = real per-category bytes
- category islands: one cube per weight template, edge = cube-root of bytes
- `#map` memory-map view: 0.25 GiB allocation blocks, colored by family
- quant-bit shimmer per cube (brighter = higher average quant bits)
- live HUD sparklines: PREFILL, DECODE, CONC, MTP (+ GPU per-device sparks)
- GPU rows: util %, SM GHz, watts, °C — green/amber/red thresholds
- KV pool: 32 blocks + 8 request lanes, phase-tinted (prefill blue, decode
  amber, idle green), hoverable
- MTP strip: runner strides per verify step, trail depth = live accept rate
- callers as hash-colored chips (solid = live sockets, ghost = seen <3 min)
- `q` lines: per-request one-liners (caller, endpoint/kind, tokens, status,
  snippet when tagged)
- net row: API bandwidth with KB/MB auto-scaling; 30-min window row:
  requests, rate, avg prompt/generation tokens, peak concurrency

**Terminal monitor (curses + `--plain` for CI)**

- same live surfaces, ASCII ramp sparklines, colored bars, view cycling

**Collector**

- vllama `/metrics` → typed samples (512 ms cadence), `ok/stale/down`
  freshness status
- `nvidia-smi` GPU scrape, `ss` socket/caller probe, `docker logs` request
  join, query-tagger feed — all degrade gracefully if absent
- 24 h GPU busy/idle rollup at 5-min buckets (persisted `live/gpu_stats.json`),
  surfaced in the web HUD as a heat strip with used/idle/unmonitored totals —
  *unmonitored never renders as idle*

**Builder**

- deterministic/idempotent output (timestamp-aware), pack-digest guard with
  exit code `3` if the pack mutates mid-build

---

## 3. Architecture

```
                     ┌───────────────────────────┐
   model pack        │  tools/build_manifest.py  │   run on demand or via
 /home/.../EXL3 ────▶│  (safetensors headers)    │   a regen loop (idempotent)
                     └───────┬───────────┬───────┘
                             │           │
                     web/atlas_data.js   live/manifest.json
                             │           │
   ┌─────────────────────────▼─┐   ┌─────▼──────────────┐
   │ python3 -m http.server    │   │ live/monitor.py    │  terminal
   │ :8000  (serve repo root)  │   │  atlas│quant│inference
   │  └─▶ web/index.html       │   └─────▲──────────────┘
   └───────────────────────────┘         │
                                        │ live/telemetry.json
   vllama engine :8009 ──/metrics───────┤───────────────────────┐
        │                               │                       │
        └─ docker logs ──(peer-port)─────┤── live/telemetry_collector.py
                                        │      ▲                │
   nvidia-smi (GPU util/temp/clock) ────┤      │                │
   ss (sockets → caller procs) ─────────┘      │                │
                                              │                │
   callers ──▶ :8010 live/query_tagger.py ────┘   live/queries.json
              (opt-in proxy, streaming-safe)      (local-only ring)
```

Key ports: `:8000` web UI · `:8009` inference engine · `:8010` query tagger.

The collector is a **single writer** — always exactly one process publishing
`live/telemetry.json` (see the runbook; this caused real pain, see §11).

---

## 4. Repository layout

```
.gitignore                    ignores runtime outputs + pycache
tools/build_manifest.py       pack → manifest/atlas builder (idempotent)
web/index.html                WebGL2 atlas + telemetry HUD (single file)
web/atlas_data.js             generated atlas payload (committed snapshot)
live/manifest.json            generated text-monitor manifest (snapshot)
live/monitor.py               curses/plain terminal monitor
live/telemetry_collector.py   metrics scraper/publisher → telemetry.json
live/query_tagger.py          opt-in proxy :8010 → :8009 + snippets
test/test_manifest.py         pytest: builder invariants, monitor fmt
test/test_gpu_rollup.py         pytest: 24h rollup bucketing/persistence
deploy/                         systemd --user units + atlas_up.sh installer
```

Runtime artifacts (git-ignored): `live/telemetry.json` (rewritten ~2/s),
`live/queries.json` (holds prompt snippets — local-only by policy),
`live/gpu_stats.json` (24 h busy/idle rollup, ~10 KB).

---

## 5. Requirements

- **Python ≥ 3.10** (developed on 3.13.11); stdlib only — no pip installs
- Linux with `iproute2` (`ss`) for caller attribution
- Optional, each auto-detected & gracefully degrading:
  - `nvidia-smi` — GPU rows (GB10: see §9 for supported fields)
  - `docker` CLI — request-log join (container auto-discovered via `ss`
    cgroup hints; no root needed)
  - a vllama/OpenAI-compatible server exposing Prometheus `/metrics`
- Optional dev tools: `node --check` (JS syntax gate), headless Chromium
  (console-error smoke), `pytest`

---

## 6. Quickstart

From the repo root (the web page fetches `../live/telemetry.json`, so the
server must serve the **root**, not `web/`):

```sh
# 1. (once, or after the pack changes) build the atlas artifacts
python3 tools/build_manifest.py

# 2. serve the UI
python3 -m http.server 8000 >/dev/null 2>&1 &

# 3. publish live telemetry (exactly ONE of these)
python3 live/telemetry_collector.py --interval 0.5 &
#    (preferred: install the supervised, boot-persistent fleet instead —
#     ./deploy/atlas_up.sh, see §11; it also retires stray manual instances)

# 4. open the atlas  (add #map for the memory-map view)
xdg-open http://127.0.0.1:8000/web/#atlas

# 5. terminal view
python3 live/monitor.py            # curses
python3 live/monitor.py --plain    # one snapshot, CI-safe

# 6. optional: query snippets — callers repoint :8009 → :8010
python3 live/query_tagger.py &
```

Point a request at the engine and the deck wakes up: sparklines rise, the
layer sweep starts running, KV blocks light and tint by phase, the MTP strip
strides, and the `q` lines attribute who did what.

---

## 7. Component reference

### 7.1 `tools/build_manifest.py` — the manifest builder

Reads only **safetensors headers** of the pack (path overridable via env
`QWEN_PACK`, default `/home/cpatriaw/models/Qwen3.8-Flash-Next-EXL3`) and
emits both consumers' payloads:

- `web/atlas_data.js` — `const ATLAS = {...};` with a file-header marking it
  generated
- `live/manifest.json` — same content for the terminal monitor

Reference numbers for the current pack (verified): **304,240** tensors,
**782** atlas cubes, **79.112 GiB** attributed vs **79.150 GiB** on-disk
(Δ 0.038 GiB = headers/padding, surfaced as an explicit `delta` summary
row), **48** decoder layers + MTP band, hybrid `layer_types`
(`full_attention` / `linear_attention`), PLE & vision tensors excluded from
the decoder towers, plus a `meta.dtype_divergences` audit list.

Behavior details that matter:

- **idempotent writes** — files are rewritten only if the payload changed
  (ignoring the `built_at` stamp), so a 30 s regen loop never dirties git
  (`kept ... (payload unchanged)` in the log)
- **pack-digest guard** — digests the pack before/after; mid-build mutation
  exits with code `3` rather than publishing a torn manifest
- writes are tmp-file + atomic `os.replace`
- console summary mirrors the byte-accounting table above

Exit codes: `0` ok · `3` pack mutated mid-run.

### 7.2 `web/index.html` — the WebGL atlas

Single file, WebGL2, zero dependencies. Fetches `../live/telemetry.json`
(~1 s poll; re-syncs on tab focus/visibility) and `web/atlas_data.js`.

**Controls** — drag orbit · wheel zoom · double-click reset · keys `1`
orbit / `2` memory map / `3` front · auto-orbit resumes after 4 s idle.

**Scene**

- decoder towers: per-category stacked segments, height = bytes, base color
  = category, brightness scaled by real quant bits
- ground tiles under each tower mark the layer type (blue = full attention,
  teal = linear)
- category islands: cube per weight template, edge = cube-root of real bytes
- `#map`: contiguous 0.25 GiB allocation blocks laid out as a memory map
- **KV pool row**: 32 blocks = `kv_cache_usage_perc`, phase-tinted, height
  grows when filled; 8 **request lanes** light per running sequence
- **GPU row** (per device): util sparkline, `u% · GHz · W · C`, thresholds
  green <60 ≤ amber ≤85 < red
- **MTP strip**: 24 cells; runner = verify-step clock, trail depth =
  `round(mtp_accept × 9)`, dim baseline when idle
- **layer sweep**: cursor advances at measured `decode_tps` (prefill adds
  `prefill_tps/256`); full-attention towers pulse harder than linear ones —
  both are architectural facts weighted by real traffic

**HUD (top-right), in order**: live chip (model · `ok/stale/down` · sample
age) → PREFILL, DECODE, **CONC**, **MTP** spark rows → GPU rows → net row →
30-min window row → caller chips → `q` request lines.

**Derived elements are labelled as such** in the HUD legend, and the sweep
freezes at idle.

### 7.3 `live/monitor.py` — the terminal view

```
python3 live/monitor.py [--path live/manifest.json] [--plain] [--fps 8] [--regen 0]
```

- `--plain` — one snapshot, no curses (CI/log friendly)
- `--fps` — redraw rate; `--regen N` — optionally re-run the builder every
  N s from the monitor itself (0 = off)
- `v` cycles views: **atlas** (weight layout, bars per category) →
  **quant** (bits by family) → **inference** (live telemetry)
- inference view mirrors the web HUD: prefill/decode/conc ASCII sparklines,
  per-GPU bars (`gpu0 GB10 |###| 92% 2481MHz 37W 67C`), net row, window row
  with `peak conc`, `[caller ×n]` chips with `(ghost)` staleness, and `q`
  request lines

### 7.4 `live/telemetry_collector.py` — the telemetry heart

```
python3 live/telemetry_collector.py [--url $VLLAMA_URL] [--interval 0.5] [--keep 240] [--once]
                          [--gpu-busy 5] [--gpu-bucket 300]
```

`VLLAMA_URL` default `http://127.0.0.1:8009/metrics`. `--once` prints one
sample as JSON and exits (great for smoke checks).

Each tick samples: engine counters/gauges (vLLM `vllm:` prefix **and**
unprefixed `http_*` histogram sums; label variants are summed — the server
emits several per handler/status), GPU state, socket/caller state, and the
tagger ring. Published atomically to `live/telemetry.json`.

| sample field | source | notes |
|---|---|---|
| `phase` | prefill/decode counters | `idle` when neither moved |
| `prefill_tps`,`decode_tps` | token counters Δt | rate over last interval |
| `run`,`wait` | engine gauges | scheduler queue state |
| `kv` | `kv_cache_usage_perc` | pool fill (the honest "memory" gauge) |
| `ttft_ms`,`itl_ms` | latency histograms | Δsum/Δcount, non-finite-safe |
| `mtp_accept` | spec-decode counters | accepted/drafted |
| `preempt` | counter delta | scheduler pressure |
| `gpus[]` | `nvidia-smi` | util, mem-util, power, temp, SM clock, enc/dec, `pstate`; `mu/mt/mclk` null where GB10 has none |
| `net{in,out,req}` | `http_*_size_bytes_sum`, `http_requests_total` | B/s over a ~7 s window |
| `win{min,reqs,per_min,avg_in,avg_out,peak}` | long counter history | horizon 1830 s (~30 min), grows from boot |
| `clients`,`socks` | `ss` to the engine port | live sockets per proc |
| `callers_active[]` | caller history | 3-min ghosting window |
| `rq[]` | `docker logs` join | peer-port→proc via socket map (300 s) |
| `qfeed[]` | `live/queries.json` | tagger snippets when opted in |
| `meta.status` | internal | `ok` <2 s, `stale` <10 s, `down` ≥10 s |

Robustness: Prometheus `NaN/Inf` drop guards, per-source try/except (missing
`nvidia-smi`/`docker`/`ss` simply yield `null` sections), atomic publish,
sample ring `--keep` deep.

### 7.5 `live/query_tagger.py` — opt-in query snippets

```
ATLAS_TAG=1 ATLAS_TAG_PORT=8010 python3 live/query_tagger.py &
```

Loopback proxy `:8010 → :8009`. Streaming-safe (SSE passes chunk-by-chunk),
forwards all headers (auth included), never buffers responses. For
`*completions` requests it records a local-only ring (`live/queries.json`,
git-ignored): time, caller **pid+proc**, client IP, endpoint kind, status,
latency, byte counts, and a ≤90-char whitespace-collapsed snippet of the
prompt / last user message.

- caller identity: `SO_PEERCRED` on the accepted socket; where the kernel
  returns zeroed credentials (this GB10 build does), falls back to an `ss`
  lookup keyed on the client's ephemeral port, cached 5 s per connection
- `ATLAS_TAG=0` → metadata only, no snippet text
- **opt-in by design**: direct callers to `:8009` never touch it; repointing
  a caller's `base_url` is the only change needed to gain titled `q` lines
- failure posture: upstream errors become a `502`; it is a single point of
  failure for traffic routed through it (run it supervised)

---

## 8. Telemetry contract

`live/telemetry.json` (atomic replace every `--interval`):

```json
{
  "meta": { "status": "ok", "age": 0.2, "model": "qwen3.8-flash-next",
            "url": "http://127.0.0.1:8009/metrics", "interval": 0.5 },
  "samples": [ { "t": 1789916927.55, "phase": "decode",
      "prefill_tps": 0.0, "decode_tps": 25.3, "run": 1, "wait": 0,
      "kv": 0.11, "ttft_ms": 118.4, "itl_ms": 37.2, "mtp_accept": 0.75,
      "preempt": 0, "gpus": [ { "i": 0, "name": "GB10", "util": 92.0,
          "mem": 0.0, "mu": null, "mt": null, "pw": 38.9, "cap": null,
          "t": 69.0, "clk": 2476.0, "mclk": null, "enc": 0.0, "dec": 0.0,
          "perf": "P0" } ],
      "net": { "in": 88.2, "out": 54.1, "req": 0.46 },
      "win": { "min": 30.0, "reqs": 212, "per_min": 7.07,
               "avg_in": 512, "avg_out": 87, "peak": 5 },
      "clients": [ { "proc": "curl", "addr": "127.0.0.1", "n": 1 } ],
      "socks": 1, "callers_active": [ ... ], "rq": [ ... ], "qfeed": [ ... ] } ]
}
```

Consumers must treat every field as optional (`null`/missing = source
unavailable) — that is what lets sources fail independently.

---

## 9. Platform findings (GB10 / container quirks)

Hard-won field notes baked into the code — revisit these first if the deck
goes blind on new hardware:

1. **Unified memory** — `memory.used/total` are `N/A`; the full table says
   *Memory-Usage Not Supported*. No VRAM bar exists anywhere in this UI by
   policy; KV pool fill is the labelled proxy.
2. `nvidia-smi dmon` **refuses** this driver ("Failed to find any metric to
   display") — memory bandwidth is therefore not measured anywhere.
3. `--query-gpu` field names differ from intuition: `clocks.memory` and
   `performance_state` are **invalid**; use `pstate`, and encoder/decoder busy
   via `utilization.encoder/decoder`.
4. **`SO_PEERCRED` returns pid 0** on this kernel — the tagger's `ss`
   fallback exists because of it.
5. The engine runs in a **docker container** (found via the listener's
   cgroup line in `ss -tlnpe`); its logs carry access lines only — no prompt
   text, which is exactly why the tagger is a proxy, not a log parser.
6. Those access lines appear in **two dialects** (`ip:port - - "POST ..."`
   and `ip:port - "POST ..."` with optional `[timestamp]`); the parser is
   format-tolerant.
7. Prometheus names: usage totals live as `http_*_size_bytes_sum/_count`
   (histograms), not `_total`; label variants must be summed, not last-wins.

---

## 10. Testing & verification

```sh
python3 -m pytest test/ -q          # 17 passed
python3 -m py_compile tools/build_manifest.py live/*.py
```

The suite covers builder invariants (byte accounting vs. on-disk pack,
category totals, cube counts, determinism/idempotence, dtype audit,
exit-3 digest guard) and monitor formatting (missing-field robustness,
sparkline/ramp shapes, window math).

JS has no framework — the CI gate is syntax-level:

```sh
# extract <script> from web/index.html, then:
node --check /tmp/idx.js
```

and a headless smoke that must show zero `uncaught` console lines:

```sh
chromium-browser --headless --no-sandbox --disable-dev-shm \
  --enable-unsafe-swiftshader --use-angle=swiftshader \
  --screenshot=/tmp/atlas.png --window-size=1600,900 \
  --virtual-time-budget=8000 --timeout=15000 http://127.0.0.1:8000/web/
```

---

## 11. Operations runbook

### Boot persistence (the supported way)

A supervised fleet ships in `deploy/` as **systemd `--user` units** plus an
installer — no root, no container restarts:

```sh
./deploy/atlas_up.sh            # install + enable + start (retires strays)
./deploy/atlas_up.sh status     # unit states + endpoint probes
./deploy/atlas_up.sh down       # stop + disable (files remain)
```

| unit | role | restart policy |
|---|---|---|
| `atlas-web` | http.server :8000 from repo root | `always`, 3 s |
| `atlas-collector` | telemetry publisher (**the** single writer) | `always`, 3 s |
| `atlas-tagger` | query tagger :8010→:8009 | `always`, 3 s |
| `atlas-regen` | builder loop every 30 s (idempotent, so it never dirties git) | `always`, 10 s |

Semantics: units are `WantedBy=default.target`, so they start at **login**;
true boot start needs `loginctl enable-linger` (needs admin — the installer
tries and tells you if it couldn't). Crash loops self-heal via
`Restart=always`; single-writer discipline is structural (one unit = one
publisher), and `up` also kills stray manual instances before starting.
Logs: `journalctl --user -u atlas-collector` etc.

The inference **engine container** is external to this repo but the deck is
useless without it — set once: `docker update --restart=unless-stopped
<container>` so it rejoins after reboot too.

Detached launch pattern that survives terminal teardown:

```sh
setsid nohup python3 live/telemetry_collector.py --interval 0.5 \
  >/tmp/telemetry.log 2>&1 </dev/null & disown
```

**The single-writer rule.** `live/telemetry.json` has no merge protocol;
concurrent collectors interleave and the ring corrupts in practice (it did,
twice, in development). Restart ritual:

```sh
pid=$(ps -eo pid,args | grep '[t]elemetry_collec''tor.py --interval' \
     | grep -v grep | awk '{print $1}' | head -1)   # avoid self-match
kill -9 $pid; sleep 0.5
ps -eo args | grep '[t]elemetry_collector.py'      # must be empty
setsid nohup python3 live/telemetry_collector.py --interval 0.5 \
  >/tmp/telemetry.log 2>&1 </dev/null & disown
```

(Do **not** `pkill -f` with a pattern that appears in the command line of
your own wrapper — it can kill the wrapper or match nothing. Kill by PID.)

The atlas regen loop (optional) is safe to leave running because builder
writes are idempotent:

```sh
while sleep 30; do python3 tools/build_manifest.py; done &
```

Health checks: `curl -s 127.0.0.1:8009/metrics | head -1` (engine) ·
`python3 live/telemetry_collector.py --once` (one full sample) ·
`jq '.meta.status' live/telemetry.json` (must be `ok`) ·
`curl -s 127.0.0.1:8010/health` (tagger, if running).

---

## 12. Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| HUD stuck on `awaiting samples` / `stale` | collector not running or engine down | run collector; `--once` to see the error |
| no GPU rows | `nvidia-smi` missing/rejected | check `nvidia-smi --query-gpu=utilization.gpu` |
| `gpus[].mu/mt/mclk` null | GB10 unified memory | expected — KV pool is the proxy gauge |
| no `q` lines at all | `docker` CLI unavailable/forbidden, or container not discovered | `docker logs --tail 5 <cid>`; the discovery is via `ss -tlnpe` cgroups |
| `q` proc shows `?` | very short-lived request socket, missed by the 2 s `ss` cadence | expect occasional `?` on sub-second requests; longer calls join correctly |
| caller chips empty | nobody connected right now (chips ghost out after 3 min) | generate a request |
| `q` has no snippet | traffic goes direct to `:8009` (by design) | route interested callers via `:8010` |
| web page blank | served from wrong root | serve the **repo root** so `../live/telemetry.json` resolves |
| duplicated/contradicting numbers | two collectors racing | apply the runbook single-writer ritual |
| tree dirty after regen | timestamp-only diff pre-idempotence-patch | pull current builder; `kept ... (payload unchanged)` |

---

## 13. Privacy & security posture

- bind targets are loopback only; no external listeners are ever opened
- `live/queries.json` contains prompt text: **git-ignored**, ring capped at
  24 entries, written atomically, never forwarded anywhere; set
  `ATLAS_TAG=0` to record metadata without any text
- `live/telemetry.json` contains counters + caller process names + request
  paths — **git-ignored** as well (it churns every 500 ms)
- no credentials are read, stored, or logged by any component; the tagger
  forwards `Authorization` untouched without recording it
- the collector shells out only to `nvidia-smi`, `ss`, and `docker logs`
  (read-only), each behind a timeout and failure guard

---

## 14. Deliberate exclusions

- **no external services** — no cloud triage/decision APIs, no telemetry
  off-box, by explicit decision
- **no per-weight GPU access counters** — vllama/NVML expose none on this
  platform; the layer sweep is a clock-derived abstraction, clearly labelled
- **no fake VRAM/util bars** — if a counter cannot exist on GB10, the deck
  shows its honest substitute or nothing
- **no runtime dependencies** — browser is vanilla WebGL2, server is stdlib
  Python; the deck must boot on a broken machine, not require a healthy one

---

*Built and verified on NVIDIA GB10 (192 GiB unified), Python 3.13, a
dockerized vllama-compatible engine serving a 79 GiB EXL3-style safetensors
pack of Qwen3.8-Flash-Next.*
