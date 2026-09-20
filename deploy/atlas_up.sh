#!/usr/bin/env bash
# boot-persistence installer for the Qwen3.8 atlas deck (user scope, no root).
#
#   ./deploy/atlas_up.sh            install + enable + start all units
#   ./deploy/atlas_up.sh status     unit status snapshot
#   ./deploy/atlas_up.sh down       stop + disable everything (files remain)
#
# Units install to ~/.config/systemd/user/ and start on login via
# default.target; enable-lingering (when permitted) makes them start at
# boot without a login session.
set -u
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
unitdir="$HOME/.config/systemd/user"
units=(atlas-web atlas-collector atlas-tagger atlas-regen)
cd "$repo" 2>/dev/null || true

_kill_strays() {  # manual instances would double-write telemetry.json
    for pat in 'bin/python3 live/telemetry_collec''tor.py --interval' \
               'bin/python3 live/query_tagger''\.py' \
               'm http.server 8000'; do
        for p in $(ps -eo pid,args | grep -E "$pat" | grep -v grep | awk '{print $1}'); do
            kill -9 "$p" 2>/dev/null && echo "  stopped stray pid $p"
        done
    done
}

case "${1:-up}" in
up)
    mkdir -p "$unitdir"
    cp -f "$here"/*.service "$here"/*.timer "$unitdir"/
    _kill_strays
    systemctl --user daemon-reload
    systemctl --user enable --now "${units[@]}" 2>&1 | grep -v '^$' || true
    loginctl enable-lingering "$(id -un)" 2>/dev/null \
        || echo "  note: no lingering (needs admin); units start at your next login/boot+login"
    sleep 2; exec "$0" status ;;
down)
    systemctl --user disable --now "${units[@]}" 2>/dev/null || true
    echo "units stopped; remove $unitdir/atlas-* to uninstall files" ;;
status)
    for u in "${units[@]}"; do
        printf '%-22s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null)"
    done
    echo "web:     $(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8000/web/)"
    echo "tagger:  $(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8010/health)"
    echo "collect: $(/home/cpatriaw/miniconda3/bin/python3 -c 'import json,os,time;
try:
    d=json.load(open("live/telemetry.json"))
    print(d["meta"]["status"],"age",round(time.time()-os.path.getmtime("live/telemetry.json"),1),"s")
except Exception as e:
    print("unavailable",repr(e)[:60])
')"

    exit 0 ;;
*)  echo "usage: atlas_up.sh [up|down|status]" ; exit 2 ;;
esac
