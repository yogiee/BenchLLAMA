#!/usr/bin/env bash
# BenchLLAMA launcher.
#   ./bench.sh [command] [flags]            → web UI (default). No command → browser selection screen.
#   ./bench.sh [command] [flags] --console  → plain-text terminal (headless / SSH / quick glance).
#
# Commands: standard · ladder · aptitude · batteries · all · vision · embedding · longctx · update
# Web: binds 0.0.0.0 by default (reachable on the LAN for monitoring from phone/iPad).
#   Control (Start/Stop) is allowed from the host machine only; LAN clients are read-only.
#   --allow-control  → let LAN clients control too   ·   --host 127.0.0.1 → localhost-only   ·   --port N
DIR="$(cd "$(dirname "$0")" && pwd)"

args=(); console=0
for a in "$@"; do
  if [ "$a" = "--console" ]; then console=1; else args+=("$a"); fi
done

if [ "$console" -eq 1 ]; then
  exec python3 "$DIR/orchestrator.py" "${args[@]}"
fi
exec python3 "$DIR/webserver.py" "${args[@]}"
