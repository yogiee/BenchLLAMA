#!/usr/bin/env bash
# BenchLLAMA launcher.
#   ./bench.sh [command] [flags]            → web UI (default). No command → browser selection screen.
#   ./bench.sh [command] [flags] --console  → plain-text terminal (headless / SSH / quick glance).
#
# Commands: standard · ladder · aptitude · batteries · all · vision · embedding · longctx · imagegen · confab · update · export
#   all flags: --with-elastic (append F-elastic) · --with-imagegen (append Battery I) — one unattended run does everything
# Web: binds 0.0.0.0 by default (reachable on the LAN for monitoring from phone/iPad).
#   Control (Start/Stop) is allowed from the host machine only; LAN clients are read-only.
#   --allow-control  → let LAN clients control too   ·   --host 127.0.0.1 → localhost-only   ·   --port N
DIR="$(cd "$(dirname "$0")" && pwd)"

args=(); console=0; report=0; battery=""; check_runtime=0
prev=""
for a in "$@"; do
  if [ "$a" = "--console" ]; then console=1;
  elif [ "$a" = "--resume-report" ]; then report=1;
  else args+=("$a"); fi
  if [ "$a" = "--check-runtime" ]; then check_runtime=1; fi
  if [ "$prev" = "--battery" ]; then battery="$a"; fi
  prev="$a"
done

# --resume-report: dry content-addressed resume plan (which models would re-run + WHY). No benchmark.
if [ "$report" -eq 1 ]; then
  rflags=()
  [ -n "$battery" ] && rflags+=(--battery "$battery")
  [ "$check_runtime" -eq 1 ] && rflags+=(--check-runtime)
  exec python3 "$DIR/resume.py" "${rflags[@]}"
fi

if [ "$console" -eq 1 ]; then
  exec python3 "$DIR/orchestrator.py" "${args[@]}"
fi
exec python3 "$DIR/webserver.py" "${args[@]}"
