#!/usr/bin/env bash
# BenchLLAMA — unified benchmark runner
# Opens the monitor in a new Terminal window, then runs the chosen suite.

REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Usage ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'

Usage:  bench.sh <command> [flags]

Commands:
  standard             Standard suite — 13 tests, 5 dimensions
  ladder               num_ctx characterisation — run before aptitude
  aptitude             Single aptitude battery (default: Battery B)
  batteries            All four aptitude batteries: A → B → C → D
  all                  Full pipeline: standard → ladder → A → B → C → D
  update               Sync models.json from local Ollama API

Flags (passed through to the Python script):
  --fast               Skip cool-down (informal results)
  --role <r>           Filter by role: router | worker  (ladder / aptitude)
  --battery <X>        Aptitude battery: A | B | C | D  (default B)
  --models m1 [m2...]  Specific models (standard / ladder positional names)
  --system-prompt <p>  Custom worker system prompt path
  --ollama <url>       Remote Ollama  (default: http://localhost:11434)
  --no-monitor         Skip opening the monitor window
  --help               Show this message

Examples:
  bench.sh standard
  bench.sh standard --fast
  bench.sh ladder --role worker
  bench.sh aptitude --battery B --system-prompt ~/alice.md
  bench.sh batteries
  bench.sh all
  bench.sh update --dry-run

EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────

CMD=""
NO_MON=""
FAST=""
PASS=()

for arg in "$@"; do
    case "$arg" in
        standard|ladder|aptitude|batteries|all|update)
            CMD="$arg" ;;
        --no-monitor)
            NO_MON=1 ;;
        --fast)
            FAST="1"
            PASS+=("$arg") ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            PASS+=("$arg") ;;
    esac
done

if [[ -z "$CMD" ]]; then
    usage
    exit 1
fi

# ── Monitor ───────────────────────────────────────────────────────────────────

if [[ -z "$NO_MON" ]]; then
    MON_CMD="python3 '$REPO/monitor.py'"
    [[ -n "$FAST" ]] && MON_CMD+=" --fast"
    osascript -e "tell application \"Terminal\"
        set w to do script \"$MON_CMD\"
        set bounds of front window to {50, 50, 700, 520}
    end tell" 2>/dev/null
    sleep 0.5
fi

# ── Runners ───────────────────────────────────────────────────────────────────

_standard() { python3 "$REPO/runner.py"          "${PASS[@]}"; }
_ladder()   { python3 "$REPO/ctx_ladder.py"       "${PASS[@]}"; }
_aptitude() { python3 "$REPO/aptitude.py"         "${PASS[@]}"; }
_update()   { python3 "$REPO/update_registry.py"  "${PASS[@]}"; }

_pause() {
    echo
    echo "━━━ $1 — starting in 10s  (Ctrl+C to abort) ━━━"
    sleep 10
}

_pause_short() {
    echo
    echo "  ── $1 — starting in 5s  (Ctrl+C to abort) ──"
    sleep 5
}

_batteries() {
    echo "━━━ Battery A ━━━"
    python3 "$REPO/aptitude.py" --battery A "${PASS[@]}" || return 1
    _pause_short "Battery B"
    echo "━━━ Battery B ━━━"
    python3 "$REPO/aptitude.py" --battery B "${PASS[@]}" || return 1
    _pause_short "Battery C"
    echo "━━━ Battery C ━━━"
    python3 "$REPO/aptitude.py" --battery C "${PASS[@]}" || return 1
    _pause_short "Battery D"
    echo "━━━ Battery D ━━━"
    python3 "$REPO/aptitude.py" --battery D "${PASS[@]}" || return 1
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$CMD" in
    standard)  _standard ;;
    ladder)    _ladder ;;
    aptitude)  _aptitude ;;
    batteries) _batteries ;;
    update)    _update ;;
    all)
        echo "━━━ Phase 1 / 3: Standard Suite ━━━"
        _standard || exit 1
        _pause "Phase 2 / 3: ctx Ladder"
        echo "━━━ Phase 2 / 3: ctx Ladder ━━━"
        _ladder  || exit 1
        _pause "Phase 3 / 3: Aptitude Batteries A → B → C → D"
        _batteries || exit 1
        echo
        echo "━━━ Full pipeline complete. ━━━"
        ;;
esac
