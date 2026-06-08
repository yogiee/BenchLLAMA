#!/bin/bash
# Launch monitor in a new Terminal window, then run the benchmark here.
#
# Usage:
#   ./run.sh                            # standard suite
#   ./run.sh --aptitude                 # Battery B aptitude
#   ./run.sh --aptitude --battery A     # specific battery
#   ./run.sh --ctx-ladder               # num_ctx characterisation pass
#   ./run.sh --ctx-ladder --role router # ladder for router models only
#   ./run.sh --fast                     # skip cool-down (any suite)
#   ./run.sh --system-prompt ~/my.md    # custom worker prompt
#   ./run.sh qwen3.5:4b-mlx gemma4      # standard suite, specific models

REPO="$(cd "$(dirname "$0")" && pwd)"
ARGS="$*"
MONITOR_ARGS=""
[[ "$ARGS" == *"--fast"* ]] && MONITOR_ARGS="--fast"

if [[ "$ARGS" == *"--aptitude"* ]]; then
    SCRIPT="$REPO/aptitude.py"
    ARGS="${ARGS/--aptitude/}"
elif [[ "$ARGS" == *"--ctx-ladder"* ]]; then
    SCRIPT="$REPO/ctx_ladder.py"
    ARGS="${ARGS/--ctx-ladder/}"
else
    SCRIPT="$REPO/runner.py"
fi

osascript -e "tell application \"Terminal\"
  set w to do script \"python3 '$REPO/monitor.py' $MONITOR_ARGS\"
  set bounds of front window to {50, 50, 700, 520}
end tell" 2>/dev/null

sleep 0.5

python3 "$SCRIPT" $ARGS
