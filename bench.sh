#!/usr/bin/env bash
# BenchLLAMA — delegate to bench_ui.py (unified TUI launcher)
# Install once: pip install textual
exec python3 "$(cd "$(dirname "$0")" && pwd)/bench_ui.py" "$@"
