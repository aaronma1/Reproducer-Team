#!/bin/sh
# Start one detached live run: ./launch.sh <args-name> [extra framework.run flags...]
# Logs go to logs/<args-name>-<time>.log; the run survives the terminal closing.
name=$1; shift
log=logs/$name-$(date +%Y%m%d-%H%M%S).log
mkdir -p logs
setsid nohup uv run python -m framework.run --backend live --args args/$name.json "$@" > "$log" 2>&1 < /dev/null &
echo "$log"
