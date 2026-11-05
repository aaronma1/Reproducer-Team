#!/bin/sh
# Freeze every live reproduction run (and its children: shells, ssh) before going offline.
# GPU jobs keep running on the GPU box. Undo with ./unpause_runs.sh.
for pid in $(pgrep -f "python -m framework.run --backend"); do
  case "$(ps -o comm= -p "$pid")" in python*|uv) ;; *) continue ;; esac
  pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
  [ -n "$pg" ] && /bin/kill -s STOP -- "-$pg" && echo "paused run pid $pid (group $pg)"
done
