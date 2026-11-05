#!/bin/sh
# Continue the runs frozen by ./pause_runs.sh. Run this once the network is back.
for pid in $(pgrep -f "python -m framework.run --backend"); do
  case "$(ps -o comm= -p "$pid")" in python*|uv) ;; *) continue ;; esac
  pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
  [ -n "$pg" ] && /bin/kill -s CONT -- "-$pg" && echo "resumed run pid $pid (group $pg)"
done
for pid in $(pgrep -f "python -m framework.run --backend"); do
  case "$(ps -o comm= -p "$pid")" in python) ps -o pid=,stat=,etime=,args= -p "$pid" | cut -c1-110 ;; esac
done
