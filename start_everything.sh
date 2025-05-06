#!/bin/bash
#
# start_everything.sh
# Prompts, then launches health monitor, watchdog, miners with resume logic,
# and a status loop—all in one terminal.

set -e

### 0) Cleanup old processes
echo "🚀 Cleaning up old processes…"
killall seeds start_everything.sh gpu_health_monitor.sh launch_watchdog.sh 2>/dev/null || true
sleep 1

### 1) Prompt the user
read -p "Enter known seed words (space-separated, or leave blank if none): " KNOWN_SEED_WORDS
read -p "Enter target Ethereum address (hex, no 0x; leave blank for zero mode): " TARGET_ADDRESS
read -p "How many workers (GPUs/threads)? " TOTAL_WORKERS

export KNOWN_SEED_WORDS
export TARGET_ADDRESS
export TOTAL_WORKERS

### 2) Generate helper scripts

# GPU health monitor (exits on found or overheat)
cat << 'EOF' > gpu_health_monitor.sh
#!/bin/bash
DANGER_TEMP=85
CHECK_INTERVAL=30
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  for t in $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); do
    (( t >= DANGER_TEMP )) && {
      echo "🔥 GPU overheat (${t}C)! Shutting down miners."
      killall seeds || true
      touch overheated.flag
      exit 0
    }
  done
  sleep $CHECK_INTERVAL
done
EOF
chmod +x gpu_health_monitor.sh

# Watchdog (crash & slow detection, with resume)
cat << 'EOF' > launch_watchdog.sh
#!/bin/bash
THRESHOLD=50000000  # 50M seeds/sec
while true; do
  [[ -f found_seeds.txt ]] && exit 0
  sleep 60
  for ((i=0; i<TOTAL_WORKERS; i++)); do
    [[ -f found_seeds.txt ]] && exit 0

    # Crash detection:
    if ! pgrep -f "seeds.*--workers $TOTAL_WORKERS.*--resume-from worker$i.offset" > /dev/null; then
      echo "💥 Worker $i crashed; resuming from offset…" | tee -a restart.log
      OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
      CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
        --workers $TOTAL_WORKERS \
        --known "$KNOWN_SEED_WORDS" \
        --address "$TARGET_ADDRESS" \
        --wordlist words.txt \
        --resume-from "$OFFSET" \
        > miner$i.log 2>&1 &
      continue
    fi

    # Slow miner detection:
    speed=$(tail -n20 miner$i.log 2>/dev/null \
      | grep -oP 'Speed: \K[0-9,]+' | tr -d ',')
    if [[ -n "$speed" && $speed -lt $THRESHOLD ]]; then
      echo "⚠️ Worker $i slow ($speed); restarting from offset…" | tee -a restart.log
      pkill -f "seeds.*--workers $TOTAL_WORKERS" || true
      sleep 2
      OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
      CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
        --workers $TOTAL_WORKERS \
        --known "$KNOWN_SEED_WORDS" \
        --address "$TARGET_ADDRESS" \
        --wordlist words.txt \
        --resume-from "$OFFSET" \
        > miner$i.log 2>&1 &
    fi
  done
done
EOF
chmod +x launch_watchdog.sh

### 3) Launch helper scripts
echo "🛡️ Starting GPU Health Monitor..."
./gpu_health_monitor.sh &
echo "🛡️ Starting Watchdog..."
./launch_watchdog.sh &

### 4) Spawn miners with resume offsets
echo "🔥 Spawning $TOTAL_WORKERS miners…"
for ((i=0; i<TOTAL_WORKERS; i++)); do
  # read last offset or start at 0
  OFFSET=$(cat worker$i.offset 2>/dev/null || echo 0)
  CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds \
    --workers $TOTAL_WORKERS \
    --known "$KNOWN_SEED_WORDS" \
    --address "$TARGET_ADDRESS" \
    --wordlist words.txt \
    --resume-from "$OFFSET" \
    > miner$i.log 2>&1 &
done

### 5) Status loop (foreground)
UNKNOWN=$((12 - $(echo $KNOWN_SEED_WORDS | wc -w)))
TOTAL_SPACE=$(printf "%.0f" "$(echo "2048^$UNKNOWN" | bc)")
BATCH=$(echo "$TOTAL_SPACE / $TOTAL_WORKERS" | bc)

echo
echo "✅ All miners running. Showing status every 60 s (Ctrl-C to exit):"
echo

while true; do
  [[ -f found_seeds.txt ]] && { echo "✅ Seed found—exiting status loop."; exit 0; }
  echo "----- Status @ $(date) -----"
  for ((i=0; i<TOTAL_WORKERS; i++)); do
    tested=$(grep -oP 'Seeds tested: \K[0-9]+' miner$i.log | tail -1)
    tested=${tested:-0}
    rem=$(echo "$BATCH - $tested" | bc)
    speed=$(grep -oP 'Speed: \K[0-9,]+' miner$i.log | tail -1 | tr -d ',')
    speed=${speed:-N/A}
    printf "Worker %d: tested %'d/%'d (remaining %'d), %s seeds/sec\n" \
      "$i" "$tested" "$BATCH" "$rem" "$speed"
  done
  echo
  sleep 60
done

