#!/bin/bash
# :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

DANGER_TEMP=85  # degrees Celsius
CHECK_INTERVAL=30  # seconds

echo "?? GPU Health Monitor running: Shutdown at ${DANGER_TEMP}C"

while true; do
    # Exit if seed found
    [[ -f found_seeds.txt ]] && { echo "? Seed found-GPU monitor exiting."; exit 0; }

    for TEMP in $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); do
        if (( TEMP >= DANGER_TEMP )); then
            echo "?? GPU Overheat at ${TEMP}C! Shutting down miners:"
            killall seeds || true
            touch overheated.flag
            exit 0
        fi
    done

    sleep $CHECK_INTERVAL
done
