##!/bin/bash

TOTAL_WORKERS=${TOTAL_WORKERS:-4}  # Default to 4 if not set

echo "🚀 Launching $TOTAL_WORKERS miners..."

for ((i=0; i<TOTAL_WORKERS; i++)); do
    echo "Launching Miner $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i nohup ./target/release/seeds $i > miner$i.log 2>&1 &
done

echo "✅ All $TOTAL_WORKERS miners launched!"
echo "📜 Use: tail -f miner*.log"
