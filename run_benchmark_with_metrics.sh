#!/bin/bash

# Wrapper script to run benchmarks with server metrics collection

# Configuration
METRICS_URL="${METRICS_URL:-http://localhost:8081/metrics}"
COLLECTION_INTERVAL="${COLLECTION_INTERVAL:-30}"
RUN_LABEL="${RUN_LABEL:-benchmark_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-./results}"

# Check if benchmark script exists
if [ ! -f "./run_benchmark_sweep.sh" ]; then
  echo "Error: run_benchmark_sweep.sh not found"
  exit 1
fi

# Check if metrics collector exists
if [ ! -f "./vllm_metrics_collector.py" ]; then
  echo "Error: vllm_metrics_collector.py not found"
  exit 1
fi

# Create results directory
mkdir -p "$RESULT_DIR"

# Output files
METRICS_CSV="${RESULT_DIR}/server_metrics_${RUN_LABEL}.csv"
METRICS_LOG="${RESULT_DIR}/metrics_collector_${RUN_LABEL}.log"
SWEEP_LOG="${RESULT_DIR}/sweep_${RUN_LABEL}.log"

echo "========================================="
echo "Starting Benchmark with Metrics Collection"
echo "========================================="
echo "Metrics URL: $METRICS_URL"
echo "Collection Interval: ${COLLECTION_INTERVAL}s"
echo "Run Label: $RUN_LABEL"
echo "Metrics CSV: $METRICS_CSV"
echo "========================================="
echo ""

# Start metrics collector in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting metrics collector..."
python vllm_metrics_collector.py \
  --url "$METRICS_URL" \
  --output "$METRICS_CSV" \
  --interval "$COLLECTION_INTERVAL" \
  > "$METRICS_LOG" 2>&1 &

COLLECTOR_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metrics collector started (PID: $COLLECTOR_PID)"
echo "$COLLECTOR_PID" > collector.pid

# Give collector a moment to start
sleep 2

# Function to cleanup on exit
cleanup() {
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping metrics collector (PID: $COLLECTOR_PID)..."
  kill $COLLECTOR_PID 2>/dev/null
  wait $COLLECTOR_PID 2>/dev/null
  rm -f collector.pid
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metrics collector stopped"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Run benchmark sweep
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting benchmark sweep..."
echo ""

./run_benchmark_sweep.sh | tee "$SWEEP_LOG"

BENCHMARK_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmark sweep completed (exit code: $BENCHMARK_EXIT_CODE)"

# Wait a bit for final metrics to be collected
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 30s for final metrics collection..."
sleep 30

# Cleanup will be called automatically by trap

echo ""
echo "========================================="
echo "Benchmark Complete"
echo "========================================="
echo "Benchmark log: $SWEEP_LOG"
echo "Metrics CSV: $METRICS_CSV"
echo "Metrics log: $METRICS_LOG"
echo ""
echo "To upload to Scuba, run in fbcode:"
echo "  python upload_to_scuba_example.py \\"
echo "    --csv_file=$METRICS_CSV \\"
echo "    --scuba_table=YOUR_TABLE \\"
echo "    --upload_label=$RUN_LABEL"
echo "========================================="

exit $BENCHMARK_EXIT_CODE
