#!/bin/bash

# Benchmark script for vLLM server testing
# Tests various ISL/OSL and concurrency configurations

# Configuration
export HF_HUB_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"
TOKENIZER="${TOKENIZER:-/home/mrasquinha/checkpoints/gpt-oss-120b}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8081}"
NUM_PROMPTS="${NUM_PROMPTS:-20000}"
RESULT_DIR="${RESULT_DIR:-./results}"
BACKEND="${BACKEND:-vllm}"

# ISL/OSL configurations (Input Sequence Length / Output Sequence Length)
declare -a ISL_OSL_CONFIGS=(
  "1024:1024"
  "1024:8192"
  "8192:1024"
)

# Concurrency options
declare -a CONCURRENCY_OPTIONS=(128 64 32 16 8)

# Create results directory if it doesn't exist
mkdir -p "$RESULT_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULT_DIR}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "Starting benchmark suite at $(date)"
echo "Results will be saved to: $RUN_DIR"
echo "Model: $MODEL_NAME"
echo "Tokenizer: $TOKENIZER"
echo "Server: http://${HOST}:${PORT}"
echo "Number of prompts per test: $NUM_PROMPTS"
echo "================================================"

# Loop through ISL/OSL configurations
for config in "${ISL_OSL_CONFIGS[@]}"; do
  IFS=':' read -r input_len output_len <<< "$config"

  echo ""
  echo "Testing ISL/OSL: ${input_len}/${output_len}"
  echo "------------------------------------------------"

  # Loop through concurrency options
  for concurrency in "${CONCURRENCY_OPTIONS[@]}"; do
    echo "  Running with concurrency: $concurrency"

    # Generate result and log filenames
    base_name="isl${input_len}_osl${output_len}_c${concurrency}"
    result_file="${RUN_DIR}/${base_name}.json"
    log_file="${RUN_DIR}/${base_name}.log"

    # Write start timestamp to log file
    {
      echo "========================================"
      echo "Benchmark Configuration"
      echo "========================================"
      echo "ISL: $input_len"
      echo "OSL: $output_len"
      echo "Concurrency: $concurrency"
      echo "Model: $MODEL_NAME"
      echo "Tokenizer: $TOKENIZER"
      echo "Host: $HOST:$PORT"
      echo "Num Prompts: $NUM_PROMPTS"
      echo "========================================"
      echo "START TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "START TIMESTAMP: $(date +%s)"
      echo "========================================"
      echo ""
    } > "$log_file"

    # Run benchmark and tee output to both console and log file
    python benchmark_serving.py \
      --backend "$BACKEND" \
      --model "$MODEL_NAME" \
      --tokenizer "$TOKENIZER" \
      --host "$HOST" \
      --port "$PORT" \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$concurrency" \
      --request-rate inf \
      --save-result \
      --result-filename "$result_file" \
      --seed 42 2>&1 | tee -a "$log_file"

    # Capture exit code from the python command (not tee)
    exit_code=${PIPESTATUS[0]}

    # Write end timestamp to log file
    {
      echo ""
      echo "========================================"
      echo "END TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "END TIMESTAMP: $(date +%s)"
      echo "EXIT CODE: $exit_code"
      echo "========================================"
    } >> "$log_file"

    # Check if benchmark completed successfully
    if [ $exit_code -eq 0 ]; then
      echo "    ✓ Completed successfully - log: $log_file"
    else
      echo "    ✗ Failed (exit code: $exit_code) - log: $log_file"
    fi
  done
done

echo ""
echo "================================================"
echo "Benchmark suite completed at $(date)"
echo "Results saved to: $RUN_DIR"

# Generate summary report
echo ""
echo "Generating summary report..."
python - <<EOF
import json
import os
from pathlib import Path

result_dir = Path("$RUN_DIR")
results = []

for json_file in sorted(result_dir.glob("*.json")):
  with open(json_file) as f:
    data = json.load(f)
    results.append({
      'config': json_file.stem,
      'completed': data.get('completed', 0),
      'request_throughput': data.get('request_throughput', 0),
      'output_throughput': data.get('output_throughput', 0),
      'mean_ttft_ms': data.get('mean_ttft_ms', 0),
      'mean_tpot_ms': data.get('mean_tpot_ms', 0),
    })

print("\n=== Summary Report ===")
print(f"{'Configuration':<30} {'Req/s':<10} {'Tok/s':<10} "
      f"{'TTFT(ms)':<12} {'TPOT(ms)':<12}")
print("-" * 80)

for r in results:
  print(f"{r['config']:<30} {r['request_throughput']:<10.2f} "
        f"{r['output_throughput']:<10.2f} {r['mean_ttft_ms']:<12.2f} "
        f"{r['mean_tpot_ms']:<12.2f}")

print(f"\nDetailed results: {result_dir}")
EOF

echo ""
echo "Done!"
