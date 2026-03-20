#!/usr/bin/env bash
# kv3d benchmark runner
# Runs the bench_driver binary and optionally plots results.
set -euo pipefail

BENCH_BIN="${1:-./build/tests/load/kv3d_bench}"
REQUESTS="${REQUESTS:-1000}"
MAX_TOKENS="${MAX_TOKENS:-64}"
SHARED_RATIO="${SHARED_RATIO:-0.8}"
OUTPUT_CSV="${OUTPUT_CSV:-bench_results/run_$(date +%Y%m%d_%H%M%S).csv}"

if [ ! -x "$BENCH_BIN" ]; then
    echo "Error: bench binary not found at $BENCH_BIN"
    echo "Build first: cmake --build build -j && ctest --test-dir build"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "Running kv3d benchmark..."
echo "  binary       : $BENCH_BIN"
echo "  requests     : $REQUESTS"
echo "  max_tokens   : $MAX_TOKENS"
echo "  shared_ratio : $SHARED_RATIO"
echo "  output       : $OUTPUT_CSV"
echo ""

"$BENCH_BIN" \
    --requests "$REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --shared-ratio "$SHARED_RATIO" \
    --output "$OUTPUT_CSV"

echo ""
echo "Benchmark complete. Results saved to: $OUTPUT_CSV"

# Print the CSV if it exists
if [ -f "$OUTPUT_CSV" ]; then
    echo ""
    cat "$OUTPUT_CSV"
fi
