#!/usr/bin/env bash
# Run ablation arms across multiple LIBERO suites, sharing one warm policy server.
# Ablation is applied per-request (attention masks server-side), so a single
# server instance serves every arm.
#
# Usage (from repo root):
#   examples/libero/run_ablation_matrix.sh <ablation[,ablation...]> [suite ...]
#
# Examples:
#   examples/libero/run_ablation_matrix.sh none
#   examples/libero/run_ablation_matrix.sh none,mask_v,mask_l,mask_vl libero_goal libero_spatial
#   TRIALS=50 examples/libero/run_ablation_matrix.sh mask_v             # all 4 default suites
#
# Outputs land in data/libero/<ablation>/<suite>/ (videos + summary.json + failures/).

set -euo pipefail

IFS=',' read -r -a ABLATIONS <<< "${1:-none}"
shift || true
SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

COMPOSE=(docker compose -f examples/libero/compose.yml)

cleanup() {
  echo "[runner] tearing down compose stack..."
  "${COMPOSE[@]}" down --remove-orphans || true
}
trap cleanup EXIT

echo "[runner] starting policy server (arms: ${ABLATIONS[*]})..."
SERVER_ARGS="--env LIBERO" "${COMPOSE[@]}" up -d --build openpi_server

echo "[runner] waiting for ws://localhost:8000 to accept connections..."
SERVER_CID="$("${COMPOSE[@]}" ps -q openpi_server)"
for i in $(seq 1 900); do
  if (echo > /dev/tcp/127.0.0.1/8000) >/dev/null 2>&1; then
    echo "[runner] server up after ${i}s"
    break
  fi
  if [ -n "$SERVER_CID" ] && [ "$(docker inspect -f '{{.State.Running}}' "$SERVER_CID" 2>/dev/null)" != "true" ]; then
    echo "[runner] server container exited early — logs:"
    "${COMPOSE[@]}" logs openpi_server | tail -50
    exit 1
  fi
  sleep 1
done

for ABLATION in "${ABLATIONS[@]}"; do
  OUT_ROOT="data/libero/${ABLATION}"
  mkdir -p "$OUT_ROOT"
  for SUITE in "${SUITES[@]}"; do
    echo "[runner] === ablation=$ABLATION suite=$SUITE ==="
    CLIENT_ARGS="--ablation ${ABLATION} --task-suite-name ${SUITE} --num-trials-per-task ${TRIALS:-10} --video-out-path /app/${OUT_ROOT}" \
      "${COMPOSE[@]}" run --rm --no-deps runtime \
        2>&1 | tee "${OUT_ROOT}/${SUITE}.log"
  done
done

echo "[runner] DONE. Summaries:"
find data/libero -name summary.json -exec echo "  {}" \;
