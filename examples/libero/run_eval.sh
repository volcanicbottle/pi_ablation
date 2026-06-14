#!/usr/bin/env bash
# Drive the LIBERO eval client against an ALREADY-RUNNING policy server.
# (Server and client live in separate envs; this only runs the client loop.)
# Ablation is applied per-request server-side, so one warm server serves all arms.
#
# Usage (from repo root, server already up on :8000):
#   bash examples/libero/run_eval.sh                  # 4 arms x libero_goal x 10 trials
#   TRIALS=2 ARMS=none bash examples/libero/run_eval.sh   # quick smoke test (~2 min)
#   ARMS="none mask_v mask_l mask_vl" SUITES="libero_goal libero_spatial" \
#     TRIALS=10 bash examples/libero/run_eval.sh
#
# Outputs: data/libero/<ablation>/<suite>/summary.json (+ videos + log).
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

LIBERO_PY="${LIBERO_PY:-examples/libero/.venv/bin/python}"
ARMS="${ARMS:-none mask_v mask_l mask_vl}"
SUITES="${SUITES:-libero_goal}"
TRIALS="${TRIALS:-10}"
OUT_ROOT="${OUT_ROOT:-data/libero}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO/third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [ ! -x "$LIBERO_PY" ]; then
  echo "[eval] ERROR: client interpreter not found: $LIBERO_PY" >&2
  echo "[eval] build it first: bash examples/libero/setup_client_env.sh" >&2
  exit 1
fi

for ab in $ARMS; do
  for suite in $SUITES; do
    out="$OUT_ROOT/$ab"
    echo "[eval] === ablation=$ab suite=$suite trials=$TRIALS -> $out ==="
    mkdir -p "$out"
    "$LIBERO_PY" examples/libero/main.py \
      --ablation "$ab" \
      --task-suite-name "$suite" \
      --num-trials-per-task "$TRIALS" \
      --video-out-path "$out" \
      2>&1 | tee "$out/$suite.log"
  done
done

echo "[eval] DONE. Summaries:"
find "$OUT_ROOT" -name summary.json -exec echo "  {}" \;
