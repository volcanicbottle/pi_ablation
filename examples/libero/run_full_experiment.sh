#!/usr/bin/env bash
# One-shot driver for the full pi05_libero mask-ablation experiment on AutoDL (4090/24GB).
# Runs everything with `uv run` (no docker) so eval and training share one mechanism:
#
#   0. blindness check  -> assert masks truly blind the model
#   1. phase-1 eval      -> 4 arms x suites on the ORIGINAL pi05_libero ckpt
#   2. LoRA fine-tune    -> pi05_libero_mask_ft with modality dropout
#   3. phase-2 eval      -> same 4 arms x suites on the FINE-TUNED ckpt
#   4. archive           -> bundle all summaries/videos into results/<stamp>.tar.gz
#
# Design + interpretation: docs/superpowers/specs/2026-06-11-pi05-mask-ablation-design.md
# Step-by-step runbook:     docs/superpowers/specs/2026-06-11-autodl-runbook.md
#
# Usage (from repo root, on the AutoDL box):
#   examples/libero/run_full_experiment.sh
#
# Everything is overridable by env var, and each phase can be skipped to resume:
#   TRIALS=10 SUITES="libero_goal libero_spatial" examples/libero/run_full_experiment.sh
#   RUN_TRAIN=0 RUN_PHASE2=0 examples/libero/run_full_experiment.sh   # just phase-1
#   RUN_BLINDNESS=0 RUN_PHASE1=0 examples/libero/run_full_experiment.sh # resume at training
#
# Resume-friendly: an arm/suite whose summary.json already exists is skipped unless FORCE=1.
# "Save to my laptop": results live on the AutoDL box; the final tar.gz is what you scp down
# (the exact scp command is printed at the end).

set -euo pipefail

# ----------------------------- config (env-overridable) -----------------------------
ARMS="${ARMS:-none mask_v mask_l mask_vl}"            # ablation arms to evaluate
SUITES="${SUITES:-libero_goal libero_spatial}"       # LIBERO suites (round 1 = goal + spatial)
TRIALS="${TRIALS:-10}"                                # rollouts per task
SEED="${SEED:-7}"

BASE_CONFIG="${BASE_CONFIG:-pi05_libero}"
BASE_CKPT="${BASE_CKPT:-gs://openpi-assets/checkpoints/pi05_libero}"

FT_CONFIG="${FT_CONFIG:-pi05_libero_mask_ft}"
EXP_NAME="${EXP_NAME:-mask_ft_run1}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"                   # must match num_train_steps in config.py
FT_CKPT="${FT_CKPT:-checkpoints/${FT_CONFIG}/${EXP_NAME}/${TRAIN_STEPS}}"

PHASE1_DIR="${PHASE1_DIR:-data/libero}"               # original-ckpt eval outputs
PHASE2_DIR="${PHASE2_DIR:-data/libero_ft}"            # fine-tuned-ckpt eval outputs

PORT="${PORT:-8000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"               # seconds to wait for server (first run downloads ~12GB)

# phase toggles (1 = run, 0 = skip)
RUN_BLINDNESS="${RUN_BLINDNESS:-1}"
RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
FORCE="${FORCE:-0}"                                   # 1 = re-run arms even if summary.json exists

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ----------------------------- server helpers -----------------------------
SERVER_PID=""

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[exp] stopping policy server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

start_server() {  # $1=config  $2=checkpoint_dir
  local cfg="$1" dir="$2"
  echo "[exp] starting server: config=$cfg dir=$dir"
  uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config "$cfg" --policy.dir "$dir" &
  SERVER_PID=$!
  echo "[exp] waiting for ws://localhost:${PORT} (timeout ${SERVER_TIMEOUT}s)..."
  for ((i=1; i<=SERVER_TIMEOUT; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[exp] ERROR: server process exited before accepting connections" >&2
      exit 1
    fi
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
      echo "[exp] server up after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[exp] ERROR: server did not come up within ${SERVER_TIMEOUT}s" >&2
  exit 1
}

# ----------------------------- eval loop -----------------------------
run_eval_matrix() {  # $1=out_root
  local out_root="$1" ab suite out summary
  for ab in $ARMS; do
    for suite in $SUITES; do
      out="${out_root}/${ab}"
      summary="${out}/${suite}/summary.json"
      if [ "$FORCE" != "1" ] && [ -f "$summary" ]; then
        echo "[exp] skip (exists): $summary"
        continue
      fi
      echo "[exp] === eval ablation=$ab suite=$suite -> $out ==="
      mkdir -p "$out"
      uv run python examples/libero/main.py \
        --ablation "$ab" \
        --task-suite-name "$suite" \
        --num-trials-per-task "$TRIALS" \
        --seed "$SEED" \
        --video-out-path "$out" \
        2>&1 | tee "${out}/${suite}.log"
    done
  done
}

# ============================== 0. blindness check ==============================
if [ "$RUN_BLINDNESS" = "1" ]; then
  echo "[exp] ### 0/4 blindness check ###"
  uv run python scripts/check_ablation_blindness.py
fi

# ============================== 1. phase-1 eval (original ckpt) ==============================
if [ "$RUN_PHASE1" = "1" ]; then
  echo "[exp] ### 1/4 phase-1 eval on ${BASE_CONFIG} ###"
  start_server "$BASE_CONFIG" "$BASE_CKPT"
  run_eval_matrix "$PHASE1_DIR"
  stop_server
fi

# ============================== 2. LoRA fine-tune ==============================
if [ "$RUN_TRAIN" = "1" ]; then
  echo "[exp] ### 2/4 LoRA fine-tune ${FT_CONFIG} exp=${EXP_NAME} steps=${TRAIN_STEPS} ###"
  if [ "$FORCE" != "1" ] && [ -d "$FT_CKPT" ]; then
    echo "[exp] skip training (checkpoint exists): $FT_CKPT"
  else
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      uv run scripts/train.py "$FT_CONFIG" --exp-name "$EXP_NAME" --overwrite
  fi
fi

# ============================== 3. phase-2 eval (fine-tuned ckpt) ==============================
if [ "$RUN_PHASE2" = "1" ]; then
  echo "[exp] ### 3/4 phase-2 eval on ${FT_CONFIG} (${FT_CKPT}) ###"
  if [ ! -d "$FT_CKPT" ]; then
    echo "[exp] ERROR: fine-tuned checkpoint not found: $FT_CKPT" >&2
    echo "[exp]        (set FT_CKPT=... to the actual step dir, or run training first)" >&2
    exit 1
  fi
  start_server "$FT_CONFIG" "$FT_CKPT"
  run_eval_matrix "$PHASE2_DIR"
  stop_server
fi

# ============================== 4. archive results ==============================
echo "[exp] ### 4/4 archive ###"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)}"
RESULTS_DIR="results"
ARCHIVE="${RESULTS_DIR}/mask_ablation_${STAMP}.tar.gz"
mkdir -p "$RESULTS_DIR"

# collect every summary.json into one combined file for quick reading
COMBINED="${RESULTS_DIR}/all_summaries_${STAMP}.txt"
: > "$COMBINED"
find "$PHASE1_DIR" "$PHASE2_DIR" -name summary.json 2>/dev/null | sort | while read -r f; do
  echo "===== $f =====" >> "$COMBINED"
  cat "$f" >> "$COMBINED"
  echo >> "$COMBINED"
done

# tar summaries + logs + videos (exclude nothing by default; set NO_VIDEOS=1 to slim it down)
TAR_TARGETS=()
[ -d "$PHASE1_DIR" ] && TAR_TARGETS+=("$PHASE1_DIR")
[ -d "$PHASE2_DIR" ] && TAR_TARGETS+=("$PHASE2_DIR")
TAR_TARGETS+=("$COMBINED")
if [ "${NO_VIDEOS:-0}" = "1" ]; then
  tar --exclude='*.mp4' -czf "$ARCHIVE" "${TAR_TARGETS[@]}"
else
  tar -czf "$ARCHIVE" "${TAR_TARGETS[@]}"
fi

echo "[exp] DONE."
echo "[exp] combined summaries: $COMBINED"
echo "[exp] archive:            $ARCHIVE"
echo
echo "[exp] pull it to your laptop with (run this on YOUR machine, fix host/port):"
echo "      scp -P <autodl_ssh_port> root@<autodl_host>:${REPO_ROOT}/${ARCHIVE} ."
