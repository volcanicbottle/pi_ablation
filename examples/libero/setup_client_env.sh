#!/usr/bin/env bash
# Build the LIBERO client venv (python 3.8) for running examples/libero/main.py.
# Separate from the openpi server env on purpose (conflicting deps). The client
# only runs the MuJoCo simulator + websocket; it does NOT use the GPU, so torch
# is installed as the plain CPU build (the +cu113 pin is stripped) to avoid the
# slow pytorch.org download. See docs/superpowers/specs/2026-06-11-autodl-runbook.md
# and the deployment notes.
#
# Usage (from repo root, on the AutoDL box):
#   bash examples/libero/setup_client_env.sh
set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
echo "[client] repo: $REPO"

# 1. LIBERO submodule must be present (provides third_party/libero).
if [ ! -f third_party/libero/requirements.txt ]; then
  echo "[client] libero submodule missing -> fetching (needs github / academic accel)..."
  git submodule update --init third_party/libero
fi

# 2. Use the plain CPU torch: strip the +cu113 local version so the wheel comes
#    from the fast domestic mirror. Idempotent (no-op if already stripped).
sed -i 's/+cu113//g' examples/libero/requirements.txt

# 3. Create the python-3.8 venv for the client.
uv venv --python 3.8 examples/libero/.venv

# 4. Install client deps from the Tsinghua mirror (fast, no pytorch.org).
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
unset UV_EXTRA_INDEX_URL UV_INDEX_STRATEGY || true
uv pip sync --python examples/libero/.venv \
  examples/libero/requirements.txt third_party/libero/requirements.txt

# 5. Editable local packages (after sync, which would otherwise remove them).
uv pip install --python examples/libero/.venv -e packages/openpi-client
uv pip install --python examples/libero/.venv -e third_party/libero

# 6. Verify.
echo "[client] verifying imports..."
examples/libero/.venv/bin/python -c "import libero, robosuite, openpi_client; print('CLIENT OK')"
echo "[client] DONE. Run evals with: examples/libero/.venv/bin/python examples/libero/main.py ..."
