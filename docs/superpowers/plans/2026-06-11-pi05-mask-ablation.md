# π₀.₅ Mask Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attention-mask based input ablation (V/L) for the pi05_libero checkpoint, plus a modality-dropout LoRA fine-tuning config, per spec `docs/superpowers/specs/2026-06-11-pi05-mask-ablation-design.md`.

**Architecture:** A generic `ApplyAblationMask` transform (no-op without an `ablation` key) is appended to the LIBERO model_transforms so the same masking code serves both inference (key sent per-request by the eval client) and training (key written by a train-only `SampleModalityDropout` transform in the repack stage). `LiberoInputs` forwards the key through its dict rebuild.

**Tech Stack:** Python, numpy, openpi transforms pipeline, pytest, uv.

**Context notes for the implementer:**
- pi05_libero takes ONLY vision + language. State is ignored by the model (`discrete_state_input=False`, no state token in pi05 suffix). Prompt is pi0-format: BOS + instruction + "\n". Masking L = whole `tokenized_prompt_mask` to False.
- Transform pipeline at inference (`policy_config.py:77-83`): repack → InjectDefaultPrompt → data_transforms (LiberoInputs) → Normalize → model_transforms (ResizeImages, TokenizePrompt, PadStatesAndActions, [ours last]).
- At training, `repack_transforms` run only on dataset data, never at inference (`config.py:293` comment) — that's why the dropout sampler lives there.
- `transforms.Group.push(inputs=[...])` appends (see delta transform usage at `config.py:340`).
- Local machine has 8GB GPU: unit tests only run on CPU; anything needing the model runs on AutoDL (4090). Tasks 1–7 are local; Task 8 is the AutoDL runbook.

---

### Task 0: Environment setup

**Files:** none (environment only)

- [ ] **Step 0.1: Init submodules and create venv**

```bash
cd /home/peng/Desktop/Computer/pi0_ablation/my_pi0_ablation
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Expected: `.venv` created, no resolution errors.

- [ ] **Step 0.2: Baseline — run existing transforms tests**

Run: `uv run pytest src/openpi/transforms_test.py -q`
Expected: all pass (establishes clean baseline before our changes).

### Task 1: `ApplyAblationMask` transform

**Files:**
- Modify: `src/openpi/transforms.py` (add class after `TokenizePrompt`, ~line 267)
- Test: `src/openpi/transforms_test.py`

- [ ] **Step 1.1: Write failing tests**

Append to `src/openpi/transforms_test.py`:

```python
def _ablation_data():
    return {
        "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        "tokenized_prompt": np.arange(10),
        "tokenized_prompt_mask": np.array([True] * 6 + [False] * 4),
        "state": np.zeros(8),
    }


def test_apply_ablation_mask_noop_without_key():
    data = _ablation_data()
    out = _transforms.ApplyAblationMask()(dict(data))
    assert out["image_mask"]["base_0_rgb"] == np.True_
    assert out["tokenized_prompt_mask"].sum() == 6


def test_apply_ablation_mask_none():
    out = _transforms.ApplyAblationMask()({**_ablation_data(), "ablation": "none"})
    assert "ablation" not in out
    assert out["image_mask"]["base_0_rgb"] == np.True_
    assert out["tokenized_prompt_mask"].sum() == 6


def test_apply_ablation_mask_v():
    out = _transforms.ApplyAblationMask()({**_ablation_data(), "ablation": "mask_v"})
    assert "ablation" not in out
    assert not any(out["image_mask"].values())
    assert out["tokenized_prompt_mask"].sum() == 6  # language untouched


def test_apply_ablation_mask_l():
    out = _transforms.ApplyAblationMask()({**_ablation_data(), "ablation": "mask_l"})
    assert out["image_mask"]["base_0_rgb"] == np.True_  # vision untouched
    assert out["tokenized_prompt_mask"].sum() == 0
    assert out["tokenized_prompt_mask"].dtype == np.bool_


def test_apply_ablation_mask_vl():
    out = _transforms.ApplyAblationMask()({**_ablation_data(), "ablation": "mask_vl"})
    assert not any(out["image_mask"].values())
    assert out["tokenized_prompt_mask"].sum() == 0


def test_apply_ablation_mask_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown ablation"):
        _transforms.ApplyAblationMask()({**_ablation_data(), "ablation": "black_img"})
```

(Ensure `import pytest` and `from openpi import transforms as _transforms` / `import numpy as np` exist at top of the test file — check what's already imported and reuse the existing import style.)

- [ ] **Step 1.2: Run tests, verify they fail**

Run: `uv run pytest src/openpi/transforms_test.py -q -k ablation`
Expected: FAIL — `AttributeError: ... has no attribute 'ApplyAblationMask'`

- [ ] **Step 1.3: Implement**

In `src/openpi/transforms.py`, after `TokenizePrompt`:

```python
_ABLATION_CHOICES = ("none", "mask_v", "mask_l", "mask_vl")


@dataclasses.dataclass(frozen=True)
class ApplyAblationMask(DataTransformFn):
    """Masks out input modalities via attention masks for ablation experiments.

    Reads and removes an optional "ablation" key ("none", "mask_v", "mask_l",
    "mask_vl"). Must run after the tokenizer transform so that
    `tokenized_prompt_mask` exists. No-op when the key is absent.
    """

    def __call__(self, data: DataDict) -> DataDict:
        ablation = data.pop("ablation", None)
        if ablation is None:
            return data
        if not isinstance(ablation, str):
            ablation = str(np.asarray(ablation).item())
        if ablation not in _ABLATION_CHOICES:
            raise ValueError(f"Unknown ablation: {ablation!r}, expected one of {_ABLATION_CHOICES}")
        if ablation == "none":
            return data
        if "v" in ablation.removeprefix("mask_"):
            data["image_mask"] = {key: np.False_ for key in data["image_mask"]}
        if "l" in ablation.removeprefix("mask_"):
            data["tokenized_prompt_mask"] = np.zeros_like(data["tokenized_prompt_mask"])
        return data
```

- [ ] **Step 1.4: Run tests, verify pass**

Run: `uv run pytest src/openpi/transforms_test.py -q -k ablation`
Expected: 6 passed

- [ ] **Step 1.5: Commit**

```bash
git add src/openpi/transforms.py src/openpi/transforms_test.py
git commit -m "feat: ApplyAblationMask transform for attention-mask input ablation"
```

### Task 2: `SampleModalityDropout` transform (train-time)

**Files:**
- Modify: `src/openpi/transforms.py` (add after `ApplyAblationMask`)
- Test: `src/openpi/transforms_test.py`

- [ ] **Step 2.1: Write failing tests**

```python
def test_sample_modality_dropout_probabilities():
    tf = _transforms.SampleModalityDropout(p_vision=1.0, p_language=0.0)
    out = tf({"prompt": "x"})
    assert out["ablation"] == "mask_v"

    tf = _transforms.SampleModalityDropout(p_vision=0.0, p_language=1.0)
    assert tf({"prompt": "x"})["ablation"] == "mask_l"

    tf = _transforms.SampleModalityDropout(p_vision=1.0, p_language=1.0)
    assert tf({"prompt": "x"})["ablation"] == "mask_vl"

    tf = _transforms.SampleModalityDropout(p_vision=0.0, p_language=0.0)
    assert "ablation" not in tf({"prompt": "x"})


def test_sample_modality_dropout_is_random():
    tf = _transforms.SampleModalityDropout(p_vision=0.5, p_language=0.5)
    seen = {tuple(sorted(tf({}).items())) for _ in range(200)}
    assert len(seen) >= 3  # at least several distinct outcomes across samples
```

- [ ] **Step 2.2: Run, verify fail**

Run: `uv run pytest src/openpi/transforms_test.py -q -k modality_dropout`
Expected: FAIL — no attribute `SampleModalityDropout`

- [ ] **Step 2.3: Implement**

```python
@dataclasses.dataclass(frozen=True)
class SampleModalityDropout(DataTransformFn):
    """Randomly drops input modalities during training (writes the "ablation" key
    consumed by `ApplyAblationMask`). Insert in the repack stage so it never runs
    at inference time. Vision and language are sampled independently per example.
    """

    p_vision: float = 0.0
    p_language: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        drop_v = np.random.rand() < self.p_vision
        drop_l = np.random.rand() < self.p_language
        if drop_v and drop_l:
            data["ablation"] = "mask_vl"
        elif drop_v:
            data["ablation"] = "mask_v"
        elif drop_l:
            data["ablation"] = "mask_l"
        return data
```

- [ ] **Step 2.4: Run, verify pass**

Run: `uv run pytest src/openpi/transforms_test.py -q -k modality_dropout`
Expected: 2 passed

- [ ] **Step 2.5: Commit**

```bash
git add src/openpi/transforms.py src/openpi/transforms_test.py
git commit -m "feat: SampleModalityDropout transform for masked fine-tuning"
```

### Task 3: `LiberoInputs` passes the `ablation` key through

**Files:**
- Modify: `src/openpi/policies/libero_policy.py:42-83`
- Test: `src/openpi/transforms_test.py` (keep ablation tests together)

- [ ] **Step 3.1: Write failing test**

```python
def test_libero_inputs_forwards_ablation_key():
    from openpi.models import model as _model
    from openpi.policies import libero_policy

    tf = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05)
    example = libero_policy.make_libero_example()
    assert "ablation" not in tf(dict(example))
    out = tf({**example, "ablation": "mask_v"})
    assert out["ablation"] == "mask_v"
```

- [ ] **Step 3.2: Run, verify fail**

Run: `uv run pytest src/openpi/transforms_test.py -q -k forwards_ablation`
Expected: FAIL — KeyError "ablation" (LiberoInputs rebuilds the dict and drops it)

- [ ] **Step 3.3: Implement**

In `LiberoInputs.__call__`, after the `if "prompt" in data:` block (`libero_policy.py:80-81`):

```python
        # Forward the ablation flag (if any) so ApplyAblationMask can consume it
        # downstream; this transform rebuilds the dict, which would drop it.
        if "ablation" in data:
            inputs["ablation"] = data["ablation"]
```

- [ ] **Step 3.4: Run, verify pass**

Run: `uv run pytest src/openpi/transforms_test.py -q -k forwards_ablation`
Expected: 1 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/openpi/policies/libero_policy.py src/openpi/transforms_test.py
git commit -m "feat: forward ablation key through LiberoInputs"
```

### Task 4: Wire transforms into `LeRobotLiberoDataConfig` + survive-the-pipeline test

**Files:**
- Modify: `src/openpi/training/config.py:282-356` (`LeRobotLiberoDataConfig`)
- Test: `src/openpi/transforms_test.py`

- [ ] **Step 4.1: Write failing test** (full inference-side pipeline: LiberoInputs → Normalize(None stats…skip) → model transforms incl. ApplyAblationMask; verifies the string key survives intermediate transforms and masks end up flipped)

```python
def test_ablation_survives_libero_pipeline():
    from openpi.models import pi0_config
    from openpi.training import config as _train_config

    train_config = _train_config.get_config("pi05_libero")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # Compose the inference-side input pipeline (policy_config.py order, minus Normalize
    # which requires norm stats and only touches state/actions).
    from openpi import transforms as _tf
    pipeline = _tf.compose(
        [*data_config.data_transforms.inputs, *data_config.model_transforms.inputs]
    )
    from openpi.policies import libero_policy
    example = {**libero_policy.make_libero_example(), "ablation": "mask_vl"}
    out = pipeline(example)
    assert "ablation" not in out
    assert not any(out["image_mask"].values())
    assert out["tokenized_prompt_mask"].sum() == 0

    baseline = pipeline(dict(libero_policy.make_libero_example()))
    assert baseline["image_mask"]["base_0_rgb"]
    assert baseline["tokenized_prompt_mask"].sum() > 0
```

Note: this test downloads the small Paligemma tokenizer model on first run (network needed).

- [ ] **Step 4.2: Run, verify fail**

Run: `uv run pytest src/openpi/transforms_test.py -q -k survives_libero`
Expected: FAIL — `image_mask` still True (ApplyAblationMask not yet in model_transforms; the key either survives untouched or crashes `Observation.from_dict` later — either way the asserts fail)

- [ ] **Step 4.3: Implement**

In `LeRobotLiberoDataConfig`:

1. Add fields after `extra_delta_transform: bool = False`:

```python
    # Modality dropout for masked fine-tuning (train-time only; 0 disables).
    dropout_p_vision: float = 0.0
    dropout_p_language: float = 0.0
```

2. In `create()`, after `repack_transform = _transforms.Group(...)` add:

```python
        # Train-time modality dropout: repack transforms only run on dataset data,
        # never at inference, so this cannot affect evaluation.
        if self.dropout_p_vision > 0 or self.dropout_p_language > 0:
            repack_transform = repack_transform.push(
                inputs=[
                    _transforms.SampleModalityDropout(
                        p_vision=self.dropout_p_vision, p_language=self.dropout_p_language
                    )
                ]
            )
```

3. Change the `model_transforms` line to append the mask applier (always on; no-op without the key):

```python
        model_transforms = ModelTransformFactory()(model_config).push(
            inputs=[_transforms.ApplyAblationMask()]
        )
```

- [ ] **Step 4.4: Run, verify pass**

Run: `uv run pytest src/openpi/transforms_test.py -q -k survives_libero`
Expected: 1 passed

- [ ] **Step 4.5: Run the whole test file**

Run: `uv run pytest src/openpi/transforms_test.py -q`
Expected: all pass

- [ ] **Step 4.6: Commit**

```bash
git add src/openpi/training/config.py src/openpi/transforms_test.py
git commit -m "feat: wire ablation mask + modality dropout into LIBERO data config"
```

### Task 5: TrainConfig `pi05_libero_mask_ft`

**Files:**
- Modify: `src/openpi/training/config.py` (insert right after the `pi05_libero` entry, ~line 763)
- Test: command-line config resolution check

- [ ] **Step 5.1: Add config**

```python
    TrainConfig(
        name="pi05_libero_mask_ft",
        # LoRA fine-tune of pi05_libero with random modality dropout (mask ablation
        # phase 2, see docs/superpowers/specs/2026-06-11-pi05-mask-ablation-design.md).
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            dropout_p_vision=0.2,
            dropout_p_language=0.2,
        ),
        batch_size=32,  # 24GB (RTX 4090) budget; raise if memory allows
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
```

- [ ] **Step 5.2: Verify config resolves**

Run: `uv run python -c "from openpi.training import config; c = config.get_config('pi05_libero_mask_ft'); print(c.name, c.batch_size, c.data.dropout_p_vision)"`
Expected: `pi05_libero_mask_ft 32 0.2`

- [ ] **Step 5.3: Commit**

```bash
git add src/openpi/training/config.py
git commit -m "feat: pi05_libero_mask_ft LoRA config with modality dropout"
```

### Task 6: Eval client — request-level ablation flag

**Files:**
- Modify: `examples/libero/main.py:49` (Args.ablation), `:135-160` (ablation branches), `:241-248` (summary)

- [ ] **Step 6.1: Implement client changes**

Replace `examples/libero/main.py:49`:

```python
    ablation: str = "none"  # none / mask_v / mask_l / mask_vl / wrong_lang
```

Replace the ablation branches (`main.py:135-160`) with:

```python
                        prompt = str(task_description)
                        if args.ablation == "wrong_lang":
                            rng = random.Random(args.seed + task_id)
                            other_ids = [i for i in range(num_tasks_in_suite) if i != task_id]
                            wrong_id = rng.choice(other_ids)
                            prompt = str(all_instructions[wrong_id])
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": prompt,
                        }
                        if args.ablation.startswith("mask_"):
                            # Server-side attention-mask ablation (ApplyAblationMask).
                            element["ablation"] = args.ablation
```

Add validation at the top of `eval_libero` (after `np.random.seed`):

```python
    valid_ablations = ("none", "mask_v", "mask_l", "mask_vl", "wrong_lang")
    if args.ablation not in valid_ablations:
        raise ValueError(f"Unknown ablation: {args.ablation}, expected one of {valid_ablations}")
```

Add `"ablation": args.ablation,` to the `summary` dict (`main.py:241-248`).

(`infer_img` / `infer_wrist` temp vars are gone — use `img` / `wrist_img` directly in `element`.)

- [ ] **Step 6.2: Syntax check** (libero deps aren't installed locally)

Run: `uv run python -m py_compile examples/libero/main.py`
Expected: exit 0

- [ ] **Step 6.3: Commit**

```bash
git add examples/libero/main.py
git commit -m "feat: switch eval client to server-side attention-mask ablation"
```

### Task 7: Runner script — multi-arm × multi-suite on one warm server

**Files:**
- Modify: `examples/libero/run_ablation_matrix.sh` (currently untracked — bring under git)

- [ ] **Step 7.1: Update script**

Change the argument handling so the first arg is a comma-separated list of ablations (server stays warm across ALL arms and suites; ablation is now per-request):

```bash
# Usage:
#   examples/libero/run_ablation_matrix.sh <ablation[,ablation...]> [suite ...]
# e.g. examples/libero/run_ablation_matrix.sh none,mask_v,mask_l,mask_vl libero_goal libero_spatial

IFS=',' read -r -a ABLATIONS <<< "${1:-none}"
shift || true
SUITES=("$@")
if [ ${#SUITES[@]} -eq 0 ]; then
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi
```

…and wrap the suite loop:

```bash
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
```

(Keep server startup/teardown as-is; `OUT_ROOT` moves inside the loop. `TRIALS` env var defaults to 10 per spec.)

- [ ] **Step 7.2: Shell syntax check**

Run: `bash -n examples/libero/run_ablation_matrix.sh`
Expected: exit 0

- [ ] **Step 7.3: Commit**

```bash
git add examples/libero/run_ablation_matrix.sh
git commit -m "feat: ablation matrix runner — multi-arm on one warm server"
```

### Task 8: Blindness check script (runs on AutoDL)

**Files:**
- Create: `scripts/check_ablation_blindness.py`

- [ ] **Step 8.1: Write script**

```python
"""Sanity check that attention-mask ablation truly blinds the model.

With ablation="mask_vl" and fixed noise, two completely different observations
must produce bit-identical action chunks. Requires a GPU with the pi05_libero
checkpoint (run on the AutoDL box, not locally).

Usage: uv run python scripts/check_ablation_blindness.py
"""

import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


def make_obs(seed: int, prompt: str) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.random(8),
        "observation/image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def main():
    config = _config.get_config("pi05_libero")
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    policy = policy_config.create_trained_policy(config, ckpt)

    noise = np.zeros((config.model.action_horizon, config.model.action_dim), dtype=np.float32)

    obs_a = {**make_obs(0, "pick up the red bowl"), "ablation": "mask_vl"}
    obs_b = {**make_obs(1, "open the top drawer"), "ablation": "mask_vl"}
    actions_a = policy.infer(obs_a, noise=noise)["actions"]
    actions_b = policy.infer(obs_b, noise=noise)["actions"]
    np.testing.assert_array_equal(actions_a, actions_b)
    print("PASS: mask_vl — model is blind to both vision and language")

    # Negative control: without ablation the same two observations must differ.
    actions_a = policy.infer(make_obs(0, "pick up the red bowl"), noise=noise)["actions"]
    actions_b = policy.infer(make_obs(1, "open the top drawer"), noise=noise)["actions"]
    assert not np.array_equal(actions_a, actions_b), "negative control failed: outputs identical without masking"
    print("PASS: negative control — outputs differ without masking")

    # mask_v: same language, different images -> identical actions.
    obs_a = {**make_obs(0, "pick up the red bowl"), "ablation": "mask_v"}
    obs_b = {**make_obs(1, "pick up the red bowl"), "ablation": "mask_v"}
    obs_b["observation/state"] = obs_a["observation/state"]
    np.testing.assert_array_equal(
        policy.infer(obs_a, noise=noise)["actions"], policy.infer(obs_b, noise=noise)["actions"]
    )
    print("PASS: mask_v — model is blind to vision")

    # mask_l: same images, different language -> identical actions.
    obs_a = {**make_obs(0, "pick up the red bowl"), "ablation": "mask_l"}
    obs_b = {**make_obs(0, "open the top drawer"), "ablation": "mask_l"}
    np.testing.assert_array_equal(
        policy.infer(obs_a, noise=noise)["actions"], policy.infer(obs_b, noise=noise)["actions"]
    )
    print("PASS: mask_l — model is deaf to language")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Syntax check locally**

Run: `uv run python -m py_compile scripts/check_ablation_blindness.py`
Expected: exit 0

- [ ] **Step 8.3: Commit**

```bash
git add scripts/check_ablation_blindness.py
git commit -m "feat: blindness sanity check for ablation masks"
```

### Task 9: AutoDL runbook

**Files:**
- Create: `docs/superpowers/specs/2026-06-11-autodl-runbook.md`

- [ ] **Step 9.1: Write runbook**

```markdown
# AutoDL 实验 Runbook（4090 / 24GB）

## 0. 环境
租 4090 实例（镜像选带 CUDA 12 的 Ubuntu 22.04），然后：
    git clone <你的仓库> && cd my_pi0_ablation
    git submodule update --init --recursive
    curl -LsSf https://astral.sh/uv/install.sh | sh
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    # LIBERO 评测依赖（无 docker 时直接装):
    uv pip install -e packages/openpi-client -e third_party/libero
    uv pip install -r examples/libero/requirements.txt 2>/dev/null || true

## 1. 盲测校验（先跑这个，确认 mask 真的生效）
    uv run python scripts/check_ablation_blindness.py
4 个 PASS 才继续。

## 2. 第一阶段评测（约 7–10 GPU 时）
有 docker 用脚本：
    TRIALS=10 examples/libero/run_ablation_matrix.sh none,mask_v,mask_l,mask_vl libero_goal libero_spatial
无 docker 手动跑：起 server
    uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero
另一终端循环跑 client（每个 ablation × suite 组合）：
    uv run python examples/libero/main.py --ablation mask_v \
      --task-suite-name libero_goal --num-trials-per-task 10 \
      --video-out-path data/libero/mask_v
结果在 data/libero/<ablation>/<suite>/summary.json。

## 3. 第二阶段微调（约 15 GPU 时）
    uv run scripts/compute_norm_stats.py --config-name pi05_libero_mask_ft   # 如报缺 norm stats
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_mask_ft \
      --exp-name mask_ft_run1 --overwrite
显存不够就把 config 里 batch_size 降到 16。先看 100 步确认 loss 下降再放着跑。

## 4. 微调后重评
serve 换成微调 checkpoint：
    uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config pi05_libero_mask_ft \
      --policy.dir checkpoints/pi05_libero_mask_ft/mask_ft_run1/20000
再跑第 2 节的 4 个臂，输出目录加前缀 data/libero_ft/。

## 5. 汇总
    find data -name summary.json | xargs -I{} sh -c 'echo {}; cat {}'
8 个 summary（2 阶段 × 4 臂 × 2 suite 合并后）按 spec §6 解读规则读。
```

- [ ] **Step 9.2: Commit**

```bash
git add docs/superpowers/specs/2026-06-11-autodl-runbook.md
git commit -m "docs: AutoDL runbook for mask ablation experiments"
```

---

## Self-review notes

- Spec coverage: §8.1→Task 1, §8.2→Task 2, §8.3→Task 3, transforms wiring→Task 4, §8.4→Task 5, §8.5→Task 6, §8.6→Task 7, §8.7→Task 8; eval protocol/budget→Task 9 runbook. ✓
- The old `empty_lang`/`black_img` client paths are deleted in Task 6 (spec §3 deprecates them); `wrong_lang` retained. ✓
- Risk "Normalize tolerates string key": covered by Task 4 test composing the data_transforms pipeline (Normalize itself is excluded there since it needs norm stats, but it only touches state/actions keys; the blindness check on AutoDL covers the full path including Normalize). 
- Type consistency: `ablation` key is a plain str end-to-end; `ApplyAblationMask` also tolerates 0-d numpy arrays in case msgpack/tree-map wraps it. ✓
