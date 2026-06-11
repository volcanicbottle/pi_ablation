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
