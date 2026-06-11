import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


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


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
