# Mask Ablation 当前进度与代码导读

日期：2026-06-13
分支：`feat/mask-ablation`（工作树干净，已全部提交）

## 一句话状态

π₀.₅ 输入模态 mask ablation 的**两个阶段代码已全部实现 + 单元测试就绪**，
但**还没在 GPU 上实跑**：`data/` 为空，无微调 checkpoint、无 rollout 结果。
当前卡点 = 需要上 AutoDL 4090 按 runbook 执行。

## 这是在做什么（科学问题）

对官方 `pi05_libero` checkpoint 做输入 ablation，验证它是否 **overfit**：
是真在用视觉/语言，还是在"背任务轨迹"或"靠场景猜任务"。
关键架构事实：`pi05_libero` **没有 state 输入**（被官方训练去掉），
有效输入只有视觉 V（双相机）+ 语言 L，所以矩阵是 4 臂。

实验方法用 **attention mask**（`input_mask=False`，token 当不存在）而非置零输入，
避免黑图/空串引入"分布外冲击"污染测量。设计与解读规则见
`2026-06-11-pi05-mask-ablation-design.md`，执行步骤见 `2026-06-11-autodl-runbook.md`。

## 进度清单

| 阶段 | 内容 | 状态 |
|---|---|---|
| 设计 | 实验设计 + 4 臂矩阵 + 解读规则 | ✅ 已批准 |
| 代码-阶段1 | 推理时 attention-mask ablation（4 臂） | ✅ 已实现+测试 |
| 代码-阶段2 | modality-dropout LoRA 微调配置 | ✅ 已实现 |
| 校验脚本 | 盲测断言 mask 真生效 | ✅ 已写（需 GPU 跑） |
| 运行脚手架 | 一个常驻 server 连跑多臂×多 suite | ✅ 已写 |
| **阶段1 实跑** | 800 rollouts，出 summary.json | ⬜ 未跑 |
| **阶段2 微调** | LoRA 2 万步 | ⬜ 未跑 |
| **阶段2 重评** | 微调后重跑 4 臂 | ⬜ 未跑 |

## 该读哪些代码（按重要度）

1. **`src/openpi/transforms.py`** — 核心机制，两个新 transform：
   - `ApplyAblationMask`：pop `ablation` 键（`none/mask_v/mask_l/mask_vl`），
     按值把 `image_mask` 全 False / `tokenized_prompt_mask` 全置零。无键 no-op。
     训练与推理共用，挂在 model_transforms 末尾（必须在 tokenizer 之后）。
   - `SampleModalityDropout`：训练时按 p_vision/p_language 独立采样写 `ablation` 键，
     只挂在 repack 段（推理不经过）。

2. **`src/openpi/training/config.py`** — 两处：
   - `LeRobotLiberoDataConfig`：新增 `dropout_p_vision/dropout_p_language`，
     `create()` 里把 dropout 挂 repack、把 `ApplyAblationMask` 挂 model_transforms 末尾。
   - 新 TrainConfig `pi05_libero_mask_ft`：pi05 + LoRA（gemma_2b_lora /
     gemma_300m_lora）+ dropout 0.2/0.2 + 从 `gs://openpi-assets/checkpoints/pi05_libero/params`
     加载 + freeze_filter + ema off + batch 32 + 2 万步。

3. **`examples/libero/main.py`** — 评测客户端：`--ablation` 取值改为
   `none/mask_v/mask_l/mask_vl/wrong_lang`，`mask_*` 按请求发 `ablation` 字段，
   删掉旧的 `empty_lang/black_img` 置零路径，summary.json 记录 ablation。

4. **`src/openpi/policies/libero_policy.py`** — `LiberoInputs` 透传 `ablation` 键
   （该 transform 重建字典会丢未知键，必须显式转发）。

5. **`scripts/check_ablation_blindness.py`** — 盲测：mask_vl/mask_v/mask_l 下喂不同
   输入 + 固定噪声，断言动作逐位相同；含 negative control。**实跑前必须先过这个**。

6. **`examples/libero/run_ablation_matrix.sh`** — docker compose 起一个常驻 server，
   循环跑 ablation×suite，结果落 `data/libero/<ablation>/<suite>/`。

7. **`src/openpi/transforms_test.py`** — `ApplyAblationMask`/`SampleModalityDropout`
   的单元测试（noop / 各 mask 值 / 拒绝未知值 / dropout 概率）。本机可跑。

## 下一步

1. 本机：`uv run pytest src/openpi/transforms_test.py` 确认测试绿。
2. 上 AutoDL 4090，按 `2026-06-11-autodl-runbook.md`：
   盲测校验 → 阶段1评测(800 rollouts) → LoRA微调 → 阶段2重评。
3. 16 份 summary.json 按设计文档 §6 解读：各阶段与自身 baseline 比；
   −L 只在 goal suite 下结论；−V 远高于 mask_vl 下界 = 背轨迹实锤。

## 风险（来自设计文档 §9）

- pi05 + LoRA 组合官方没出现过（官方 LoRA 例子是 pi0），需先冒烟跑几十步。
- 4090 上 rollout 速度未实测，预算按 30–45 秒/rollout 估。
- `Normalize` 对混入的字符串 `ablation` 键容忍性——已有单元测试覆盖。
</content>
</invoke>
