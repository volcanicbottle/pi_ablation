# π₀.₅ 输入模态 Mask Ablation 实验设计

日期：2026-06-11
状态：待用户确认

## 1. 背景与科学问题

对 `pi05_libero` checkpoint（π₀.₅ 在 LIBERO 上微调的官方模型）做输入模态 ablation，
回答：**模型是否 overfit——即它是否在"背轨迹"或"靠场景猜任务"，而不是真正使用视觉
和语言信息？**

具体怀疑（三个都查）：
- 模型可能没在看图（靠 state 在记住的轨迹上"查表"续写动作）
- 模型可能没在听指令（LIBERO 场景与任务强相关，看场景即可猜出任务）
- 模型可能过度依赖本体状态 state

## 2. 模型输入结构（π₀.₅ 特有，影响实现）

π₀.₅-LIBERO 共四路输入：

| 输入 | 进入模型的方式 | mask 通路 |
|---|---|---|
| 主相机图像 | SigLIP → image tokens | `Observation.image_masks[name]`（pi0.py embed_prefix） |
| 手腕相机图像 | 同上 | 同上 |
| 语言指令 L | tokenize 进 prompt | `tokenized_prompt_mask` 的 "Task: ..." 段 |
| 本体状态 S | **离散化成文本拼进 prompt**（tokenizer.py: `"Task: {prompt}, State: {state_str};\nAction: "`） | `tokenized_prompt_mask` 的 "State: ..." 段 |

关键点：π₀.₅ 的 L 和 S 共享同一条 token 流（knowledge insulation 设计），
分别 ablate 必须做 **token 段级 mask**——按 tokenizer 的拼接结构定位
Task 段与 State 段的 token 区间，分段控制 mask。

## 3. Mask 实现语义：attention mask（决定）

统一用 **attention mask**（置 `input_mask=False`，token 等于不存在），不用置零输入
（黑图/空字符串）。理由：黑图是一张分布外的真实图像，掉点会混入"分布外冲击"，
污染"信息缺失"的测量。

注意：仓库中已有的客户端置零实现（commit 086fa7e：`empty_lang` / `wrong_lang` /
`black_img`，在 `examples/libero/main.py`）**不符合本设计**，机制需替换为服务端
attention mask。但其评测脚手架（`run_ablation_matrix.sh` 跑批脚本、summary.json
输出、失败视频留存）直接复用。

mask 开关从评测客户端经 websocket 传到 policy server，在服务端的
policy transforms / 模型侧生效（具体接口在实施计划中定）。

## 4. 第一阶段：推理时 mask ablation（6 臂矩阵）

对原 `pi05_libero` checkpoint，不改权重，只在推理时 mask：

| # | 臂 | mask 内容 | 回答的问题 |
|---|---|---|---|
| 1 | baseline | 无 | 性能上界 |
| 2 | −V | 两个相机全 mask | 模型看不看图 |
| 3 | −L | Task 段 token | 模型听不听指令 |
| 4 | −S | State 段 token | 模型是否依赖本体状态定位轨迹进度 |
| 5 | −V−L | 视觉+语言，只留 S | **背轨迹的最直接检测** |
| 6 | 全 mask | V+L+S 全 mask | 模型先验下界（"肌肉记忆"能拿几分） |

已砍掉的臂：分相机 ablation（−主相机 / −腕相机）——主问题不需要，留作后续可选。
可选附加臂：`wrong_lang`（喂错误任务的指令，复用已有实现）——区分"忽略语言"与
"错误使用语言"，第二优先。

### 下界参照

- 随机/零动作的环境水分：按用户判断取 ≈0（LIBERO 任务需实际完成操作）。
- 臂 6（全 mask）作为模型先验下界。
- 解读规则：mask 后成功率应与下界比，**掉得越少越说明该信息没被使用 = overfit 证据**。

### 评测协议

- Suite：第一轮只跑 **libero_goal**（语言 ablation 最有诊断力：同场景多任务）
  + **libero_spatial**（视觉 ablation 最有诊断力：空间关系必须看图）。
  object / long 视第一轮信号决定是否补。
- 每任务 10 trials，每 suite 10 任务 → 每臂 200 rollouts，6 臂共 1200 rollouts。
- 统计精度：±5% 左右的二项置信区间，足以分辨"掉到下界"vs"保留大量能力"的大效应。
- 固定 seed，各臂使用相同初始状态集，保证臂间可比。

## 5. 第二阶段：modality-dropout LoRA 微调 + 重跑矩阵

### 动机

第一阶段的掉点混杂两种原因：(1) 信息真的必要；(2) 模型没见过缺模态输入，
被分布外情况"吓到"。第二阶段消除 (2)。

### 微调方案

- 起点：`pi05_libero` checkpoint，**LoRA**（24GB 显存约束），openpi 自带 LIBERO
  训练配置与数据。
- 唯一改动在输入侧：每个训练样本独立以 **p=0.2** 的概率分别丢弃 V / L / S
  （三次独立采样，约 51% 样本输入完整）。
- 丢弃的实现与评测时**完全相同**（attention mask），避免引入新的训练/测试不一致。
- 步数 1–2 万步，验证 loss 平了即停；LoRA rank / 学习率用 openpi 默认。

### 微调后

用同一个微调模型**重跑第一阶段全部 6 臂**。

## 6. 解读规则（两阶段对照）

各阶段内部和**自己的 baseline** 比（dropout 微调可能使 baseline 略降 1–2 点，
不跨阶段比绝对值）。

| 观察 | 结论 |
|---|---|
| 微调前 −L 几乎不掉 | 模型本来就没在听指令 → 直接 overfit 证据 |
| 微调前 −V 大掉，微调后 −V 几乎不掉 | 看似依赖视觉实为"吓到"；任务可被记忆解决 → overfit 隐患实锤 |
| 微调前后 −V 都掉到下界附近 | 视觉是真刚需 → 无背轨迹证据 |
| −V−L（只留 S）远高于下界 | 纯靠本体状态即可续写动作 → 背轨迹实锤 |
| −S 几乎不掉 | state 信息冗余（视觉已覆盖），不算 overfit |

## 7. 可选第三阶段（仅当第二阶段结果可疑时）

若微调后 −V−L 臂成功率仍显著高于下界（如 >40%），追加一次**专属微调**：
训练全程 mask V+L、只留 S，得到"纯 state 模型"。该模型若能训到高成功率，
即证明任务可纯靠轨迹记忆完成（定罪的最后一锤）；若收敛后仍低，说明背轨迹
这条路本身走不通。

## 8. 算力与预算

- 平台：AutoDL 租 4090（24GB），约 3 元/小时。本机（RTX 5060 Laptop 8GB）
  只用于改代码调试，不跑模型。
- 第一阶段评测：1200 rollouts ≈ 10–15 GPU 时 ≈ 40 元
- LoRA 微调：≈ 15 GPU 时 ≈ 45 元
- 第二阶段评测：≈ 40 元
- 合计 ≈ 130 元；可选第三阶段再 +60 元左右。

## 9. 实现要点（细节归实施计划）

1. **V mask**：评测端传开关 → 服务端在构造 `Observation` 时把对应
   `image_masks[name]` 置 False（机制现成，`pi0.py:113-125` 已按 mask 跳过）。
2. **L/S 段级 mask**：改 `tokenizer.py` 的 `tokenize()`，分段 tokenize
   （Task 段 / State 段 / "Action:" 尾），返回各段边界；按臂配置将对应段的
   `tokenized_prompt_mask` 置 False。"Action: " 尾段永不 mask。
3. **dropout 微调**：在训练数据 transforms 中加随机模态丢弃（与上述同一套
   mask 机制），写进训练 config。
4. **评测脚手架**：复用 `run_ablation_matrix.sh` + `examples/libero/main.py`
   的 summary/视频输出，把 `--ablation` 参数语义从置零改为 mask 配置。
5. 已有的客户端置零代码路径保留但不再使用（或显式标记 deprecated），
   `wrong_lang` 保留为可选臂。

## 10. 风险

- token 段边界定位要小心 BOS/分隔符的归属，需要单元测试验证"mask Task 段后
  State 段 token 完全不变"。
- LIBERO 评测在 4090 上的实际 rollout 速度未实测，预算按 30–45 秒/rollout 估，
  偏差 ±50% 在可接受范围。
- `pi05_libero` 推理显存若超 24GB（不太可能）需要降 batch 或换卡。
