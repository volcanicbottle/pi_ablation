# π₀.₅ 输入模态 Mask Ablation 实验设计

日期：2026-06-11（修订 2：确认 pi05_libero 无 state 输入，矩阵缩为 4 臂）
状态：已批准

## 1. 背景与科学问题

对 `pi05_libero` checkpoint（π₀.₅ 在 LIBERO 上微调的官方模型）做输入模态 ablation，
回答：**模型是否 overfit——即它是否在"背任务轨迹"或"靠场景猜任务"，而不是真正使用
视觉和语言信息？**

具体怀疑（两个都查）：
- 模型可能没在看图（按指令开环回放背下来的动作序列）
- 模型可能没在听指令（LIBERO 场景与任务强相关，看场景即可猜出任务）

## 2. 模型输入结构（实测确认，与直觉不同）

**`pi05_libero` 只有两路有效输入：视觉 V（两个相机）和语言 L。**

确认过程：
- `config.py:745`：该配置显式 `discrete_state_input=False` → state 不拼进 prompt
- `pi0.py:151`：连续 state token 仅在 `not pi05` 分支加入 → pi05 模型无 state 投影层
- `obs.state` 在 forward 中唯一用途是取 batch size（`pi0.py:229`）

即：评测客户端发送的机器人状态被服务端模型**完全忽略**。官方训练时刻意去掉了
state 输入，"靠本体状态背轨迹"这条 overfit 路径被架构直接排除。

| 输入 | 进入模型的方式 | mask 通路 |
|---|---|---|
| 主相机 base_0_rgb | SigLIP → image tokens | `image_mask["base_0_rgb"] = False` |
| 腕相机 left_wrist_0_rgb | 同上 | `image_mask["left_wrist_0_rgb"] = False` |
| 语言指令 L | pi0 格式 tokenize（无 State 段） | `tokenized_prompt_mask` 全置 False |

附带简化：原设计的"Task/State 段级 token mask"不再需要——prompt 里只有指令，
−L 直接把整条 `tokenized_prompt_mask` 置 False 即可。

## 3. Mask 实现语义：attention mask（决定）

统一用 **attention mask**（置 `input_mask=False`，token 等于不存在），不用置零输入
（黑图/空字符串）。理由：黑图是一张分布外的真实图像，掉点会混入"分布外冲击"，
污染"信息缺失"的测量。

仓库已有的客户端置零实现（commit 086fa7e：`empty_lang` / `black_img`）废弃；
`wrong_lang`（喂错误任务指令）保留为可选臂。评测脚手架（`run_ablation_matrix.sh`、
summary.json、失败视频留存）复用。

实现位置：新增通用 transform `ApplyAblationMask`（读取数据字典中的 `ablation` 键，
无键时 no-op），挂在 LIBERO 数据配置 model_transforms 末尾；评测客户端按请求传
`ablation` 字段。好处：一个常驻 policy server 可连续跑完所有臂，不需重启；
训练与推理共用同一套 mask 代码，保证训练/测试一致。

## 4. 第一阶段：推理时 mask ablation（4 臂矩阵）

对原 `pi05_libero` checkpoint，不改权重，只在推理时 mask：

| # | 臂 | ablation 值 | 回答的问题 |
|---|---|---|---|
| 1 | baseline | `none` | 性能上界 |
| 2 | −V | `mask_v` | **背轨迹主检测**：指令告诉它任务，失明还能做成 = 背下了任务→动作序列（开环执行，无视觉反馈） |
| 3 | −L | `mask_l` | 听不听指令（仅 goal suite 有诊断力，见 §6） |
| 4 | −V−L（=全 mask） | `mask_vl` | 无条件生成 = 模型先验下界 |

可选附加臂：`wrong_lang`，区分"忽略语言"与"错误使用语言"。

### 下界参照

- 随机/零动作的环境水分：按用户判断取 ≈0（LIBERO 任务需实际完成操作）。
- 臂 4（mask_vl）即模型先验下界，不需要单独的全 mask 臂。
- 解读规则：mask 后成功率与下界比，**掉得越少越说明该信息没被使用 = overfit 证据**。

### 评测协议

- Suite：第一轮只跑 **libero_goal** + **libero_spatial**；object / long 视信号决定是否补。
- 每任务 10 trials，每 suite 10 任务 → 每臂 200 rollouts，4 臂共 800 rollouts。
- 统计精度：约 ±5% 二项置信区间，足以分辨大效应。
- 固定 seed，各臂使用相同初始状态集（`get_task_init_states` 按 episode_idx 索引，
  天然一致），保证臂间可比。

## 5. 第二阶段：modality-dropout LoRA 微调 + 重跑矩阵

### 动机

第一阶段的掉点混杂两种原因：(1) 信息真的必要；(2) 模型没见过缺模态输入，被分布外
情况"吓到"。第二阶段消除 (2)。

### 微调方案

- 起点：`pi05_libero` checkpoint，**LoRA**（24GB 显存约束），新增 TrainConfig
  `pi05_libero_mask_ft`，openpi 自带 LIBERO 数据（`physical-intelligence/libero`）。
- 输入侧改动：每个训练样本独立以 **p=0.2** 概率丢 V、**p=0.2** 概率丢 L
  （独立采样，64% 样本完整）。实现为训练专用 transform `SampleModalityDropout`
  （挂在 repack 段，只在训练管道运行），写 `ablation` 键，由同一个
  `ApplyAblationMask` 执行——与评测时的 mask 机制完全一致。
- 步数 2 万步以内，看验证 loss 提前停；LoRA rank / 学习率用 openpi 默认；
  batch size 按 24GB 调（起步 32）。

### 微调后

用同一个微调模型**重跑第一阶段全部 4 臂**。

## 6. 解读规则（两阶段对照）

各阶段内部和**自己的 baseline** 比（dropout 微调可能使 baseline 略降 1–2 点，
不跨阶段比绝对值）。

**−L 臂的结论只在 goal suite 上有效**：libero_goal 同场景多任务，指令是区分任务的
唯一信息，−L 不掉 = overfit 实锤；而 libero_spatial / object 的场景布局本身可能
唯一确定任务，语言冗余，−L 不掉是正常现象，不构成证据。反向利用：spatial 上的
−L 预期只小掉，若大掉则提示 mask 实现可能有 bug。

| 观察 | 结论 |
|---|---|
| −V 远高于下界（mask_vl） | 背下了任务轨迹，闭眼开环也能执行 → overfit 实锤 |
| 微调前 −L 几乎不掉（goal suite） | 模型本来就没在听指令 → 直接 overfit 证据 |
| 微调前 −V 大掉，微调后 −V 几乎不掉 | 看似依赖视觉实为"吓到"；任务可被记忆解决 → overfit 隐患实锤 |
| 微调前后 −V 都掉到下界附近 | 视觉是真刚需，无背轨迹证据 |
| mask_vl 显著 >0 | 模型先验（平均动作风格）本身能蒙对的水分，解读其他臂时扣除 |

## 7. 算力与预算

- 平台：AutoDL 租 4090（24GB），约 3 元/小时。本机（RTX 5060 Laptop 8GB）只改代码
  跑单元测试，不跑模型。
- 第一阶段评测：800 rollouts ≈ 7–10 GPU 时 ≈ 27 元
- LoRA 微调：≈ 15 GPU 时 ≈ 45 元
- 第二阶段评测：≈ 27 元
- 合计 ≈ 100 元。

## 8. 实现要点

1. `ApplyAblationMask`（`src/openpi/transforms.py`）：pop `ablation` 键
   （`none/mask_v/mask_l/mask_vl`），按值把 `image_mask` 全 False /
   `tokenized_prompt_mask` 全 False。无键 no-op。挂在
   `LeRobotLiberoDataConfig.create()` 的 model_transforms 末尾（训练与推理共用）。
2. `SampleModalityDropout`（同文件）：按概率采样写 `ablation` 键；只挂在
   repack_transforms（训练专用管道段，推理不经过）。
3. `LiberoInputs`（`src/openpi/policies/libero_policy.py`）：透传 `ablation` 键
   （该 transform 重建字典会丢未知键）。
4. TrainConfig `pi05_libero_mask_ft`（`src/openpi/training/config.py`）：pi05 LoRA
   变体 + dropout 数据配置 + 从 `gs://openpi-assets/checkpoints/pi05_libero/params`
   加载权重 + freeze filter + ema off。
5. 评测客户端 `examples/libero/main.py`：`--ablation` 取值改为
   `none/mask_v/mask_l/mask_vl/wrong_lang`，mask 系列按请求发送 `ablation` 字段；
   删除 `empty_lang`/`black_img` 置零路径；summary.json 记录 ablation。
6. `run_ablation_matrix.sh`：支持一次启动 server 连跑多臂 × 多 suite。
7. 盲测校验脚本：mask_vl 下喂两组不同图像+指令、固定噪声，断言输出动作逐位相同
   （证明模型对被 mask 的输入真正不可见）；在 AutoDL 上跑。

## 9. 风险

- pi05 + LoRA 变体组合未在官方配置中出现过（官方 LoRA 例子是 pi0），需在 AutoDL
  上先跑通几十步冒烟验证。
- LIBERO 评测在 4090 上的实际 rollout 速度未实测，预算按 30–45 秒/rollout 估。
- `Normalize` transform 对字典里混入字符串键（`ablation`）的容忍性需单元测试确认。
