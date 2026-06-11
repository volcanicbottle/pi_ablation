# AutoDL 实验 Runbook（4090 / 24GB）

按顺序执行；每步的预期耗时和产出都标了。设计与解读规则见
`2026-06-11-pi05-mask-ablation-design.md`。

## 0. 环境

租 4090 实例（镜像选带 CUDA 12 的 Ubuntu 22.04），然后：

```bash
git clone <你的仓库> && cd my_pi0_ablation
git submodule update --init --recursive
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
# LIBERO 评测依赖（无 docker 时直接装）:
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install -r examples/libero/requirements.txt
```

## 1. 盲测校验（先跑这个，确认 mask 真的生效）

```bash
uv run python scripts/check_ablation_blindness.py
```

4 个 PASS 才继续。首跑会下载 pi05_libero checkpoint（约 12GB，缓存在
`~/.cache/openpi`，AutoDL 注意系统盘空间，可设 `OPENPI_DATA_HOME` 到数据盘）。

## 2. 第一阶段评测（约 7–10 GPU 时 ≈ 27 元）

有 docker 用脚本（一个 server 连跑 4 臂 × 2 suite）：

```bash
TRIALS=10 examples/libero/run_ablation_matrix.sh none,mask_v,mask_l,mask_vl libero_goal libero_spatial
```

无 docker 手动跑——终端 A 起 server：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero
```

终端 B 循环跑 client（每个 ablation × suite 组合一次）：

```bash
for ab in none mask_v mask_l mask_vl; do
  for suite in libero_goal libero_spatial; do
    uv run python examples/libero/main.py --ablation $ab \
      --task-suite-name $suite --num-trials-per-task 10 \
      --video-out-path data/libero/$ab
  done
done
```

结果在 `data/libero/<ablation>/<suite>/summary.json`。

## 3. 第二阶段微调（约 15 GPU 时 ≈ 45 元）

```bash
# 如报缺 norm stats 先算（pi05_libero 的 stats 会从 checkpoint assets 取，一般不用）:
uv run scripts/compute_norm_stats.py --config-name pi05_libero_mask_ft

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_mask_ft \
  --exp-name mask_ft_run1 --overwrite
```

- 显存不够：把 `config.py` 里 `pi05_libero_mask_ft` 的 `batch_size` 降到 16。
- 先看前 100 步 loss 确认在降，再放着跑；2 万步内验证 loss 平了可提前停。

## 4. 微调后重评（约 7–10 GPU 时 ≈ 27 元）

server 换微调 checkpoint（步数目录按实际）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero_mask_ft \
  --policy.dir checkpoints/pi05_libero_mask_ft/mask_ft_run1/20000
```

再跑第 2 节的 4 臂循环，输出目录改成 `data/libero_ft/$ab`。

## 5. 汇总

```bash
find data -name summary.json | xargs -I{} sh -c 'echo {}; cat {}'
```

共 16 份 summary（2 阶段 × 4 臂 × 2 suite），按设计文档 §6 解读规则读：
各阶段与自己的 baseline 比；−L 只在 goal suite 上下结论；−V 远高于 mask_vl
下界 = 背轨迹证据。
