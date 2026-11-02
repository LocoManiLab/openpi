# Fine-tuning pi0.5 on a custom G1 Sonic VLA dataset

本文档记录如何用本仓库（下文以 `/path/to/openpi` 表示 checkout 路径）在自己的 LeRobot 格式数据上微调 pi0.5（base 来自 `gs://openpi-assets/checkpoints/pi05_base`，下载到本地后以 `/path/to/pi05_base` 表示）。示例数据：`/path/to/datasets/2026-04-28-take-walk-and-put`。

如果只是想"复刻一遍"——直接跳到 [运行命令](#运行命令)。如果你的数据集 schema 跟示例不一样，先看 [适配自己的数据](#适配自己的数据)。

---

## 数据 / 模型布局

- 数据集：`/path/to/datasets/<repo_id>`，LeRobot v2.0 格式（`meta/`、`data/chunk-XXX/`、`videos/chunk-XXX/<cam_key>/`）。
- base 模型：`/path/to/pi05_base/params/`（Orbax checkpoint）。
- 训练配置名：`pi05_g1_sonic`，定义在 `src/openpi/training/config.py`。
- 数据 / 动作适配：`src/openpi/policies/g1_sonic_policy.py`。
- 容忍 shape 不匹配的 weight loader：`PartialCheckpointWeightLoader`，在 `src/openpi/training/weight_loaders.py`。

模型行为（pi0.5）：
- 单 RGB 输入 `observation.images.ego_view → base_0_rgb`，左右 wrist 摄像头补零、mask=False。
- State 43 dim → pad 到 80（pi05 的 state 走 token 路径，不存在 `state_proj`）。
- Action 78 dim = `action.motion_token (64) + teleop.left_hand_joints (7) + teleop.right_hand_joints (7)`，pad 到 80。
- 模型 `action_dim=80, action_horizon=16`；`action_in_proj` / `action_out_proj` 因维度跟 base (32) 不匹配会被重新初始化，**其余 48/51 个参数 leaf 从 pi05_base 加载**。

---

## 一次性准备

下面三件事只需要在新机器 / 新数据上做一次。

### 1. 把数据集 codebase_version 标为 v2.0

LeRobot v2.1 需要 `meta/episodes_stats.jsonl`，但示例数据集只导出了聚合 `stats.json`。把 `meta/info.json` 里：

```json
"codebase_version": "v2.1",
```

改成：

```json
"codebase_version": "v2.0",
```

LeRobot 加载时会走 `backward_compatible_episodes_stats` 兼容路径，使用 `stats.json`。本仓库会自己用 `compute_norm_stats.py` 重算 norm stats，所以这个 stats 内容不影响训练。

### 2. 准备 PaliGemma 分词器

openpi 训练时会从 GCS 拉 `gs://big_vision/paligemma_tokenizer.model`。如果机器到 GCS 的路径不稳，建议在 setup 阶段直接把它放进 openpi 缓存：

```bash
mkdir -p ~/.cache/openpi/big_vision
curl -sSL https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

要换缓存路径的话设 `OPENPI_DATA_HOME=<dir>`。

### 3. 把 base checkpoint 路径写进 train config（如已就位则跳过）

`pi05_g1_sonic` 里的 base 路径是占位符 `/path/to/local/pi05-base/params`。训练前把它改成本地实际路径：改 `src/openpi/training/config.py` 里同名 `TrainConfig` 的 `weight_loader=PartialCheckpointWeightLoader(...)`。

---

## 运行命令

每次开新 shell：

```bash
# 进环境
cd /path/to/openpi && source .venv/bin/activate

# 关键：让 LeRobot 用本地数据，不要去 HF Hub 拉
export HF_LEROBOT_HOME=/path/to/datasets
```

### Step 1 ─ 计算 norm stats

第一次跑，或换了数据集后必须重算：

```bash
JAX_PLATFORMS=cpu \
uv run scripts/compute_norm_stats.py --config-name pi05_g1_sonic
```

**`JAX_PLATFORMS=cpu` 不能省**：脚本本身只是 numpy 累加 RunningStats（不需要 GPU），但 `import openpi.training.config` 会拉起 JAX，JAX 默认会预分配 75% 显存。设成 cpu 后这部分显存就让出来了。

视频解码是真正的瓶颈（98840 帧 × 50fps mp4 seek+decode），数据集那么大 `num_workers=4` 大约 30–50 分钟。CPU 核多就把 config 里 `num_workers` 拉到 8–16。

输出落到 `/path/to/openpi/assets/pi05_g1_sonic/<repo_id>/norm_stats.json`。
跑完打开 `norm_stats.json` 扫一眼 `state` 和 `actions` 的 `q01` / `q99` / `std`，**任何维度 std≈0 或 q99-q01≈0 都会让归一化出 inf**——typically 是该维度数据里只有恒定值，需要决定剔除该维度或人工把对应 std 改成一个合理值（README 末尾 troubleshooting 也提到了这点）。

### Step 2 ─ 启训

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_g1_sonic \
    --exp-name=take_walk_put_v0 \
    --overwrite
```

- checkpoint 落到 `/path/to/openpi/checkpoints/pi05_g1_sonic/take_walk_put_v0/<step>/`。
- `--exp-name` 是这次实验的名字，会作为子目录、wandb run name；想多版本对比就改它，不加 `--overwrite`。
- 多 GPU FSDP（显存不够时）：`--fsdp-devices <N>`，N=可用 GPU 数。
- 关 wandb：加 `--no-wandb-enabled`，或临时 `WANDB_MODE=offline`。
- 调 batch / step：`--batch-size 16 --num-train-steps 50000` 这种 tyro flag 直接覆盖。

启动头一段日志应当包含：

```
WARNING openpi.training.weight_loaders:PartialCheckpointWeightLoader: re-initializing 3 params due to shape mismatch:
  action_in_proj/kernel: ckpt (32, 1024) vs model (80, 1024)
  action_out_proj/bias:  ckpt (32,)      vs model (80,)
  action_out_proj/kernel: ckpt (1024, 32) vs model (1024, 80)
```

**只这三行**——多于或少于这三行说明 config / 数据维度对不上，停下来排查。

### Step 3 ─ 起 policy server（推理 / 评估）

训练拿到一个 step 后（默认每 1000 步存一次，保留 5000 步整数倍）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_g1_sonic \
    --policy.dir=checkpoints/pi05_g1_sonic/take_walk_put_v0/<step>
```

server 监听 8000，输入字典遵循 `g1_sonic_policy.G1SonicInputs` 的结构：

```python
example = {
    "state": np.ndarray(43,),                           # 原始 43-dim 关节
    "image": {"ego_view": np.ndarray(H, W, 3)},         # uint8 H,W,C
    "prompt": "take the cup and walk forward and put it in basket",
}
```

返回字典遵循 `G1SonicOutputs`：

```python
{
    "motion_token":      (action_horizon, 64),
    "left_hand_joints":  (action_horizon, 7),
    "right_hand_joints": (action_horizon, 7),
}
```

---

## 适配自己的数据

如果你的数据 schema 跟示例不一样，按以下顺序改：

1. **`info.json` 字段名**：确认你的 image key、state key、action keys 跟 `LeRobotG1SonicDataConfig.repack_transforms` 里写的一致。键名带 `.`（LeRobot 习惯）就直接写 `"observation.images.ego_view"` 这种字面字符串，不要换成 `/`。
2. **维度**：
   - 若 action 总维度 ≤ 32：把 `pi05_g1_sonic` 的 `action_dim=32`，可换回 `CheckpointWeightLoader`，loss 收敛会快得多。
   - 若 > 32：保持 `PartialCheckpointWeightLoader` 加上 ≥ 实际维度的 `action_dim`（凑整到 8 倍数即可），让两个 proj 层重训。
3. **State 维度**：pi05 不存在 `state_proj`，state 大小不影响 shape；但 state 维度越大 → 离散化后 token 越多，要确保 `max_token_len` 够（pi05 默认 200；80 个 state 数 + prompt 一般 < 200）。
4. **多摄像头**：把 `LeRobotG1SonicDataConfig` 的 repack 加更多 image key，`G1SonicInputs` 里把 `left_wrist_0_rgb` / `right_wrist_0_rgb` 替换为真实 wrist 图，并把对应 mask 设为 `True`。pi05 最多 3 路：`base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb`。
5. **Action 拼接顺序**：`G1SonicInputs.__call__` 里 concat 的顺序就是你 inference 拿回 action 后的切片顺序，改了别忘了同步 `G1SonicOutputs` 里的 `MOTION_TOKEN_DIM / LEFT_HAND_DIM / RIGHT_HAND_DIM`。
6. **是否需要 delta action**：默认是 absolute（直接喂原始动作）。如果你的训练目标本来就是相对当前 state 的 delta，可以参考 `LeRobotLiberoDataConfig` 用 `_transforms.DeltaActions / AbsoluteActions` 组合。

---

## LoRA 低显存路线（可选）

22GB 显卡也能跑。在 `pi05_g1_sonic` 基础上复制一份，改：

```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_dim=80,
    action_horizon=16,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
),
freeze_filter=pi0_config.Pi0Config(
    pi05=True, action_dim=80, action_horizon=16,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
ema_decay=None,
```

LoRA 训练要保证 `freeze_filter` 跟 `model` 用同一组 variant 字符串。

---

## Troubleshooting

| 现象 | 解决 |
|---|---|
| `FileNotFoundError: episodes_stats.jsonl` | `info.json` 还是 v2.1，改成 v2.0。 |
| `ClientConnectorDNSError ... storage.googleapis.com` | 离线机器，按"准备 PaliGemma 分词器"那一节用 curl 提前下载到 `~/.cache/openpi/big_vision/`。 |
| 启动时 `PartialCheckpointWeightLoader` 列出 > 3 行 re-init | 你的 config 跟 base 模型其他维度也不一致（例如 `action_horizon` 改了但跟 base 没冲突就不会列；列出来就是真的对不齐），核对 `Pi0Config` 字段。 |
| `Token length (250) exceeds max length (200)` | state 维度太高，把 `max_token_len` 提到 256 或更大。 |
| `Local batch size larger than dataset size` | 数据集太小或 fps × episodes 没估准；要么减 `batch_size`，要么补数据。 |
| 训练 loss 飘到 inf / NaN | 看 `norm_stats.json` 中是否存在 std≈0 / q99-q01≈0 的维度，参考 README troubleshooting 末尾。 |
| `RuntimeError: shape mismatch` 在 `_load_weights_and_validate` | 用了 `CheckpointWeightLoader` 而 `action_dim` 跟 base 不一致，换回 `PartialCheckpointWeightLoader`。 |

---

## 涉及的代码文件（一图速览）

```
src/openpi/policies/g1_sonic_policy.py        # G1SonicInputs / G1SonicOutputs
src/openpi/training/weight_loaders.py         # PartialCheckpointWeightLoader
src/openpi/training/config.py                 # LeRobotG1SonicDataConfig + pi05_g1_sonic TrainConfig
docs/finetune_g1_sonic.md                     # 本文档
```

数据 / 资源：

```
/path/to/datasets/2026-04-28-take-walk-and-put/        # LeRobot 数据集
/path/to/pi05_base/params/                             # base ckpt
/path/to/openpi/assets/pi05_g1_sonic/<repo_id>/        # compute_norm_stats 输出（自动建）
/path/to/openpi/checkpoints/pi05_g1_sonic/<exp>/       # 训练 ckpt（自动建）
~/.cache/openpi/big_vision/paligemma_tokenizer.model    # PaliGemma 分词器
```
