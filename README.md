# Slime-OSWorld

[![HF Model](https://img.shields.io/badge/HF-SFT%20Checkpoint-yellow)](https://huggingface.co/Jarrodbarnes/osworld-vlm-sft-step25)
[![HF Dataset](https://img.shields.io/badge/HF-Training%20Data-blue)](https://huggingface.co/datasets/Jarrodbarnes/osworld-train-v1)

Train multi-turn computer-use agents on OSWorld using slime's FSDP backend.

![Architecture](public/architecture.jpeg)

## Overview

This cookbook demonstrates GSPO training with experience replay for sparse reward environments. The pipeline:

```
SFT Warmup -> GSPO with Replay -> Evaluation
```

OSWorld tasks have sparse rewards (0 or 1 for task completion). GSPO requires within-prompt variance for advantage computation. When all rollouts fail, advantages collapse. Experience replay injects successful trajectories when online samples all fail.

**Note**: On-policy distillation (UI-TARS-2, arXiv:2509.02544) achieves stronger results (47.5% on OSWorld) than off-policy approaches. This cookbook uses off-policy replay as a reproducible baseline.

## Installation

```bash
pip install -e .

# With training dependencies
pip install -e ".[train]"
```

## Quick Start (4x H100)

```bash
# Container setup
docker pull slimerl/slime:latest
docker run --gpus all --ipc=host --shm-size=16g \
  -v /ephemeral:/ephemeral -it slimerl/slime:latest /bin/bash

# Inside container
git clone https://github.com/THUDM/slime.git /root/slime && cd /root/slime && pip install -e .
git clone https://github.com/jbarnes850/Slime-OSWorld.git /root/Slime-OSWorld && cd /root/Slime-OSWorld && pip install -e .

# Download checkpoints and datasets
huggingface-cli download Jarrodbarnes/osworld-vlm-sft-step25 --local-dir /ephemeral/osworld-vlm-sft-step25-hf
huggingface-cli download Jarrodbarnes/osworld-train-v1 --repo-type dataset --local-dir /ephemeral/osworld_train

# Start OSWorld server on host (requires KVM, see Environment Setup)
# Then run training
export OSWORLD_SERVER_URL=http://172.17.0.1:8100
./scripts/train_grpo.sh
```

## Environment Setup

OSWorld requires KVM for VM acceleration. The training container and OSWorld server run separately due to torch version conflicts (`desktop-env` pins torch 2.5.1, sglang requires 2.9+).

```
Host (osworld_venv)              Container (slime_train)
--------------------             -------------------------
torch 2.5.1                      torch 2.9.1
desktop-env + KVM                sglang + GSPO
osworld_server.py :8100  <---->  HTTPRemoteDesktopEnv
```

### Host Setup

```bash
python3 -m venv ~/osworld_venv && source ~/osworld_venv/bin/activate
pip install desktop-env
git clone https://github.com/xlang-ai/OSWorld.git ~/OSWorld && cd ~/OSWorld
git clone https://github.com/jbarnes850/Slime-OSWorld.git ~/Slime-OSWorld
sudo -E ~/osworld_venv/bin/python quickstart.py --provider_name docker  # Downloads 11.4GB VM
sudo chown -R "$USER:$USER" ~/OSWorld

# Start server (run in tmux)
cd ~/OSWorld
sudo -E ~/osworld_venv/bin/python ~/Slime-OSWorld/tools/osworld_env_server.py --port 8100
```

### Parallel Rollouts

Scale with multiple servers:

```bash
# On host: start servers on different ports
for port in 8100 8101 8102 8103; do
  sudo -E ~/osworld_venv/bin/python ~/Slime-OSWorld/tools/osworld_env_server.py --port $port &
done

# In container: comma-separated URLs
export OSWORLD_SERVER_URL="http://172.17.0.1:8100,http://172.17.0.1:8101,http://172.17.0.1:8102,http://172.17.0.1:8103"
```

## Training

### GSPO with Experience Replay

Key configuration in `scripts/train_grpo.sh`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--n-samples-per-prompt` | 4 | Within-prompt variance |
| `--rollout-temperature` | 0.8 -> 0.5 | Exploration -> stability |
| `OSWORLD_REPLAY_BUFFER` | osworld_replay_train.jsonl | Replay injection |
| `--rollout-function-path` | slime_osworld.rollout.generate_rollout | Custom batch rollout |

Recommended two-phase schedule:

```bash
# Phase 1: exploration (higher temperature)
SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.8 bash scripts/train_grpo.sh

# Phase 2: stability (reduce repeated actions)
SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.5 bash scripts/train_grpo.sh
```

### Reward Shaping

```
shaped_reward = task_reward + 0.3 * partial_score - penalties
```

**Partial scores**: action parsing, execution, a11y grounding, efficiency, screen changes

**Penalties**: repetition, excessive waits, fallback parsing

Enable reward debugging:

```bash
OSWORLD_REWARD_DEBUG_LIMIT=10 bash scripts/train_grpo.sh
```

### VLM Training Constraints

Context Parallel (CP) must be 1 for VLM training. ring_flash_attn requires causal attention, but VLM vision encoders use bidirectional attention. Setting CP>1 causes assertion failures.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OSWORLD_SERVER_URL` | `http://localhost:8100` | OSWorld HTTP server(s), comma-separated for parallel |
| `OSWORLD_REPLAY_BUFFER` | - | Path to replay buffer JSONL |
| `OSWORLD_REPLAY_THRESHOLD` | `0.5` | Success threshold for replay injection |
| `OSWORLD_TRAIN_TRUNCATE_TURNS` | `8` | Max turns per trajectory |
| `OSWORLD_MAX_TURNS` | - | Hard limit on trajectory length |
| `OSWORLD_SCREEN_DIFF_THRESHOLD` | `0.005` | Screen change detection threshold |
| `OSWORLD_TIMEOUT` | `900` | Episode timeout in seconds |
| `OSWORLD_REWARD_ALPHA` | `0.3` | Partial reward weight |
| `OSWORLD_REWARD_DEBUG_LIMIT` | `5` | Max reward debug prints |

## Evaluation

Use OSWorld's native evaluation pipeline:

```bash
cd ~/OSWorld
python run.py --observation_type screenshot --model your_model --result_dir results/
```

Or the lightweight eval script:

```bash
python tools/eval.py \
  --checkpoint /path/to/model \
  --tasks /path/to/tasks.parquet \
  --max-turns 12
```

## Project Structure

```
slime_osworld/
  __init__.py
  rollout.py          # Multi-turn VLM rollout + replay injection
  reward.py           # Hybrid reward computation
  env.py              # OSWorld wrapper + HTTP client
  replay_buffer.py    # Experience replay buffer
scripts/
  train_grpo.sh       # Training launcher
tools/
  osworld_env_server.py   # HTTP server (runs on HOST)
  build_union_datasets.py
  eval.py
  prepare_tasks.py
configs/
  grpo_config.yaml
  .env.template
```

## Artifacts

### Checkpoints

- [Jarrodbarnes/osworld-vlm-sft-step25](https://huggingface.co/Jarrodbarnes/osworld-vlm-sft-step25) - SFT warmup (required to start GSPO)

### Datasets

- [Jarrodbarnes/osworld-train-v1](https://huggingface.co/datasets/Jarrodbarnes/osworld-train-v1) - Training data
  - `osworld_tasks_train.parquet` - 66 Ubuntu tasks with replay overlap
  - `osworld_replay_train.jsonl` - Expanded replay buffer

### Logs

- W&B: `jbarnes850-near-protocol/osworld-grpo`

## Rebuild from Sources

If you need to rebuild union artifacts from raw datasets:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /root/OSWorld
python tools/build_union_datasets.py \
  --hf-root /ephemeral \
  --osworld-repo /root/OSWorld \
  --output-dir /ephemeral/osworld_train
```

## References

- [OSWorld Benchmark](https://os-world.github.io/) - Desktop automation evaluation
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) - SOTA on OSWorld
- [UI-TARS-2](https://arxiv.org/abs/2509.02544) - On-policy distillation for GUI agents
- [slime](https://github.com/THUDM/slime) - RL training framework

## License

Apache-2.0
