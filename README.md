# Qwen2.5-7B GRPO Training (100 步成功版)

经济决策 Agent 的 GRPO 强化学习训练项目。

## 重要：100 步训练成功！

使用**保守参数**成功完成 100 步训练：
- ✅ JSON 解析成功率: 100%
- ✅ reward=-1.0 比例: 0%
- ✅ Work std: 0.196 (有多样性)
- ✅ Consumption std: 0.106
- ✅ LoRA mean reward > Base mean reward

## 目录结构

```
├── RL/
│   ├── prepare_verl_data.py    # 数据准备脚本
│   ├── reward.py               # Reward 函数 (严格格式检查版)
│   ├── retrieval.py            # Few-shot 检索
│   ├── good_decision.py        # 好决策识别
│   └── verl_dataset/           # 训练/验证数据 (需重新生成)
├── checkpoints/
│   └── global_step_100/
│       └── actor/lora_adapter/ # LoRA checkpoint (78MB)
├── run_train.sh                # 训练脚本
├── analyze_v2.py               # 分析脚本
└── config.yaml                 # 训练配置
```

## 从 Step 100 恢复训练到 1000

### 1. 环境准备

```bash
pip install verl vllm transformers accelerate peft torch ray[default] pandas numpy
```

### 2. 启动 Ray

```bash
ray start --head --port=6379 --num-cpus=8 --num-gpus=1
```

### 3. 下载 Base Model

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /workspace/models/Qwen2.5-7B-Instruct
```

### 4. 准备数据 (如果 verl_dataset 不存在)

```bash
cd RL
python prepare_verl_data.py
```

### 5. 运行训练 (100 → 1000)

```bash
bash run_train.sh
```

或手动运行：

```bash
python -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.address="172.17.0.3:6379" \
  data.train_files=/workspace/QWEN2.5_42_7b_main/RL/verl_dataset/train.parquet \
  data.val_files=/workspace/QWEN2.5_42_7b_main/RL/verl_dataset/val.parquet \
  data.train_batch_size=4 \
  data.val_batch_size=1 \
  data.max_prompt_length=1536 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path=/workspace/models/Qwen2.5-7B-Instruct \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=false \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0.02 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.prompt_length=1536 \
  actor_rollout_ref.rollout.response_length=128 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  custom_reward_function.path=/workspace/QWEN2.5_42_7b_main/RL/reward.py \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=1000 \
  trainer.save_freq=1000 \
  trainer.test_freq=1000 \
  trainer.val_before_train=false \
  trainer.default_local_dir=/workspace/QWEN2.5_42_7b_main/checkpoints \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node=1 \
  trainer.logger=[console]
```

## 关键参数说明 (保守版 - 已验证有效)

| 参数 | 值 | 说明 |
|------|-----|------|
| rollout.n | 2 | 每个 prompt 生成的 response 数 |
| rollout.temperature | 0.7 | 生成温度（保守）|
| entropy_coeff | 0.02 | Entropy 系数（保守）|
| kl_ctrl.kl_coef | 0.005 | KL 惩罚系数 |
| train_batch_size | 4 | 训练 batch 大小 |
| resume_mode | auto | 自动从最新 checkpoint 恢复 |

## 100 步训练结果

### 分布统计

| 指标 | Base | LoRA |
|------|------|------|
| Work mean | 0.830 | 0.825 |
| Work std | 0.194 | 0.194 |
| Consumption mean | 0.694 | 0.684 |
| Consumption std | 0.106 | 0.105 |
| Reward mean | -0.067 | -0.057 |

### 关键发现

- LoRA 比 Base reward 高 +0.01
- JSON 解析 100% 成功
- 无 mode collapse (work/consumption 有多样性)
- 无训练错误

## Checkpoint 说明

- `checkpoints/global_step_100/actor/lora_adapter/`: LoRA 权重 (78MB)
- 可与 Qwen2.5-7B-Instruct base model 合并使用

## 验证模型

```bash
python analyze_v2.py
```
