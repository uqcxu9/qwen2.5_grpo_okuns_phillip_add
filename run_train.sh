#!/bin/bash
# 保守版 GRPO 训练 - 100 步测试

cd /workspace/QWEN2.5_42_7b_main

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
