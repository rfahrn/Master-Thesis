#!/bin/bash
#SBATCH --job-name=grpo-qwen
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=05:00:00
#SBATCH --output=job_outputs/%x_%.out
#SBATCH --exclusive

srun --environment=verl --gpus-per-task=4 bash -c "
export SLURM_CPUS_PER_TASK=288
export SLURM_GPUS=4

unset ROCR_VISIBLE_DEVICES
set -x
ENGINE=\${1:-vllm}

# CRITICAL: Set PYTHONPATH to include verl
export PYTHONPATH=...
echo \"PYTHONPATH is set to: \$PYTHONPATH\"

export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/

export WORK_DIR=...
export CUSTOM_REWARD_DIR=...
export DATA_DIR=...
export SAVE_PATH=...
cd \$WORK_DIR

mkdir -p \"\$SAVE_PATH\"

export WANDB_RESUME=allow
export WANDB_ENTITY=\"krauthammerlab\"
export WANDB_API_KEY=...
WANDB_DIR=\${SAVE_PATH}

# Ray cluster setup
NODES=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | tr \"\n\" \" \" | xargs)
NODES_ARR=(\$NODES)
MASTER_NODE=\${NODES_ARR[0]}

export MASTER_NODE_IP=\$(srun --overlap --nodes=1 --ntasks=1 -w \"\$MASTER_NODE\" hostname --ip-address)

echo \"Master node: \${MASTER_NODE} (\${MASTER_NODE_IP})\"

export PORT=\$((6542 + \$SLURM_JOB_ID % 1000))
export RAY_ADDRESS=\"\${MASTER_NODE_IP}:\${PORT}\"
echo \"Ray address: \${RAY_ADDRESS}\"

# Start Ray cluster
if [[ \$SLURM_PROCID -eq 0 ]]; then
  echo \"[Rank 0] Starting Ray HEAD node\"
  ray start --head --node-ip-address=\$MASTER_NODE_IP --port=\$PORT \\
    --num-cpus=\$SLURM_CPUS_PER_TASK --num-gpus=\$SLURM_GPUS --block &
else
  echo \"[Rank \${SLURM_PROCID}] Starting Ray WORKER node\"
  ray start --address=\$RAY_ADDRESS --num-cpus=\$SLURM_CPUS_PER_TASK \\
    --num-gpus=\$SLURM_GPUS --block &
fi

sleep 30
ray status || echo \"Ray status check failed (may be normal)\"

if [[ \$SLURM_PROCID -eq 0 ]]; then
  echo \"=========================================\"

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.norm_adv_by_std_in_grpo=True \\
    algorithm.use_kl_in_reward=False \\
    \\
    data.train_files=\$DATA_DIR/train.parquet \\
    data.val_files=\$DATA_DIR/val.parquet \\
    data.train_batch_size=256  \\
    data.val_batch_size=1\\
    data.max_prompt_length=768 \\
    data.max_response_length=2048 \\
    data.image_key=images \\
    \\
    actor_rollout_ref.model.path=.../qwen2.5VL_full \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.01 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.loss_agg_mode=token-mean \\
    \\
    actor_rollout_ref.rollout.name=\$ENGINE \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    \\
    custom_reward_function.path=\$CUSTOM_REWARD_DIR/custom_rewards/fbeta_reward.py \\  # example fbeta reward
    custom_reward_function.name=compute_score \\
    \\
    trainer.logger='[\"console\",\"wandb\"]' \\
    trainer.project_name=grpo_medical_grounding \\
    trainer.experiment_name=..\\  # the names of the experiment 
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=8 \\
    trainer.save_freq=10 \\
    trainer.test_freq=5 \\
    trainer.val_before_train=True \\
    trainer.default_local_dir=\${SAVE_PATH} \\
    trainer.total_epochs=1

  echo \"=========================================\"
  echo \"Training completed!\"
  echo \"=========================================\"

  ray stop --force || true
  exit 0

else
  wait
fi
"
