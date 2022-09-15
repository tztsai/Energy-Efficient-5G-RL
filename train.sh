#!/bin/sh
seed_max=1
log_level="NOTICE"

traffic_type="A"
accelerate=600
n_training_threads=4
n_rollout_threads=42
num_env_steps=$((50400 * 100))  # steps_per_episode * episodes
experiment="check"

algo="mappo"
gain=0.01
lr=7e-4
critic_lr=7e-4
ppo_epoch=15
num_mini_batch=1

w_pc=1.0
w_drop=1.0
w_delay=2.0

log_interval=1

wandb_user="tcai7"
wandb_api_key="e3662fa8db0f243936c7514a1d0c69f2374ce721"

echo "algo is ${algo}, traffic type is ${traffic_type}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train.py $@ --algorithm_name ${algo} --experiment_name ${experiment} --traffic_type ${traffic_type} --accelerate ${accelerate} --seed ${seed} --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --user_name ${wandb_user} --log_level ${log_level} --log_interval ${log_interval} --w_pc ${w_pc} --w_drop ${w_drop} --w_delay ${w_delay} #--use_eval --eval_interval ${eval_interval} --n_eval_rollout_threads ${n_eval_rollout_threads}
done
