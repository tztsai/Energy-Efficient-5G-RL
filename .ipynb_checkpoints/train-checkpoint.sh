#!/bin/sh
seed_max=1
log_level="NOTICE"

traffic_type="A"
accel_rate=600
n_training_threads=4
n_rollout_threads=42
num_env_steps=3024000

algo="mappo"
gain=0.01
lr=7e-4
critic_lr=7e-4
ppo_epoch=10

w_pc=1.0
w_drop=0.08

log_interval=1

wandb_api_key="e3662fa8db0f243936c7514a1d0c69f2374ce721"

echo "algo is ${algo}, traffic type is ${traffic_type}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train.py $@ --algorithm_name ${algo} --traffic_type ${traffic_type} --accel_rate ${accel_rate} --seed ${seed} --n_training_threads ${n_training_threads} --n_rollout_threads ${n_rollout_threads} --num_mini_batch 1 --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} --gain ${gain} --lr ${lr} --critic_lr ${critic_lr} --user_name "tcai7" --log_level ${log_level} --log_interval ${log_interval} #--use_eval --eval_interval ${eval_interval} --n_eval_rollout_threads ${n_eval_rollout_threads}
done
