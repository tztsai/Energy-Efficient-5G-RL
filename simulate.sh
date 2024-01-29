#!/bin/sh
acc=3000

for seed in 1 2 3; do
for S in A B C; do
    for w in 1 2 4 8; do
        ./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
    done
    w=4
    ./simulate.py -S $S -A fixed --seed $seed -a $acc $@
    ./simulate.py -S $S -A simple1 --seed $seed -a $acc $@
    ./simulate.py -S $S -A simple1 --no_offload --seed $seed -a $acc $@
    for max_s in 1 2; do  # max depth of sleep
        ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --max_sleep $max_s $@
    done
    ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --no_interf $@
    ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --no_offload $@
    ./simulate.py -S $S -A dqn --seed $seed -a $acc --use_wandb --run_version run28 $@
    ./simulate.py -S $S -A dqn --w_qos 4 --seed $seed -a $acc --use_wandb --run_version run23 --no_offload $@
done
done