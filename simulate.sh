#!/bin/sh
seed=99
acc=3000

for S in A B C; do
    for w in 1 2 4 8; do
        # for max_s in 1 2 3; do
        ./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
    done
done

for S in A B C; do
    for A in fixed simple1; do
        ./simulate.py -S $S -A $A --seed $seed -a $acc $@
    done
    for w in 4; do
        for max_s in 1 2 3; do
            ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --max_sleep $max_s $@
        done
        ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --no_interf $@
    done
done
