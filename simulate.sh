#!/bin/sh
seed=99
acc=3000

for S in A B C; do
    for w in 4 8; do
        ./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
    done
    for A in fixed simple simple1 simple2; do
        ./simulate.py -S $S -A $A --seed $seed -a $acc $@
    done
done
