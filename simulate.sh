#!/bin/sh
for S in A B C; do
    for A in mappo fixed simple simple1 simple2 sleepy fixed; do
        ./simulate.py -S $S -A $A --seed 42 #-V 40
    done
done
