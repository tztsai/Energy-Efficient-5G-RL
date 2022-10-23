#!/bin/sh
for S in A B C; do
    for A in mappo fixed simple simple1 simple2; do
        ./simulate.py -S $S -A $A -a 60000 #-V 40
    done
done
