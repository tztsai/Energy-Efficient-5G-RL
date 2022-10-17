for S in A B C; do
    for A in simple fixed mappo; do
        ./simulate.py -S $S -A $A --seed 42
    done
done