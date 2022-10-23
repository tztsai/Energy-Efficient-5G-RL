import os
import json
from pathlib import Path

run_results_dir = Path('results/MultiCellNetwork')
run_dirs = run_results_dir.glob('*/*/*/wandb/run*/')

for d in run_dirs:
    with open(d / 'files/wandb-summary.json') as f:
        s = json.load(f)
    if 'reward_mean' not in s:
        print(d)
        