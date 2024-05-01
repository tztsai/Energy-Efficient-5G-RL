# %%
import numpy as np
import pandas as pd

stats_file = '../results/p-r.txt'
a = np.loadtxt(stats_file)
df = pd.DataFrame(a, columns=['power', 'sinr', 'rate'])
df

# %%
df.plot(kind='scatter', x='power', y='rate')

df.plot(kind='scatter', x='power', y='sinr', logy=True)
