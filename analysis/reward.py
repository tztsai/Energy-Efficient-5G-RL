# %%
import numpy as np
import pandas as pd

(df := pd.read_csv('reward.csv'))
df

# %%
drop_df = pd.DataFrame(np.hstack([[np.fromstring(s[1:-1], sep=' ') for s in df.pop(k)] for k in ['v_drop', 'n_drop']]), index=df.index, columns = pd.MultiIndex.from_product([['r', 'n'], range(3)], names=['', 'app']))
drop_df['v'] = df['drop']

# %%
drop_df = drop_df.sort_values(by='v', ascending=False)

drop_df.head(10)

# %%
drop_df.describe()

# %%
delay_df = pd.DataFrame(np.vstack([np.fromstring(s[1:-1], sep=' ') for s in df.pop('v_delay')]))
delay_df[''] = df['delay']

delay_df.describe()

# %%
df['pc'] /= 1e3
df['pc'].describe()

# %%
df['penalty'].describe()

# %%
df.corr()

# %%
df.plot(kind='scatter', x='power', y='rate')

df.plot(kind='scatter', x='power', y='sinr', logy=True)
