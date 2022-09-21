# %%
import numpy as np
import pandas as pd

(df := pd.read_csv('reward.csv'))
df

# %%
def parse_np_series(s):
    l = [np.fromstring(a[1:-1], sep=' ') for a in s]
    return pd.DataFrame(l, index=s.index)

drop_df = pd.concat([parse_np_series(df.pop(k)) for k in ['drop_ratios', 'drop_counts']], axis=1, keys=['ratio', 'count'])
drop_df['weighted_percent'] = df['drop']
drop_df['ratio'].plot()
drop_df.describe()

# %%
drop_df = drop_df.sort_values(by='weighted_percent', ascending=False)

drop_df.head(10)

# %%
delay_df = parse_np_series(df.pop('ue_delays'))
delay_df.plot()
delay_df['weighted'] = df['delay']
delay_df.describe()

# %%
df['pc'].plot()
df['pc'].describe()

# %%
penalty = -df['reward']
penalty.plot()
penalty.describe()

# %%
df.corr()

# %%
